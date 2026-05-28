import random
import math
import numpy as np
import torch
from src.ADCRezero.data.base import RezeroBaseDataset
from src.ADCRezero.data.make_isotropic import generate_diffuse_noise

class RezeroOnTheFlyDataset(RezeroBaseDataset):
    """
    On-the-fly ReZero dataset.
    Input:
        args.speech_dir  (str): Directory containing speech audio files.
        args.noise_dir   (str): Directory containing noise audio files.
        args.region_type (str): Type of region for query ('angular', 'conical', or 'spherical').
        args.mic_arch    (str): Type of microphone array ('circular' or 'linear').
        args.limit_mic_z (bool): Whether to limit the z-coordinate of microphones.
        args.elevation_limit (bool): Whether to limit the elevation of speech sources.
        args.first_positioning (str): First positioning ('mic' or 'speaker').
        args.decision_query_region (str): Method for deciding query region if args.first_positioning == 'mic' and args.region_type == 'conical' ('no_limit', 'angle_limit', or 'distance_limit').
        args.mic_in_center        (bool): Whether to place the microphone array at the room center.
        args.room_size (list): Fixed room size [x, y, z].
        args.infinity_room (bool): Whether to use an infinite room for simulation if args.first_positioning == 'mic'.
        args.no_noise (bool): Whether to exclude noise sources.
        args.diffuse_noise_prob (float): Probability of adding diffuse noise when sampling noise sources.
        args.query2D (bool): Consider the query area in a two-dimensional plane.
        args.config (str): Path to the configuration file.
        args.mode (str): Mode of the dataset ('train', 'val', or 'test').
        args.cpuram (bool): Whether to load audio files into RAM.
    Returns:
        mix         (Tensor[B, C, T]) # fixed length
        theta_l     (Tensor[B])       # lower bound of angular query region in degrees
        theta_h     (Tensor[B])       # upper bound of angular query region in degrees
        d_query     (Tensor[B])       # distance threshold sampled from config
        Q           (Tensor[B])       # number of sources within query region
        target      (Tensor[B, C, T]) # synthesized target query waveforms
    """
    def __init__(self, args):
        super().__init__(args)
        
    def __len__(self):
        return self.iterations

    def __getitem__(self, idx):
        # 1) Pre-sample Q
        if not hasattr(self, 'Q_val'):
            if not self.args.one_speaker:
                self.Q_val = random.choices([0, 1, 2], weights=self.split_Q)[0]
            else:
                self.Q_val = random.choice([0, 1])
        
        # 2) Sample number of speech/noise sources
        if not hasattr(self, 'n_s'):
            if not self.args.one_speaker:
                if self.Q_val == 1:
                    self.n_s = random.choices([1, 2], weights=self.dataset_cfg['Q1_distribution'][self.args.mode])[0]
                else:
                    self.n_s = random.randint(max(1, self.Q_val), self.max_speech)
            else:
                self.n_s = 1
        if not hasattr(self, 'n_n'):
            if not self.args.no_noise:
                self.n_n  = random.randint(*self.dataset_cfg['noise_per_mix'])
            else:
                self.n_n = 0

        # 3) Base room configuration
        if not hasattr(self, 'room'):
            min_size = self.dataset_cfg['room_size_m']['min']
            max_size = self.dataset_cfg['room_size_m']['max']
            if self.args.room_size is None:
                if not self.args.infinity_room:
                    room = [random.uniform(lo, hi) for lo, hi in zip(min_size, max_size)]
                    self.room = np.array(room, dtype=np.float32)
                else:
                    self.room = np.array(max_size, dtype=np.float32)
            else:
                self.room = np.array(self.args.room_size, dtype=np.float32)
                
        if not hasattr(self, 'rt60'):
            self.rt60 = random.uniform(*self.dataset_cfg['t60_s_range'])
        
        # 4) Determine query region, mic and speech position
        if self.args.first_positioning == 'mic':
            if not hasattr(self, 'mic_pos'):
                self.array_pos, self.mic_pos = self.generate_mic_array(self.room, array_num=1)
            arr = self.array_pos[0]
            
            d_query, theta_l_query, theta_h_query = self.decision_query_region_first_mic(arr, self.room, self.min_wall)
            if d_query is None and theta_l_query is None and theta_h_query is None:
                return RezeroOnTheFlyDataset.__getitem__(self, idx)

            speech_pos, region_indices_A, region_indices_D, region_indices_C = self.decision_speaker_pos_first_mic(arr, self.room, self.min_wall, d_query, theta_l_query, theta_h_query, self.Q_val, self.n_s)
            if speech_pos is None:
                return RezeroOnTheFlyDataset.__getitem__(self, idx)
            
        elif self.args.first_positioning == 'speaker': 
            speech_pos, region_indices_A, region_indices_D, region_indices_C, self.array_pos, arr, self.mic_pos, d_query, theta_l_query, theta_h_query = self.decision_query_region_and_mic_pos_first_speaker(self.room, self.min_wall, self.Q_val, self.n_s)
            if speech_pos is None:
                return RezeroOnTheFlyDataset.__getitem__(self, idx)
        
        if self.region_type == 'angular':
            d_query = 10.0
            region_indices = region_indices_A
        elif self.region_type == 'spherical':
            theta_l_query = 0.
            theta_h_query = 360.
            region_indices = region_indices_D
        elif self.region_type == 'conical':
            region_indices = region_indices_C
        
        q_a = len(region_indices_A) if region_indices_A is not None else self.n_s
        q_d = len(region_indices_D) if region_indices_D is not None else self.n_s

        # 5) Sample noise source positions (random)
        use_diffuse = (self.n_n > 0) and (random.random() < self.args.diffuse_noise_prob)
        if use_diffuse:
            n_n_wo_diffuse_noise = self.n_n - 1
        else:
            n_n_wo_diffuse_noise = self.n_n
        
        noise_pos = []
        for _ in range(n_n_wo_diffuse_noise):
            x = random.uniform(self.min_wall, self.room[0] - self.min_wall)
            y = random.uniform(self.min_wall, self.room[1] - self.min_wall)
            z = random.uniform(self.min_wall, self.room[2] - self.min_wall)
            pos = np.array([x, y, z], dtype=np.float32)
            noise_pos.append(pos)
            
        if n_n_wo_diffuse_noise > 0:
            noise_pos = torch.from_numpy(np.stack(noise_pos, axis=0))
        
        # 6) Prepare Audio
        raw_len = int(random.uniform(*self.dataset_cfg['mixture_length_s']) * self.sr)
        if self.args.cpuram:
            selected_speech_wavs = self.load_random_segments(self.n_s, raw_len, paths=None, wavs=self.speech_wavs)
            selected_noise_wavs = self.load_random_segments(n_n_wo_diffuse_noise, raw_len, paths=None, wavs=self.noise_wavs)
        else:
            selected_speech_wavs = self.load_random_segments(self.n_s, raw_len, paths=self.speech_paths, wavs=None)
            selected_noise_wavs = self.load_random_segments(n_n_wo_diffuse_noise, raw_len, paths=self.noise_paths, wavs=None)
        
        if use_diffuse:
            duration_sec = raw_len / float(self.sr)
            target_rms = None
            if n_n_wo_diffuse_noise > 0:
                ref_noise = selected_noise_wavs[0]
                target_rms = float(np.sqrt(np.mean(ref_noise**2)) * 0.05)
            diffuse_np = generate_diffuse_noise(
                mic_pos=self.mic_pos, sr=self.sr, duration_sec=duration_sec,
                field='3d', n_fft=self.n_fft, hop_length=self.hop, window=self.win_type,
                psd_kind=None, target_rms=target_rms
            )  # -> (C, T) float32
            diffuse_noise_torch = torch.from_numpy(diffuse_np)
        else:
            diffuse_noise_torch = torch.zeros((self.n_mics, raw_len), dtype=torch.float32)
        
        # 7) RIR Generation
        rir_t, dir_t, dir_s = self.rir_generation(self.mic_pos, self.rt60, self.room, speech_pos, noise_pos, self.n_s, n_n_wo_diffuse_noise)
        
        # 8) Generate Mix and Target
        if not self.args.infinity_room:
            mix, target = self.generation_mix_and_target(selected_speech_wavs, selected_noise_wavs, self.n_s, n_n_wo_diffuse_noise, rir_t, dir_s, region_indices, use_diffuse, diffuse_noise=diffuse_noise_torch)
        else:
            mix, target = self.generation_mix_and_target(selected_speech_wavs, selected_noise_wavs, self.n_s, n_n_wo_diffuse_noise, dir_t, dir_s, region_indices, use_diffuse, diffuse_noise=diffuse_noise_torch) # 直接音のみ
            
        # 9) Convert to tensors
        theta_l_t = torch.tensor(theta_l_query, dtype=torch.float32)
        theta_h_t = torch.tensor(theta_h_query, dtype=torch.float32)
        d_query_t = torch.tensor(d_query, dtype=torch.float32)
        Q_t = torch.tensor(self.Q_val, dtype=torch.int64)
        
        if torch.isnan(mix).any() or torch.isinf(mix).any() \
            or torch.isnan(target).any() or torch.isinf(target).any() \
            or torch.isnan(speech_pos).any() or torch.isinf(speech_pos).any():
            print("Detected nan or inf value!")
            return RezeroOnTheFlyDataset.__getitem__(self, idx)

        if self.args.dataset_generation: # generation freeze dataset
            return {
                'mix': mix,
                'theta_l_query': theta_l_t,
                'theta_h_query': theta_h_t,
                'd_query': d_query_t,
                'selected_speech_wavs': selected_speech_wavs,
                'selected_noise_wavs': selected_noise_wavs,
                'target': target,
                'speech_pos': speech_pos,
                'noise_pos': noise_pos,
                'region_indices': torch.tensor(region_indices),
                'q_a': torch.tensor(q_a),
                'q_d': torch.tensor(q_d),
                'use_diffuse': use_diffuse,
                'diffuse_noise': diffuse_noise_torch,
                'n_n_wo_diffuse_noise': n_n_wo_diffuse_noise,
            }
        else: # onthefly (in training)
            for attr in ('n_s', 'n_n', 'Q_val', 'room', 'rt60', 'mic_pos', 'array_pos'):
                if hasattr(self, attr):
                    delattr(self, attr)
            return {
                'mix': mix,
                'theta_l': theta_l_t,
                'theta_h': theta_h_t,
                'd_query': d_query_t,
                'Q': Q_t,
                'target': target,
            }
    
    # マイクの位置を先に決めた場合のクエリ領域の決定
    def decision_query_region_first_mic(self, arr, room, min_wall):
        '''
        Decide query region based on the microphone positions.
        Input:
            arr (np.ndarray): Array position [x, y, z].
            room (np.ndarray): Room size [width, height, depth].        
            min_wall (float): Minimum distance from walls.
        Returns:
            d_query (float): Distance threshold for spherical query.
            theta_l (float): Lower bound of angular query region in degrees.
            theta_h (float): Upper bound of angular query region in degrees.
        '''
        d_min, d_max = self.dataset_cfg['query_region']['distance_threshold_m']
        d_min = max(d_min, self.radius)
        ang_min, ang_max = self.dataset_cfg['query_region']['angular_width_deg']
        
        if self.region_type == 'spherical':
            d_query = random.uniform(d_min, d_max)
            return d_query, None, None
        
        elif self.region_type == 'angular':
            width  = random.uniform(ang_min, ang_max)
            center = random.uniform(0.0, 360.0)
            theta_l_query = (center - width/2) % 360.0
            theta_h_query = (center + width/2) % 360.0
            return None, theta_l_query, theta_h_query
        
        elif self.region_type == 'conical':
            max_trials = 100
            # 制約なしでクエリ領域を決定(現状再帰制限に引っかかる)
            if self.args.decision_query_region == 'no_limit':
                d_query = random.uniform(d_min, d_max)
                width  = random.uniform(ang_min, ang_max)
                center = random.uniform(0.0, 360.0)
                theta_l_query = (center - width/2) % 360.0
                theta_h_query = (center + width/2) % 360.0
                return d_query, theta_l_query, theta_h_query
            
            # 距離から決めて一定のクエリ領域の体積を担保するため角度に制約をつける
            elif self.args.decision_query_region == 'angle_limit':
                d_query = random.uniform(d_min, d_max)
                for i in range(max_trials):
                    width  = random.uniform(ang_min, ang_max)
                    center = random.uniform(0.0, 360.0)
                    theta_l_query = (center - width/2) % 360.0
                    theta_h_query = (center + width/2) % 360.0

                    r_l = math.radians(theta_l_query)
                    r_h = math.radians(theta_h_query)
                    x_l = arr[0] + d_query * math.cos(r_l)
                    y_l = arr[1] + d_query * math.sin(r_l)
                    x_h = arr[0] + d_query * math.cos(r_h) 
                    y_h = arr[1] + d_query * math.sin(r_h)
                    
                    if ((min_wall < x_l < room[0]-min_wall and min_wall < y_l < room[1]-min_wall)
                        or (min_wall < x_h < room[0]-min_wall and min_wall < y_h < room[1]-min_wall)):
                        break
                else:
                    d_query, theta_l_query, theta_h_query = None, None, None
                return d_query, theta_l_query, theta_h_query
            
            # 角度から決めて一定のクエリ領域の体積を担保するため距離に制約をつける
            elif self.args.decision_query_region == 'distance_limit':
                width  = random.uniform(ang_min, ang_max)
                center = random.uniform(0.0, 360.0)
                theta_l_query = (center - width/2) % 360.0
                theta_h_query = (center + width/2) % 360.0
                for i in range(max_trials):
                    d_query = random.uniform(d_min, d_max)

                    r_l = math.radians(theta_l_query)
                    r_h = math.radians(theta_h_query)
                    x_l = arr[0] + d_query * math.cos(r_l)
                    y_l = arr[1] + d_query * math.sin(r_l)
                    x_h = arr[0] + d_query * math.cos(r_h) 
                    y_h = arr[1] + d_query * math.sin(r_h)

                    if ((min_wall < x_l < room[0]-min_wall and min_wall < y_l < room[1]-min_wall)
                        or (min_wall < x_h < room[0]-min_wall and min_wall < y_h < room[1]-min_wall)):
                        break
                else:
                    d_query, theta_l_query, theta_h_query = None, None, None
                return d_query, theta_l_query, theta_h_query
            else:
                raise ValueError("Please specify args.decision_query_region.")

    # マイク位置を先に決めた時のクエリ領域に基づく話者音源位置の決定
    def decision_speaker_pos_first_mic(self, arr, room, min_wall, d_query, theta_l_query, theta_h_query, Q_val, n_s):
        # a) Sample speech source positions in the query region
        speech_pos = []
        region_indices_A = []
        region_indices_D = []
        region_indices_C = []
        max_trials = 100
        if self.region_type == 'angular':
            for i in range(Q_val):
                if theta_l_query <= theta_h_query:
                    az = random.uniform(theta_l_query, theta_h_query)
                else:
                    span1 = 360 - theta_l_query
                    if random.random() < span1 / (span1 + theta_h_query):
                        az = random.uniform(theta_l_query, 360)
                    else:
                        az = random.uniform(0, theta_h_query)
                az_rad = math.radians(az)
                
                for j in range(max_trials):
                    if self.args.query2D:
                        dx = math.cos(az_rad)
                        dy = math.sin(az_rad)
                        d = [dx, dy]
                        
                        t = []
                        for k in range(2):
                            if d[k] < 0:
                                t.append((min_wall - arr[k]) / d[k])
                            elif d[k] > 0:
                                t.append((room[k] - min_wall - arr[k]) / d[k])
                            elif d[k] == 0:
                                t.append(float('inf'))
                        max_dist = min(t)
                        
                        if max_dist == float('inf') or max_dist < self.radius:
                            continue
                        dist = self.radius + (max_dist - self.radius) * random.random() ** (1 / 2)
                        
                        # ─── 極座標 → 直交座標 ───
                        x = arr[0] + dist * math.cos(az_rad)
                        y = arr[1] + dist * math.sin(az_rad)
                        if self.args.elevation_limit:
                            elev_min, elev_max = self.dataset_cfg['speaker_elevation_range']
                            sin_elev_min = math.sin(math.radians(elev_min))
                            sin_elev_max = math.sin(math.radians(elev_max))
                            # 安全な範囲内での一様サンプリング
                            sin_elev_min = np.clip(sin_elev_min, -1.0, 1.0)
                            sin_elev_max = np.clip(sin_elev_max, -1.0, 1.0)
                            elev = math.degrees(math.asin(random.uniform(sin_elev_min, sin_elev_max)))
                            elev_rad = math.radians(elev)
                            # 2Dクエリでも仰角制限を適用してz座標を計算（distを使用）
                            z = arr[2] + dist * math.tan(elev_rad)
                            if not (min_wall < z < room[2] - min_wall):
                                continue
                        else:
                            z = random.uniform(min_wall, room[2] - min_wall)
                    else:
                        if self.args.elevation_limit:
                            elev_min, elev_max = self.dataset_cfg['speaker_elevation_range']
                            sin_elev_min = math.sin(math.radians(elev_min))
                            sin_elev_max = math.sin(math.radians(elev_max))
                            elev = math.degrees(math.asin(random.uniform(sin_elev_min, sin_elev_max)))
                        else:
                            elev = math.degrees(math.asin(random.uniform(-1.0, 1.0)))
                        elev_rad = math.radians(elev)
                        dx = math.cos(elev_rad) * math.cos(az_rad)
                        dy = math.cos(elev_rad) * math.sin(az_rad)
                        dz = math.sin(elev_rad)
                        d = [dx, dy, dz]
                        
                        # マイクアレイ中心arrから方向dに線を伸ばした時の(壁-min_wall)の範囲に収まる最長の距離の計算
                        t = []
                        for k in range(3):
                            if d[k] < 0:
                                t.append((min_wall - arr[k]) / d[k])
                            elif d[k] > 0:
                                t.append((room[k] - min_wall - arr[k]) / d[k])
                            elif d[k] == 0:
                                t.append(float('inf'))
                        max_dist = min(t)
                        
                        if max_dist == float('inf') or max_dist < self.radius:
                            continue
                        dist = self.radius + (max_dist - self.radius) * random.random() ** (1 / 3)
                        
                        # ─── 極座標 → 直交座標 ───
                        x = arr[0] + dist * math.cos(elev_rad) * math.cos(az_rad)
                        y = arr[1] + dist * math.cos(elev_rad) * math.sin(az_rad)
                        z = arr[2] + dist * math.sin(elev_rad)
                    
                    # ─── 壁からの距離条件をチェック ───
                    if self.args.infinity_room:
                        pos = np.array([x, y, z], dtype=np.float32)
                        speech_pos.append(pos)
                        region_indices_A.append(i)
                        break
                    else:
                        if (min_wall < x < room[0] - min_wall and
                            min_wall < y < room[1] - min_wall and
                            min_wall < z < room[2] - min_wall):
                            pos = np.array([x, y, z], dtype=np.float32)
                            speech_pos.append(pos)
                            region_indices_A.append(i)
                            break
                else:
                    return None, None, None, None
        
        elif self.region_type == 'spherical':
            # 領域外話者と領域内話者で角度揃えて距離だけ違うパターンのみ
            if self.args.spherical_fixed_azimuth:
                az = random.uniform(0, 360)
                az_rad = math.radians(az)
            for i in range(Q_val):
                for j in range(max_trials):
                    # 距離サンプリング - 各話者で異なる距離を取得
                    if self.args.query2D:
                        dist = self.radius + (d_query - self.radius) * random.random() ** (1 / 2)
                    else:
                        dist = self.radius + (d_query - self.radius) * random.random() ** (1 / 3)
                    
                    # 角度サンプリング - 各話者で異なる角度を取得
                    if not self.args.spherical_fixed_azimuth:
                        az = random.uniform(0, 360)
                        az_rad = math.radians(az)
                    if self.args.query2D:
                        # ─── 極座標 → 直交座標 ───
                        x = arr[0] + dist * math.cos(az_rad)
                        y = arr[1] + dist * math.sin(az_rad)
                        if self.args.elevation_limit:
                            elev_min, elev_max = self.dataset_cfg['speaker_elevation_range']
                            sin_elev_min = math.sin(math.radians(elev_min))
                            sin_elev_max = math.sin(math.radians(elev_max))
                            # 安全な範囲内での一様サンプリング
                            sin_elev_min = np.clip(sin_elev_min, -1.0, 1.0)
                            sin_elev_max = np.clip(sin_elev_max, -1.0, 1.0)
                            elev = math.degrees(math.asin(random.uniform(sin_elev_min, sin_elev_max)))
                            elev_rad = math.radians(elev)
                            # 2Dクエリでも仰角制限を適用してz座標を計算（distを使用）
                            z = arr[2] + dist * math.tan(elev_rad)
                            if not (min_wall < z < room[2] - min_wall):
                                continue
                        else:
                            z = random.uniform(min_wall, room[2] - min_wall)
                    else:
                        if self.args.elevation_limit:
                            elev_min, elev_max = self.dataset_cfg['speaker_elevation_range']
                            sin_elev_min = math.sin(math.radians(elev_min))
                            sin_elev_max = math.sin(math.radians(elev_max))
                            elev = math.degrees(math.asin(random.uniform(sin_elev_min, sin_elev_max)))
                        else:
                            elev = math.degrees(math.asin(random.uniform(-1.0, 1.0)))
                        elev_rad = math.radians(elev)
                        # ─── 極座標 → 直交座標 ───
                        x = arr[0] + dist * math.cos(elev_rad) * math.cos(az_rad)
                        y = arr[1] + dist * math.cos(elev_rad) * math.sin(az_rad)
                        z = arr[2] + dist * math.sin(elev_rad)
                        
                    # ─── 壁からの距離条件をチェック ───
                    if self.args.infinity_room:
                        pos = np.array([x, y, z], dtype=np.float32)
                        speech_pos.append(pos)
                        region_indices_D.append(i)
                        break
                    else:
                        if (min_wall < x < room[0] - min_wall and
                            min_wall < y < room[1] - min_wall and
                            min_wall < z < room[2] - min_wall):
                            pos = np.array([x, y, z], dtype=np.float32)
                            speech_pos.append(pos)
                            region_indices_D.append(i)
                            break
                else:
                    return None, None, None, None
        
        elif self.region_type == 'conical':
            for i in range(Q_val):
                for j in range(max_trials):
                    # ─── 角度（azimuth）のサンプリング ───
                    if theta_l_query <= theta_h_query:
                        az = random.uniform(theta_l_query, theta_h_query)
                    else:
                        span1 = 360 - theta_l_query
                        if random.random() < span1 / (span1 + theta_h_query):
                            az = random.uniform(theta_l_query, 360)
                        else:
                            az = random.uniform(0, theta_h_query)
                    az_rad = math.radians(az)
                    
                    if self.args.query2D:
                        dist = self.radius + (d_query - self.radius) * random.random() ** (1 / 2)
                        # ─── 極座標 → 直交座標 ───
                        x = arr[0] + dist * math.cos(az_rad)
                        y = arr[1] + dist * math.sin(az_rad)
                        if self.args.elevation_limit:
                            elev_min, elev_max = self.dataset_cfg['speaker_elevation_range']
                            sin_elev_min = math.sin(math.radians(elev_min))
                            sin_elev_max = math.sin(math.radians(elev_max))
                            # 安全な範囲内での一様サンプリング
                            sin_elev_min = np.clip(sin_elev_min, -1.0, 1.0)
                            sin_elev_max = np.clip(sin_elev_max, -1.0, 1.0)
                            elev = math.degrees(math.asin(random.uniform(sin_elev_min, sin_elev_max)))
                            elev_rad = math.radians(elev)
                            # 2Dクエリでも仰角制限を適用してz座標を計算（distを使用）
                            z = arr[2] + dist * math.tan(elev_rad)
                            if not (min_wall < z < room[2] - min_wall):
                                continue
                        else:
                            z = random.uniform(min_wall, room[2] - min_wall)
                    else:
                        # ─── 仰角（elevation）と距離のサンプリング ───
                        if self.args.elevation_limit:
                            elev_min, elev_max = self.dataset_cfg['speaker_elevation_range']
                            sin_elev_min = math.sin(math.radians(elev_min))
                            sin_elev_max = math.sin(math.radians(elev_max))
                            elev = math.degrees(math.asin(random.uniform(sin_elev_min, sin_elev_max)))
                        else:
                            elev = math.degrees(math.asin(random.uniform(-1.0, 1.0)))
                        elev_rad = math.radians(elev)
                        dist = self.radius + (d_query - self.radius) * random.random() ** (1 / 3)

                        # ─── 極座標 → 直交座標 ───
                        x = arr[0] + dist * math.cos(elev_rad) * math.cos(az_rad)
                        y = arr[1] + dist * math.cos(elev_rad) * math.sin(az_rad)
                        z = arr[2] + dist * math.sin(elev_rad)

                    # ─── 壁からの距離条件をチェック ───
                    if self.args.infinity_room:
                        pos = np.array([x, y, z], dtype=np.float32)
                        speech_pos.append(pos)
                        region_indices_C.append(i)
                        break
                    else:
                        if (min_wall < x < room[0] - min_wall and
                            min_wall < y < room[1] - min_wall and
                            min_wall < z < room[2] - min_wall):
                            pos = np.array([x, y, z], dtype=np.float32)
                            speech_pos.append(pos)
                            region_indices_C.append(i)
                            break
                else:
                    return None, None, None, None
        
        # b) Sample noise source positions in outside the query region
        outside_needed = n_s - Q_val
        
        # 領域外話者と領域内話者で角度揃えて距離だけ違うパターン
        if self.args.spherical_fixed_azimuth and self.region_type == 'spherical':
            for j in range(max_trials):
                dx = math.cos(az_rad)
                dy = math.sin(az_rad)
                d = [dx, dy]
                
                t = []
                for k in range(2):
                    if d[k] < 0:
                        t.append((min_wall - arr[k]) / d[k])
                    elif d[k] > 0:
                        t.append((room[k] - min_wall - arr[k]) / d[k])
                    elif d[k] == 0:
                        t.append(float('inf'))
                max_dist = min(t)
                
                if max_dist == float('inf') or max_dist < self.radius:
                    continue
                break
            for _ in range(outside_needed):
                for _ in range(max_trials):
                    # ─── 仰角（elevation）と距離のサンプリング ───
                    if self.args.elevation_limit:
                        elev_min, elev_max = self.dataset_cfg['speaker_elevation_range']
                        sin_elev_min = math.sin(math.radians(elev_min))
                        sin_elev_max = math.sin(math.radians(elev_max))
                        elev = math.degrees(math.asin(random.uniform(sin_elev_min, sin_elev_max)))
                    else:
                        elev = math.degrees(math.asin(random.uniform(-1.0, 1.0)))
                    elev_rad = math.radians(elev)
                    dist = (max_dist - d_query) * random.random() ** (1 / 3)

                    # ─── 極座標 → 直交座標 ───
                    x = arr[0] + dist * math.cos(elev_rad) * math.cos(az_rad)
                    y = arr[1] + dist * math.cos(elev_rad) * math.sin(az_rad)
                    z = arr[2] + dist * math.sin(elev_rad)
                    
                    # ─── 壁からの距離条件をチェック ───
                    if self.args.infinity_room:
                        pos = np.array([x, y, z], dtype=np.float32)
                        speech_pos.append(pos)
                        region_indices_C.append(i)
                        break
                    else:
                        if (min_wall < x < room[0] - min_wall and
                            min_wall < y < room[1] - min_wall and
                            min_wall < z < room[2] - min_wall):
                            pos = np.array([x, y, z], dtype=np.float32)
                            speech_pos.append(pos)
                            region_indices_C.append(i)
                            break
                else:
                    return None, None, None, None
        else:
            for _ in range(outside_needed):
                for _ in range(max_trials):
                    x = random.uniform(min_wall, room[0] - min_wall)
                    y = random.uniform(min_wall, room[1] - min_wall)
                    if self.args.elevation_limit:
                        elev_min, elev_max = self.dataset_cfg['speaker_elevation_range']
                        sin_elev_min = math.sin(math.radians(elev_min))
                        sin_elev_max = math.sin(math.radians(elev_max))
                        # 安全な範囲内での一様サンプリング
                        sin_elev_min = np.clip(sin_elev_min, -1.0, 1.0)
                        sin_elev_max = np.clip(sin_elev_max, -1.0, 1.0)
                        elev = math.degrees(math.asin(random.uniform(sin_elev_min, sin_elev_max)))
                        elev_rad = math.radians(elev)
                        # 正しい水平距離を使用
                        horizontal_dist = math.sqrt((x - arr[0])**2 + (y - arr[1])**2)
                        z = arr[2] + horizontal_dist * math.tan(elev_rad)
                        if not (min_wall < z < room[2] - min_wall):
                            continue
                    else:
                        z = random.uniform(min_wall, room[2] - min_wall)
                    if self.args.query2D:
                        pos = np.array([x, y, z], dtype=np.float32)
                        vec = pos[:2] - arr[:2]
                    else:
                        pos = np.array([x, y, z], dtype=np.float32)
                        vec = pos - arr
                        
                    if self.region_type == 'spherical':
                        dist = np.linalg.norm(vec)
                        in_region = dist <= d_query
                    elif self.region_type == 'angular':
                        az = (math.degrees(math.atan2(vec[1], vec[0])) + 360.0) % 360.0
                        in_region = ((theta_l_query <= theta_h_query and theta_l_query <= az <= theta_h_query) or
                                    (theta_l_query > theta_h_query and (az >= theta_l_query or az <= theta_h_query)))
                    elif self.region_type == 'conical':
                        az = (math.degrees(math.atan2(vec[1], vec[0])) + 360.0) % 360.0
                        dist = np.linalg.norm(vec)
                        in_region = ((theta_l_query <= theta_h_query and theta_l_query <= az <= theta_h_query) or
                                    (theta_l_query > theta_h_query and (az >= theta_l_query or az <= theta_h_query))) \
                                    and (dist <= d_query)
                    if not in_region:
                        speech_pos.append(pos)
                        break
                else:
                    return None, None, None, None
        
        speech_pos = torch.from_numpy(np.stack(speech_pos, axis=0))
        
        if self.region_type == 'angular':
            region_indices_D = list(range(n_s))
            region_indices_C = None
        elif self.region_type == 'spherical':
            region_indices_A = list(range(n_s))
            region_indices_C = None
        elif self.region_type == 'conical':
            for i in range(n_s):
                if self.is_within_angular(speech_pos[i], arr, theta_l_query, theta_h_query):
                    region_indices_A.append(i)
                if self.is_within_distance(speech_pos[i], arr, d_query):
                    region_indices_D.append(i)
        
        return speech_pos, region_indices_A, region_indices_D, region_indices_C
    
    # 話者の位置を先に決めた場合のクエリ領域とマイク位置の決定
    def decision_query_region_and_mic_pos_first_speaker(self, room, min_wall, Q_val, n_s):
        # クエリ領域の決定
        d_min, d_max = self.dataset_cfg['query_region']['distance_threshold_m']
        d_query = random.uniform(d_min, d_max)
        ang_min, ang_max = self.dataset_cfg['query_region']['angular_width_deg']
        width  = random.uniform(ang_min, ang_max)
        center = random.uniform(0.0, 360.0)
        theta_l_query = (center - width/2) % 360.0
        theta_h_query = (center + width/2) % 360.0
        theta_l_rad = math.radians(theta_l_query)
        theta_h_rad = math.radians(theta_h_query)

        # Z 軸制限用にレンジ取得
        if self.args.limit_mic_z:
            mic_z_min, mic_z_max = self.dataset_cfg['mic_z_range']

        # マイク配置試行
        max_trials = 100
        for _ in range(max_trials):
            arr = None
            speech_pos_list = []
            region_indices_A = []
            region_indices_D = []
            region_indices_C = []
            
            # --- Q_val = 0: 一人もクエリ内にいない ---
            if Q_val == 0:
                # arr をランダムに選択
                arr_x = random.uniform(min_wall + self.radius, room[0] - min_wall - self.radius)
                arr_y = random.uniform(min_wall + self.radius, room[1] - min_wall - self.radius)
                if self.args.limit_mic_z:
                    arr_z = random.uniform(mic_z_min, mic_z_max)
                else:
                    arr_z = random.uniform(min_wall + self.radius, room[2] - min_wall - self.radius)
                arr = np.array([arr_x, arr_y, arr_z], dtype=np.float32)
            else:
                # 1人だけランダムに配置
                x = random.uniform(min_wall, room[0] - min_wall)
                y = random.uniform(min_wall, room[1] - min_wall)
                z = random.uniform(min_wall, room[2] - min_wall)
                speech_pos_list.append(np.array([x, y, z], dtype=np.float32))

                # --- Q_val = 1: 1 人だけクエリ内 ---
                sp = speech_pos_list[0]
                if Q_val == 1:
                    if self.region_type == 'spherical':
                        for _ in range(max_trials):
                            if self.args.query2D:
                                # 円内サンプリング
                                r  = d_query * random.random() ** (1 / 2)
                                u = np.random.normal(size=2)
                                u /= np.linalg.norm(u)
                                cand_xy = sp[:2] + u * r
                                cand_z = random.uniform(mic_z_min, mic_z_max) if self.args.limit_mic_z else \
                                    random.uniform(min_wall + self.radius, room[2] - min_wall - self.radius)
                                cand = np.array([cand_xy[0], cand_xy[1], cand_z], dtype=np.float32)
                            else:
                                # 球内サンプリング
                                r = self.radius + (d_query - self.radius) * random.random() ** (1 / 3)
                                u = np.random.normal(size=3)
                                u /= np.linalg.norm(u)
                                cand = sp + u * r
                            # 部屋内チェック
                            if not (min_wall + self.radius <= cand[0] <= room[0] - min_wall - self.radius and
                                    min_wall + self.radius <= cand[1] <= room[1] - min_wall - self.radius and
                                    ((self.args.limit_mic_z and mic_z_min <= cand[2] <= mic_z_max) or
                                    (not self.args.limit_mic_z and min_wall + self.radius <= cand[2] <= room[2] - min_wall - self.radius))):
                                continue
                            arr = cand
                            region_indices_D = [0]
                            break
                        else:
                            return None, None, None, None, None, None, None, None, None, None

                    elif self.region_type == 'angular':
                        for _ in range(max_trials):
                            cand_x = random.uniform(min_wall + self.radius, room[0] - min_wall - self.radius)
                            cand_y = random.uniform(min_wall + self.radius, room[1] - min_wall - self.radius)
                            cand_z = random.uniform(mic_z_min, mic_z_max) if self.args.limit_mic_z else \
                                    random.uniform(min_wall + self.radius, room[2] - min_wall - self.radius)
                            az = (math.atan2(sp[1]-cand_y, sp[0]-cand_x) + 2 * math.pi) % (2 * math.pi)
                            if not RezeroBaseDataset.angle_in_range(az, theta_l_rad, theta_h_rad):
                                continue
                            arr = np.array([cand_x, cand_y, cand_z], dtype=np.float32)
                            region_indices_A = [0]
                            break
                        else:
                            return None, None, None, None, None, None, None, None, None, None

                    elif self.region_type == 'conical':
                        for _ in range(max_trials):
                            if self.args.query2D:
                                # 円内サンプリング
                                r  = d_query * random.random() ** (1 / 2)
                                u = np.random.normal(size=2)
                                u /= np.linalg.norm(u)
                                cand_xy = sp[:2] + u * r
                                cand_z = random.uniform(mic_z_min, mic_z_max) if self.args.limit_mic_z else \
                                    random.uniform(min_wall + self.radius, room[2] - min_wall - self.radius)
                                cand = np.array([cand_xy[0], cand_xy[1], cand_z], dtype=np.float32)
                                dist = np.linalg.norm(sp[:2] - cand[:2])
                            else:
                                # 球内サンプリング
                                r = self.radius + (d_query - self.radius) * random.random() ** (1 / 3)
                                u = np.random.normal(size=3)
                                u /= np.linalg.norm(u)
                                cand = sp + u * r
                                dist = np.linalg.norm(sp - cand)
                            az  = (math.atan2(sp[1]-cand[1], sp[0]-cand[0]) + 2 * math.pi) % (2 * math.pi)
                            if dist > d_query or not RezeroBaseDataset.angle_in_range(az, theta_l_rad, theta_h_rad):
                                continue
                            # 部屋内チェック
                            if not (min_wall + self.radius <= cand[0] <= room[0] - min_wall - self.radius and
                                    min_wall + self.radius <= cand[1] <= room[1] - min_wall - self.radius and
                                    ((self.args.limit_mic_z and mic_z_min <= cand[2] <= mic_z_max) or
                                    (not self.args.limit_mic_z and min_wall + self.radius <= cand[2] <= room[2] - min_wall - self.radius))):
                                continue
                            arr = cand
                            region_indices_C = [0]
                            break
                        else:
                            return None, None, None, None, None, None, None, None, None, None

                # --- Q_val >= 2: 複数人クエリ内 ---
                else:
                    if self.region_type == 'spherical':
                        region_indices_D = list(range(Q_val))
                        for _ in range(Q_val - 1):
                            for _ in range(max_trials):
                                az = random.uniform(0, 2 * math.pi)
                                if self.args.query2D:
                                    r  = d_query * random.random() ** (1 / 2)
                                    x = sp[0] + r * math.cos(az)
                                    y = sp[1] + r * math.sin(az)
                                    if self.args.elevation_limit:
                                        elev_min, elev_max = self.dataset_cfg['speaker_elevation_range']
                                        sin_elev_min = math.sin(math.radians(elev_min))
                                        sin_elev_max = math.sin(math.radians(elev_max))
                                        elev = math.degrees(math.asin(random.uniform(sin_elev_min, sin_elev_max)))
                                        elev_rad = math.radians(elev)
                                        z = sp[2] + r * math.tan(elev_rad)
                                        if not (min_wall < z < room[2] - min_wall):
                                            continue
                                    else:
                                        z = random.uniform(min_wall, room[2] - min_wall)
                                else:
                                    if self.args.elevation_limit:
                                        elev_min, elev_max = self.dataset_cfg['speaker_elevation_range']
                                        sin_elev_min = math.sin(math.radians(elev_min))
                                        sin_elev_max = math.sin(math.radians(elev_max))
                                        elev = math.asin(random.uniform(sin_elev_min, sin_elev_max))
                                    else:
                                        elev = math.asin(random.uniform(-1.0, 1.0))
                                    r  = d_query * random.random() ** (1 / 3)
                                    x = sp[0] + r * math.cos(elev) * math.cos(az)
                                    y = sp[1] + r * math.cos(elev) * math.sin(az)
                                    z = sp[2] + r * math.sin(elev)
                                
                                if (min_wall <= x <= room[0] - min_wall and
                                    min_wall <= y <= room[1] - min_wall and
                                    min_wall <= z <= room[2] - min_wall):
                                    speech_pos_list.append(np.array([x, y, z], dtype=np.float32))
                                    break
                            else:
                                return None, None, None, None, None, None, None, None, None, None
                        for _ in range(max_trials):
                            if self.args.query2D:
                                # 円内サンプリング
                                r  = d_query * random.random() ** (1 / 2)
                                u = np.random.normal(size=2)
                                u /= np.linalg.norm(u)
                                cand_xy = sp[:2] + u * r
                                cand_z = random.uniform(mic_z_min, mic_z_max) if self.args.limit_mic_z else \
                                    random.uniform(min_wall + self.radius, room[2] - min_wall - self.radius)
                                cand = np.array([cand_xy[0], cand_xy[1], cand_z], dtype=np.float32)
                            else:
                                # 球内サンプリング
                                r = self.radius + (d_query - self.radius) * random.random() ** (1 / 3)
                                u = np.random.normal(size=3)
                                u /= np.linalg.norm(u)
                                cand = sp + u * r
                            if not (min_wall + self.radius <= cand[0] <= room[0] - min_wall - self.radius and
                                    min_wall + self.radius <= cand[1] <= room[1] - min_wall - self.radius and
                                    ((self.args.limit_mic_z and mic_z_min <= cand[2] <= mic_z_max) or
                                    (not self.args.limit_mic_z and min_wall + self.radius <= cand[2] <= room[2] - min_wall - self.radius))):
                                continue
                            # 全話者がクエリ領域内に収まるかチェック
                            if self.args.query2D:
                                valid = all(np.linalg.norm(cand[:2] - speech_pos_list[i][:2]) <= d_query for i in region_indices_D)
                            else:
                                valid = all(np.linalg.norm(cand - speech_pos_list[i]) <= d_query for i in region_indices_D)
                            if valid:
                                arr = cand
                                break
                        else:
                            return None, None, None, None, None, None, None, None, None, None

                    elif self.region_type == 'angular':
                        region_indices_A = list(range(Q_val))
                        for _ in range(Q_val - 1):
                            for _ in range(max_trials):
                                az = RezeroBaseDataset.sample_angle_in_range(theta_l_rad, theta_h_rad)
                                if self.args.query2D:
                                    dx = np.cos(az)
                                    dy = np.sin(az)
                                    d = [dx, dy]
                                    t = []
                                    for k in range(2):
                                        if d[k] < 0:
                                            t.append((min_wall - sp[k]) / d[k])
                                        elif d[k] > 0:
                                            t.append((room[k] - min_wall - sp[k]) / d[k])
                                        elif d[k] == 0:
                                            t.append(float('inf'))
                                    max_dist = min(t)
                                    if max_dist == float('inf') or max_dist < self.radius:
                                        continue
                                    r = max_dist * random.random() ** (1 / 2)
                                    x = sp[0] + r * math.cos(az)
                                    y = sp[1] + r * math.sin(az)
                                    z = random.uniform(min_wall, room[2] - min_wall)
                                else:
                                    if self.args.elevation_limit:
                                        elev_min, elev_max = self.dataset_cfg['speaker_elevation_range']
                                        sin_elev_min = math.sin(math.radians(elev_min))
                                        sin_elev_max = math.sin(math.radians(elev_max))
                                        elev = math.asin(random.uniform(sin_elev_min, sin_elev_max))
                                    else:
                                        elev = math.asin(random.uniform(-1.0, 1.0))
                                    dx = np.cos(elev) * np.cos(az)
                                    dy = np.cos(elev) * np.sin(az)
                                    dz = np.sin(elev)
                                    d = [dx, dy, dz]
                                    t = []
                                    for k in range(3):
                                        if d[k] < 0:
                                            t.append((min_wall - sp[k]) / d[k])
                                        elif d[k] > 0:
                                            t.append((room[k] - min_wall - sp[k]) / d[k])
                                        elif d[k] == 0:
                                            t.append(float('inf'))
                                    max_dist = min(t)
                                    if max_dist == float('inf') or max_dist < self.radius:
                                        continue
                                    r = max_dist * random.random() ** (1 / 3)
                                    x = sp[0] + r * math.cos(elev) * math.cos(az)
                                    y = sp[1] + r * math.cos(elev) * math.sin(az)
                                    z = sp[2] + r * math.sin(elev)

                                if (min_wall <= x <= room[0] - min_wall and
                                    min_wall <= y <= room[1] - min_wall and
                                    min_wall <= z <= room[2] - min_wall):
                                    speech_pos_list.append(np.array([x, y, z], dtype=np.float32))
                                    break
                            else:
                                return None, None, None, None, None, None, None, None, None, None
                        for _ in range(max_trials):
                            cand_x = random.uniform(min_wall + self.radius, room[0] - min_wall - self.radius)
                            cand_y = random.uniform(min_wall + self.radius, room[1] - min_wall - self.radius)
                            cand_z = random.uniform(mic_z_min, mic_z_max) if self.args.limit_mic_z else \
                                    random.uniform(min_wall + self.radius, room[2] - min_wall - self.radius)
                            cand =  np.array([cand_x, cand_y, cand_z], dtype=np.float32)
                            phis = [(math.atan2(
                                        speech_pos_list[i][1] - cand[1],
                                        speech_pos_list[i][0] - cand[0]
                                    )  + 2 * math.pi) % (2 * math.pi)for i in region_indices_A]
                            if all(RezeroBaseDataset.angle_in_range(p, theta_l_rad, theta_h_rad) for p in phis):
                                arr = cand
                                break
                        else:
                            return None, None, None, None, None, None, None, None, None, None

                    elif self.region_type == 'conical':
                        region_indices_C = list(range(Q_val))
                        for _ in range(Q_val - 1):
                            for _ in range(max_trials):
                                az = RezeroBaseDataset.sample_angle_in_range(theta_l_rad, theta_h_rad)
                                if self.args.query2D:
                                    r  = d_query * random.random() ** (1 / 2)
                                    x = sp[0] + r * math.cos(az)
                                    y = sp[1] + r * math.sin(az)
                                    z = random.uniform(min_wall, room[2] - min_wall)
                                else:
                                    if self.args.elevation_limit:
                                        elev_min, elev_max = self.dataset_cfg['speaker_elevation_range']
                                        sin_elev_min = math.sin(math.radians(elev_min))
                                        sin_elev_max = math.sin(math.radians(elev_max))
                                        elev = math.asin(random.uniform(sin_elev_min, sin_elev_max))
                                    else:
                                        elev = math.asin(random.uniform(-1.0, 1.0))
                                    r  = d_query * random.random() ** (1 / 3)
                                    x = sp[0] + r * math.cos(elev) * math.cos(az)
                                    y = sp[1] + r * math.cos(elev) * math.sin(az)
                                    z = sp[2] + r * math.sin(elev)
                                
                                if (min_wall <= x <= room[0] - min_wall and
                                    min_wall <= y <= room[1] - min_wall and
                                    min_wall <= z <= room[2] - min_wall):
                                    speech_pos_list.append(np.array([x, y, z], dtype=np.float32))
                                    break
                            else:
                                return None, None, None, None, None, None, None, None, None, None
                        for _ in range(max_trials):
                            if self.args.query2D:
                                az = RezeroBaseDataset.sample_angle_in_range(theta_l_rad, theta_h_rad)
                                u = np.array([math.cos(az), math.sin(az)], dtype=np.float32)
                                r = self.radius + (d_query - self.radius) * random.random() ** (1 / 2)
                                cand_xy = sp[:2] + u * r
                                cand_z = random.uniform(mic_z_min, mic_z_max) if self.args.limit_mic_z else \
                                    random.uniform(min_wall + self.radius, room[2] - min_wall - self.radius)
                                cand = np.array([cand_xy[0], cand_xy[1], cand_z], dtype=np.float32)
                                dist_ok = all(np.linalg.norm(cand[:2] - speech_pos_list[i][:2]) <= d_query for i in region_indices_C)
                            else:
                                az = RezeroBaseDataset.sample_angle_in_range(theta_l_rad, theta_h_rad)
                                cos_elev = random.uniform(-1.0, 1.0)
                                sin_elev = math.sqrt(1.0 - cos_elev**2)
                                u = np.array([
                                    cos_elev * math.cos(az),
                                    cos_elev * math.sin(az),
                                    sin_elev
                                ], dtype=np.float32)
                                r = self.radius + (d_query - self.radius) * random.random() ** (1 / 3)
                                cand = sp + u * r
                                dist_ok = all(np.linalg.norm(cand - speech_pos_list[i]) <= d_query for i in region_indices_C)
                            # 部屋内チェック
                            room_ok = (min_wall + self.radius <= cand[0] <= room[0] - min_wall - self.radius and
                                        min_wall + self.radius <= cand[1] <= room[1] - min_wall - self.radius and
                                        ((self.args.limit_mic_z and mic_z_min <= cand[2] <= mic_z_max) or
                                        (not self.args.limit_mic_z and min_wall + self.radius <= cand[2] <= room[2] - min_wall - self.radius)))
                            if not room_ok:
                                continue
                            phis = [(math.atan2(
                                        speech_pos_list[i][1] - cand[1],
                                        speech_pos_list[i][0] - cand[0]
                                    ) + 2 * math.pi) % (2 * math.pi) for i in region_indices_C]
                            ang_ok = all(RezeroBaseDataset.angle_in_range(p, theta_l_rad, theta_h_rad) for p in phis)
                            if dist_ok and ang_ok:
                                arr = cand
                                break
                        else:
                            return None, None, None, None, None, None, None, None, None, None

            # --- 残りの話者をクエリ領域外に配置 ---
            if n_s > Q_val:
                for _ in range(n_s - Q_val):
                    for _ in range(max_trials):
                        x = random.uniform(min_wall, room[0] - min_wall)
                        y = random.uniform(min_wall, room[1] - min_wall)
                        if self.args.elevation_limit:
                            elev_min, elev_max = self.dataset_cfg['speaker_elevation_range']
                            sin_elev_min = math.sin(math.radians(elev_min))
                            sin_elev_max = math.sin(math.radians(elev_max))
                            elev = math.degrees(math.asin(random.uniform(sin_elev_min, sin_elev_max)))
                            elev_rad = math.radians(elev)
                            horizontal_dist = math.sqrt((x - arr[0])**2 + (y - arr[1])**2)
                            z = arr[2] + horizontal_dist * math.tan(elev_rad)
                            if not (min_wall < z < room[2] - min_wall):
                                continue
                        else:
                            z = random.uniform(min_wall, room[2] - min_wall)
                        cand = np.array([x, y, z], dtype=np.float32)
                        
                        if arr is not None:
                            if self.region_type == 'spherical':
                                if self.args.query2D:
                                    if np.linalg.norm(cand[:2] - arr[:2]) <= d_query:
                                        continue
                                else:
                                    if np.linalg.norm(cand - arr) <= d_query:
                                        continue
                            elif self.region_type == 'angular':
                                az = (math.atan2(cand[1] - arr[1], cand[0] - arr[0]) + 2 * math.pi) % (2 * math.pi)
                                if RezeroBaseDataset.angle_in_range(az, theta_l_rad, theta_h_rad):
                                    continue
                            elif self.region_type == 'conical':
                                if self.args.query2D:
                                    dist = np.linalg.norm(cand[:2] - arr[:2])
                                else:
                                    dist = np.linalg.norm(cand - arr)
                                az = (math.atan2(cand[1] - arr[1], cand[0] - arr[0]) + 2 * math.pi) % (2 * math.pi)
                                if dist <= d_query and RezeroBaseDataset.angle_in_range(az, theta_l_rad, theta_h_rad):
                                    continue
                        speech_pos_list.append(cand)
                        break
                    else:
                        return None, None, None, None, None, None, None, None, None, None

            # --- マイクアレイ内に話者がいるかチェック ---
            if arr is not None and self.args.mic_arch == 'circular':
                if any((sp[0] - arr[0])**2 + (sp[1] - arr[1])**2 < self.radius**2 for sp in speech_pos_list):
                    continue

            # --- マイク素子位置を算出 ---
            if arr is not None:
                mic_pos = []
                if self.args.mic_arch == 'circular':
                    for j in range(self.n_mics):
                        ang = 2 * math.pi * j / self.n_mics
                        mic_pos.append([
                            arr[0] + self.radius * math.cos(ang),
                            arr[1] + self.radius * math.sin(ang),
                            arr[2]
                        ])
                elif self.args.mic_arch == 'linear':
                    for j in range(self.n_mics):
                        offset = (j - (self.n_mics - 1) / 2) * (self.aperture / (self.n_mics - 1))
                        mic_pos.append([arr[0] + offset, arr[1], arr[2]])
                break
        else:
            return None, None, None, None, None, None, None, None, None, None

        speech_pos = torch.tensor(np.stack(speech_pos_list, axis=0), dtype=torch.float32)  # (n_s, 3)
        mic_pos = np.array(mic_pos, dtype=np.float32)
        array_pos  = arr.reshape(1, 3)
        
        if self.region_type == 'angular':
            region_indices_D = list(range(n_s))
            region_indices_C = None
        elif self.region_type == 'spherical':
            region_indices_A = list(range(n_s))
            region_indices_C = None
        elif self.region_type == 'conical':
            for i in range(n_s):
                if self.is_within_angular(speech_pos[i], arr, theta_l_query, theta_h_query):
                    region_indices_A.append(i)
                if self.is_within_distance(speech_pos[i], arr, d_query):
                    region_indices_D.append(i)

        return speech_pos, region_indices_A, region_indices_D, region_indices_C, array_pos, arr, mic_pos, d_query, theta_l_query, theta_h_query