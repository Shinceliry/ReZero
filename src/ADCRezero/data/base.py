import os
import random
import yaml
import math
import numpy as np
import soundfile as sf
import librosa
import torch
from torch.utils.data import Dataset
from itertools import combinations
import glob
import matplotlib.pyplot as plt
import matplotlib
import json
from pathlib import Path
from src.ADCRezero.data.FRAM_RIR.FRAM_RIR import FRAM_RIR
from src.utils.serialize import _to_serializable
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Rezeroデータセットにおける共通の処理
class RezeroBaseDataset(Dataset):
    '''
    Input:
        args.speech_dir            (str): Directory containing speech audio files.
        args.noise_dir             (str): Directory containing noise audio files.
        args.region_type           (str): Type of region for query ('angular', 'conical', or 'spherical').
        args.mic_arch              (str): Type of microphone array ('circular' or 'linear').
        args.limit_mic_z          (bool): Whether to limit the z-coordinate of microphones.
        args.elevation_limit      (bool): Whether to limit the elevation of speech sources.
        args.first_positioning     (str): First positioning ('mic' or 'speaker').
        args.decision_query_region (str): Method for deciding query region if args.first_positioning == 'mic' ('no_limit', 'angle_limit', or 'distance_limit').
        args.mic_in_center        (bool): Whether to place the microphone array at the room center.
        args.infinity_room        (bool): Whether to use an infinite room for simulation.
        args.config                (str): Path to the configuration file.
        args.mode                  (str): Mode of the dataset ('train', 'val', or 'test').
        args.cpuram               (bool): Whether to load audio files into RAM.
    '''
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # Gather audio files
        pattern_wav = os.path.join(args.speech_dir, '**', '*.wav')
        pattern_flac = os.path.join(args.speech_dir, '**', '*.flac')
        self.speech_paths = glob.glob(pattern_wav, recursive=True) + glob.glob(pattern_flac, recursive=True)
        if not self.speech_paths:
            raise RuntimeError('No audio found in speech directory')
        
        pattern_wav_n = os.path.join(args.noise_dir, '**', '*.wav')
        pattern_flac_n = os.path.join(args.noise_dir, '**', '*.flac')
        self.noise_paths = glob.glob(pattern_wav_n, recursive=True) + glob.glob(pattern_flac_n, recursive=True)
        if not self.noise_paths:
            raise RuntimeError('No audio found in noise directory')
        
        print(f"Found {len(self.speech_paths)} speech files and {len(self.noise_paths)} noise files.")
        
        # Load config
        cfg_file = args.config
        with open(cfg_file, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        self.dataset_cfg = cfg['dataset_generation']
        self.audio_cfg = cfg['audio']
        self.iterations = int(cfg['training']['iterations'])
        
        # Audio settings
        self.sr = self.audio_cfg['sample_rate']
        self.n_fft = int(self.sr * self.audio_cfg['stft']['window_size_ms'] / 1000)
        self.hop = int(self.sr * self.audio_cfg['stft']['hop_size_ms'] / 1000)
        self.win_type = self.audio_cfg['stft']['window_type'].lower()
        
        # Fixed output length
        min_len_s, max_len_s = self.dataset_cfg['mixture_length_s']
        self.min_len = int(min_len_s * self.sr)
        self.max_len = int(max_len_s * self.sr)
        
        # Max counts for padding
        min_s, max_s = self.dataset_cfg['speech_per_mix']
        min_n, max_n = self.dataset_cfg['noise_per_mix']
        self.min_speech = min_s
        self.max_speech = max_s
        self.min_noise = min_n
        self.max_noise = max_n
        self.max_total = self.max_speech + self.max_noise
        
        # Microphone setup
        self.mic_cfg = cfg['microphone_array'][args.mic_arch]
        self.n_mics = self.mic_cfg['channels']
        if args.mic_arch == 'circular':
            self.diameter = self.mic_cfg['diameter_m']
            self.radius = self.diameter / 2
        elif args.mic_arch == 'linear':
            self.aperture = self.mic_cfg['aperture_m']
        self.pairs = list(combinations(range(self.n_mics), 2))
        self.min_wall = self.dataset_cfg['min_source_wall_dist_m']
        
        # Region type and split
        self.region_type = args.region_type
        if self.region_type not in ['angular', 'conical', 'spherical']:
            raise ValueError("region_type must be 'angular', 'conical', or 'spherical'")
        self.split_Q = self.dataset_cfg['query_region']['Q_distribution'][self.args.mode][self.region_type]
        
        # CPUメモリに音声データを格納
        if self.args.cpuram:
            print("Loading speech audio files into RAM...")
            speech_wavs = []
            speech_counts = 0
            for path in tqdm(self.speech_paths):
                wav = self.load_segments(path)
                if not self.silent_check(wav):
                    speech_wavs.append(wav)
                    speech_counts += 1
                
            print("Loading Noise audio files into RAM...")
            noise_wavs = []
            noise_counts = 0
            for path in tqdm(self.noise_paths):
                wav = self.load_segments(path)
                if not self.silent_check(wav):
                    noise_wavs.append(wav)
                    noise_counts += 1
                    
            self.speech_wavs = speech_wavs
            self.noise_wavs = noise_wavs
            print(f"Loaded {speech_counts} speech files and {noise_counts} noise files into RAM.")
    
    # マイクアレイの座標を取得する関数
    def generate_mic_array(self, room, array_num=1):
        # マイクアレイ中心
        array_pos = []
        for _ in range(array_num):
            if not self.args.mic_in_center:
                if not self.args.infinity_room:
                    x = random.uniform(self.min_wall + self.radius, room[0] - self.min_wall - self.radius)
                    y = random.uniform(self.min_wall + self.radius, room[1] - self.min_wall - self.radius)
                    if self.args.limit_mic_z:
                        mic_z_min, mic_z_max = self.dataset_cfg['mic_z_range']
                        z = random.uniform(mic_z_min, mic_z_max)
                    else:
                        z = random.uniform(self.min_wall + self.radius, room[2] - self.min_wall - self.radius)
                else:
                    x = room[0] / 2
                    y = room[1] / 2
                    z = room[2] / 2
            else:
                x = room[0] / 2
                y = room[1] / 2
                z = room[2] / 2
            array_pos.append([x, y, z])
        array_pos = np.array(array_pos, dtype=np.float32)
        
        # 各マイクの座標
        mic_pos = []
        for arr in array_pos:
            if self.args.mic_arch == 'circular':
                for j in range(self.n_mics):
                    x_mic = arr[0] + self.radius * math.cos(2 * math.pi * j / self.n_mics)
                    y_mic = arr[1] + self.radius * math.sin(2 * math.pi * j / self.n_mics)
                    mic_pos.append([x_mic, y_mic, arr[2]])
            elif self.args.mic_arch == 'linear':
                for j in range(self.n_mics):
                    offset = (j - (self.n_mics - 1) / 2) * ((self.aperture/ 2) / (self.n_mics - 1))
                    x_mic = arr[0] + offset
                    y_mic = arr[1]
                    mic_pos.append([x_mic, y_mic, arr[2]])
        mic_pos = np.array(mic_pos, dtype=np.float32)
        
        return array_pos, mic_pos
    
    # パスから音声を読み込む関数
    def load_segments(self, path):
        wav, sr = sf.read(path)
        if wav.ndim > 1:
            wav = np.mean(wav, axis=1)
        if sr != self.sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sr)  # [T]
        return wav
    
    # 3秒以上thresh=1e-3の閾値より小さい区間がないか判定
    def silent_check(self, wav, thresh=1e-3, min_silence_time=3.0):
        min_silence_samples = int(self.sr * min_silence_time)
        amp = np.abs(wav)
        silent = amp < thresh
        run = 0
        has_long_silence = False
        for s in silent:
            if s:
                run += 1
                if run >= min_silence_samples:
                    has_long_silence = True
                    break
            else:
                run = 0
        return has_long_silence # True if long silence found, else False
    
    # 音声をランダムに選択して読み込む
    def load_random_segments(self, counts, length, paths=None, wavs=None):
        if paths is not None and wavs is not None:
            raise ValueError("Either paths or wavs should be provided, not both.")
        selected_segs = []
        for _ in range(counts):
            while True:
                if paths is not None:
                    path = random.choice(paths)
                    wav = self.load_segments(path)
                    if self.silent_check(wav):
                        continue
                else:
                    wav = random.choice(wavs)
                
                if len(wav) < length:
                    wav = np.pad(wav, (0, length - len(wav)))
                else:
                    start = random.randint(0, len(wav) - length)
                    wav = wav[start:start + length]
                    
                selected_segs.append(wav)
                break
        if counts == 0:
            selected_segs = [np.zeros(length, dtype=np.float32)]
        return selected_segs

    # RIR generation
    def rir_generation(self, mic_pos, rt60, room, speech_pos, noise_pos, n_s, n_n_wo_diffuse_noise):
        '''
        Input:
            speech_pos, noise_pos: torch.Tensor
        Return:
            rir_t: 全音源のインパルス応答 [C, n_s+n_n_wo_diffuse_noise, R]
            dit_t: 全音源の直接音 [C, n_s+n_n_wo_diffuse_noise, R]
            dir_s: 話者音源の直接音 [C, n_s, R]
        '''
        # Get early reflection window from config
        early_refl_window = tuple(self.dataset_cfg['early_reflection_window_ms'])
        rir_s, dir_s = FRAM_RIR(mic_pos, self.sr, rt60, room, speech_pos.numpy(), num_src=n_s, direct_range=early_refl_window)
        
        # 全音源のRIR生成
        if n_n_wo_diffuse_noise > 0:
            rir_n, dir_n = FRAM_RIR(mic_pos, self.sr, rt60, room, noise_pos.numpy(), num_src=n_n_wo_diffuse_noise, direct_range=early_refl_window)
            rir_t = torch.from_numpy(np.concatenate([rir_s, rir_n], axis=1))
            dir_t = torch.from_numpy(np.concatenate([dir_s, dir_n], axis=1))
        else:
            rir_t = torch.from_numpy(rir_s)
            dir_t = torch.from_numpy(dir_s)
        
        # Pad/trim RIRs
        C, t, R = rir_t.shape
        if t < self.max_total:
            pad_t = torch.zeros(C, self.max_total - t, R)
            rir_t = torch.cat([rir_t, pad_t], dim=1)
        else:
            rir_t = rir_t[:, :self.max_total, :]
        if R < self.max_len:
            pad_R = torch.zeros(C, self.max_total, self.max_len - R)
            rir_t = torch.cat([rir_t, pad_R], dim=2)
        else:
            rir_t = rir_t[:, :, :self.max_len]
        
        # Pad/trim direct sound
        C, t, R = dir_t.shape
        if t < self.max_total:
            pad_t = torch.zeros(C, self.max_total - t, R)
            dir_t = torch.cat([dir_t, pad_t], dim=1)
        else:
            dir_t = dir_t[:, :self.max_total, :]
        if R < self.max_len:
            pad_R = torch.zeros(C, self.max_total, self.max_len - R)
            dir_t = torch.cat([dir_t, pad_R], dim=2)
        else:
            dir_t = dir_t[:, :, :self.max_len]
        
        return rir_t, dir_t, dir_s
    
    # インパルス応答の畳み込みをFFTを用いて行う関数
    def fft_convolve_torch(self, src: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        """
        FFTを用いたインパルス応答の畳み込み
        Args:
            src: [S, 1, T_src] 音源信号
            ir:  [S*M, 1, T_ir] インパルス応答 (各音源に対してM個のマイクのIR)
        Returns:
            conv: [S, M, T_out] 畳み込み結果
        """
        S, _, T_src = src.shape
        SM, _, T_ir = ir.shape
        
        if SM % S != 0 or SM // S != self.n_mics:
            raise ValueError(f"IR数({SM})が音源数({S})とマイク数({self.n_mics})の積になっていません")
        M = SM // S
        
        src_exp = src.unsqueeze(1).expand(-1, M, -1, -1)  # [S, M, 1, T_src]: 音源信号を各マイクに複製
        src_flat = src_exp.reshape(S*M, 1, T_src)         # [S*M, 1, T_src]
        T_out = T_src + T_ir - 1
        N = 1 << ((T_out - 1).bit_length())
        
        # FFT畳み込み: IFFT(FFT(x) * FFT(h))
        src_f = torch.fft.rfft(src_flat, n=N, dim=-1)
        ir_f  = torch.fft.rfft(ir, n=N, dim=-1)
        conv_f = src_f * ir_f
        conv = torch.fft.irfft(conv_f, n=N, dim=-1)
        conv = conv[..., :T_out]                          # [S*M, 1, T_out]
        conv = conv.squeeze(1).reshape(S, M, T_out)       # [S, M, T_out]
        return conv
    
    # Target音声の生成
    def generation_target(self, selected_speech_wavs, region_indices, dir_s, gains=None):
        if region_indices is None or len(region_indices) == 0:
            return torch.zeros(self.n_mics, self.max_len)
        
        if isinstance(dir_s, np.ndarray):
            dir_s = torch.from_numpy(dir_s)
        rir_target = dir_s[:, region_indices, :]                                 # [C, Q_val, R]

        target = []
        selected_speech_wavs = torch.tensor(np.stack(selected_speech_wavs), dtype=torch.float32)
        for i, src_idx in enumerate(region_indices):
            gain = 1.0 if gains is None else float(gains[src_idx])
            wav = (gain * selected_speech_wavs[src_idx]).unsqueeze(0).unsqueeze(0)  # [1,1,T]
            ir  = rir_target[:, i, :].unsqueeze(1)                                  # [C,1,R]
            conv = self.fft_convolve_torch(wav, ir).squeeze(0)                      # [C,T_out]
            target.append(conv)

        target = torch.stack(target, dim=0).sum(dim=0)                             # [C,T]
        target = self.fit_length(target, self.max_len)
        return target
    
    # Mix, Target音声の生成
    def generation_mix_and_target(self, selected_speech_wavs, selected_noise_wavs_wo_diffuse_noise, n_s, n_n_wo_diffuse_noise, rir_t, dir_s, region_indices, use_diffuse, diffuse_noise=None):
        """
        selected_speech_wavs: [n_s, T]
        selected_noise_wavs:  [n_n_wo_diffuse_noise, T]
        rit_t: [C, n_s+n_n_wo_diffuse_noise, R]
        dir_s: [C, n_s, R]
        diffuse_noise: optional [C, T]
        """
        # speech RIR convolution
        src_s = torch.tensor(np.stack(selected_speech_wavs), dtype=torch.float32).unsqueeze(1)  # [S,1,T]
        rir_s = rir_t[:, :n_s, :].permute(1,0,2).reshape(-1,1,self.max_len)                     # [(S*C),1,R]
        conv_s = self.fft_convolve_torch(src_s, rir_s)                                          # [S,C,T_out]
        conv_s = self.fit_length(conv_s, self.max_len)                                          # [S,C,T]

        # noise RIR convolution
        if n_n_wo_diffuse_noise > 0:
            src_n = torch.tensor(np.stack(selected_noise_wavs_wo_diffuse_noise), dtype=torch.float32).unsqueeze(1)         # [N,1,T]
            rir_n = rir_t[:, n_s:n_s+n_n_wo_diffuse_noise, :].permute(1,0,2).reshape(-1,1,self.max_len)   # [(N*C),1,R]
            conv_n = self.fft_convolve_torch(src_n, rir_n)                                                # [N,C,T_out]
            conv_n = self.fit_length(conv_n, self.max_len)                                                # [N,C,T]
            if use_diffuse:
                if isinstance(diffuse_noise, np.ndarray):
                    diffuse_noise = torch.from_numpy(diffuse_noise)
                dn = self.fit_length(diffuse_noise, self.max_len).to(conv_n.dtype)        # [C, T]
                conv_n = torch.cat([conv_n, dn.unsqueeze(0)], dim=0)
        else:
            # Only diffuse noise (Do not use RIR for diffusion noise.)
            if use_diffuse:
                if isinstance(diffuse_noise, np.ndarray):
                    diffuse_noise = torch.from_numpy(diffuse_noise)
                dn = self.fit_length(diffuse_noise, self.max_len).to(conv_s.dtype)        # [C, T]
                conv_n = dn.unsqueeze(0)                                                  # [1, C, T]
            else:
                conv_n = torch.zeros((0, self.n_mics, self.max_len), dtype=conv_s.dtype)  # [0, C, T]

        # Apply SIR for speech signals (from config)
        gains_speech = [1.0 for _ in range(n_s)]
        if n_s > 1:
            sir_lo, sir_hi = self.dataset_cfg['sir_speech_db']
            rms_ref = self.compute_rms(conv_s[0])
            for i in range(1, n_s):
                sir_db = random.uniform(sir_lo, sir_hi)
                rms_i  = self.compute_rms(conv_s[i])
                gain_i = float(rms_ref / (rms_i * (10.0 ** (sir_db / 20.0) + 1e-12)))
                conv_s[i] *= gain_i
                gains_speech[i] = gain_i

        # Apply SIR for noise signals (from config)
        if conv_n.shape[0] > 1:
            sirn_lo, sirn_hi = self.dataset_cfg['sir_noise_db']
            rmsn_ref = self.compute_rms(conv_n[0]) if n_n_wo_diffuse_noise > 0 else None
            for i in range(1, n_n_wo_diffuse_noise):
                sir_db = random.uniform(sirn_lo, sirn_hi)
                rms_i  = self.compute_rms(conv_n[i])
                gain_i = float(rmsn_ref / (rms_i * (10.0 ** (sir_db / 20.0) + 1e-12)))
                conv_n[i] *= gain_i

        # SNR-based scaling of noise (from config)
        if conv_n.shape[0] > 0:
            snr_lo, snr_hi = self.dataset_cfg['snr_db']
            snr_db = random.uniform(snr_lo, snr_hi)
            speech_sum = conv_s.sum(dim=0)
            noise_sum  = conv_n.sum(dim=0)
            rms_s_tot  = self.compute_rms(speech_sum)
            rms_n_tot  = self.compute_rms(noise_sum)
            if rms_n_tot > 0:
                gain_n = float(rms_s_tot / (rms_n_tot * (10.0 ** (snr_db / 20.0))))
                conv_n = conv_n * gain_n
            else:
                conv_n = torch.zeros_like(conv_n)

        # Generation mix
        mix = conv_s.sum(dim=0) + (conv_n.sum(dim=0) if conv_n.shape[0] > 0 else 0.0)  # [C,T]
        mix = self.fit_length(mix, self.max_len)

        # Generaton target
        target = self.generation_target(selected_speech_wavs, region_indices, dir_s, gains=gains_speech)
        
        # RMS normalization
        if self.args.rms_norm:
            mix, rate = self.rms_normalization(mix)
            target = target * rate

        return mix, target
    
    # Mix, Target音声の生成 (複数回推論検証用)
    def generation_mix_and_target_for_multistep_inference(self, selected_speech_wavs, selected_noise_wavs_wo_diffuse_noise, n_s, n_n_wo_diffuse_noise, rir_t, dir_s, use_diffuse, diffuse_noise=None):
        """
        selected_speech_wavs: [n_s, T]
        selected_noise_wavs:  [n_n_wo_diffuse_noise, T]
        rit_t: [C, n_s+n_n_wo_diffuse_noise, R]
        dir_s: [C, n_s, R]
        diffuse_noise: optional [C, T]
        """
        # speech RIR convolution
        src_s = torch.tensor(np.stack(selected_speech_wavs), dtype=torch.float32).unsqueeze(1)  # [S,1,T]
        rir_s = rir_t[:, :n_s, :].permute(1,0,2).reshape(-1,1,self.max_len)                     # [(S*C),1,R]
        conv_s = self.fft_convolve_torch(src_s, rir_s)                                          # [S,C,T_out]
        conv_s = self.fit_length(conv_s, self.max_len)                                          # [S,C,T]

        # noise RIR convolution
        if n_n_wo_diffuse_noise > 0:
            src_n = torch.tensor(np.stack(selected_noise_wavs_wo_diffuse_noise), dtype=torch.float32).unsqueeze(1)         # [N,1,T]
            rir_n = rir_t[:, n_s:n_s+n_n_wo_diffuse_noise, :].permute(1,0,2).reshape(-1,1,self.max_len)   # [(N*C),1,R]
            conv_n = self.fft_convolve_torch(src_n, rir_n)                                                # [N,C,T_out]
            conv_n = self.fit_length(conv_n, self.max_len)                                                # [N,C,T]
            if use_diffuse:
                if isinstance(diffuse_noise, np.ndarray):
                    diffuse_noise = torch.from_numpy(diffuse_noise)
                dn = self.fit_length(diffuse_noise, self.max_len).to(conv_n.dtype)        # [C, T]
                conv_n = torch.cat([conv_n, dn.unsqueeze(0)], dim=0)
        else:
            # Only diffuse noise (Do not use RIR for diffusion noise.)
            if use_diffuse:
                if isinstance(diffuse_noise, np.ndarray):
                    diffuse_noise = torch.from_numpy(diffuse_noise)
                dn = self.fit_length(diffuse_noise, self.max_len).to(conv_s.dtype)        # [C, T]
                conv_n = dn.unsqueeze(0)                                                  # [1, C, T]
            else:
                conv_n = torch.zeros((0, self.n_mics, self.max_len), dtype=conv_s.dtype)  # [0, C, T]

        # Apply SIR for speech signals (from config)
        gains_speech = [1.0 for _ in range(n_s)]
        if n_s > 1:
            sir_lo, sir_hi = self.dataset_cfg['sir_speech_db']
            rms_ref = self.compute_rms(conv_s[0])
            for i in range(1, n_s):
                sir_db = random.uniform(sir_lo, sir_hi)
                rms_i  = self.compute_rms(conv_s[i])
                gain_i = float(rms_ref / (rms_i * (10.0 ** (sir_db / 20.0) + 1e-12)))
                conv_s[i] *= gain_i
                gains_speech[i] = gain_i

        # Apply SIR for noise signals (from config)
        if conv_n.shape[0] > 1:
            sirn_lo, sirn_hi = self.dataset_cfg['sir_noise_db']
            rmsn_ref = self.compute_rms(conv_n[0]) if n_n_wo_diffuse_noise > 0 else None
            for i in range(1, n_n_wo_diffuse_noise):
                sir_db = random.uniform(sirn_lo, sirn_hi)
                rms_i  = self.compute_rms(conv_n[i])
                gain_i = float(rmsn_ref / (rms_i * (10.0 ** (sir_db / 20.0) + 1e-12)))
                conv_n[i] *= gain_i

        # SNR-based scaling of noise (from config)
        if conv_n.shape[0] > 0:
            snr_lo, snr_hi = self.dataset_cfg['snr_db']
            snr_db = random.uniform(snr_lo, snr_hi)
            speech_sum = conv_s.sum(dim=0)
            noise_sum  = conv_n.sum(dim=0)
            rms_s_tot  = self.compute_rms(speech_sum)
            rms_n_tot  = self.compute_rms(noise_sum)
            if rms_n_tot > 0:
                gain_n = float(rms_s_tot / (rms_n_tot * (10.0 ** (snr_db / 20.0))))
                conv_n = conv_n * gain_n
            else:
                conv_n = torch.zeros_like(conv_n)

        # Generation mix
        mix = conv_s.sum(dim=0) + (conv_n.sum(dim=0) if conv_n.shape[0] > 0 else 0.0)  # [C,T]
        mix = self.fit_length(mix, self.max_len)

        # Generaton target
        target_near = self.generation_target(selected_speech_wavs, [0], dir_s, gains=gains_speech)
        target_far = self.generation_target(selected_speech_wavs, [1], dir_s, gains=gains_speech)
        target_sum = self.generation_target(selected_speech_wavs, [0, 1], dir_s, gains=gains_speech)
        
        # RMS normalization
        if self.args.rms_norm:
            mix, rate = self.audio_norm(mix)
            target_near = target_near * rate
            target_far = target_far * rate
            target_sum = target_sum * rate

        return mix, target_near, target_far, target_sum
    
    # 最後の次元を L にパディング/トリム
    @staticmethod
    def fit_length(x: torch.Tensor, L: int) -> torch.Tensor:
        T = x.shape[-1]
        if T < L:
            pad = torch.zeros(*x.shape[:-1], L-T, device=x.device, dtype=x.dtype)
            return torch.cat([x, pad], dim=-1)
        else:
            return x[..., :L]
    
    # RMS計算
    @staticmethod
    def compute_rms(wav: torch.Tensor, dims=(-2, -1), eps: float = 1e-12) -> torch.Tensor:
        return torch.sqrt(torch.mean(wav**2, dim=dims) + eps)
    
    # RMS正規化
    @staticmethod
    def rms_normalization(x: torch.Tensor, eps: float = 1e-12) -> float:
        rms = (x ** 2).mean() ** 0.5
        scalar = 10 ** (-25 / 20) / (rms + eps)
        x = x * scalar
        pow_x = x**2
        avg_pow_x = pow_x.mean()
        rmsx = pow_x[pow_x>avg_pow_x].mean()**0.5
        scalarx = 10 ** (-25 / 20) / (rmsx + eps)
        x = x * scalarx
        return x, scalar * scalarx
    
    # 角度領域内か判定 (deg)
    @staticmethod
    def is_within_angular(pos, arr, theta_l, theta_h):
        vec = pos - arr
        az = (math.degrees(math.atan2(vec[1], vec[0])) + 360.0) % 360.0
        if theta_l <= theta_h:
            return theta_l <= az <= theta_h
        else:
            return az >= theta_l or az <= theta_h

    # 球領域内か判定
    @staticmethod
    def is_within_distance(pos, arr, d_query):
        dist = np.linalg.norm(pos - arr)
        return dist <= d_query
    
    # 角度範囲チェック用ヘルパー (rad)
    @staticmethod
    def angle_in_range(phi, lo, hi):
        return (lo <= hi and lo <= phi <= hi) or (lo > hi and (phi >= lo or phi <= hi))
    
    # 角度サンプリング用ヘルパー (rad)
    @staticmethod
    def sample_angle_in_range(lo, hi):
            if lo <= hi:
                return random.uniform(lo, hi)
            else:
                span1 = 2 * math.pi - lo
                if random.random() < span1 / (span1 + hi):
                    return random.uniform(lo, 2 * math.pi)
                else:
                    return random.uniform(0, hi)
    
    # スペクトログラムのプロット（行=音声の種類, 列=マイクch）
    def make_spectrogram(self, out_root, out_name, 
                        wavs,                                     # Tensor/ndarray or [(label, wav), ...] or [wav, ...]
                        mic_sens_v_per_pa: float = 17.8e-3,       # [V/Pa]
                        adc_full_scale_volts: float = 1.0,        # [V]
                        preamp_gain_db: float = 0.0):             # [dB]
        matplotlib.use('Agg') 
        if isinstance(wavs, torch.Tensor):
            wav_pairs = [(str(out_name), wavs)]
        elif isinstance(wavs, np.ndarray):
            wav_pairs = [(str(out_name), torch.as_tensor(wavs))]
        elif isinstance(wavs, (list, tuple)):
            wav_pairs = []
            for i, it in enumerate(wavs):
                if isinstance(it, tuple) and len(it) == 2:
                    lbl, w = it
                    if isinstance(w, np.ndarray):
                        w = torch.as_tensor(w)
                    elif not isinstance(w, torch.Tensor):
                        raise ValueError(f"wavs[{i}] は Tensor/ndarray である必要がある")
                    wav_pairs.append((str(lbl), w))
                elif isinstance(it, (torch.Tensor, np.ndarray)):
                    w = torch.as_tensor(it) if isinstance(it, np.ndarray) else it
                    wav_pairs.append((f"item{i}", w))
                else:
                    raise ValueError(f"wavs[{i}] の型が不正: {type(it)}")
        else:
            raise ValueError(f"`wavs` の型が不正: {type(wavs)}")

        # ---- デジタル -> Pa 換算 ----
        gain_linear = 10.0 ** (preamp_gain_db / 20.0)
        pa_per_digital = adc_full_scale_volts / (mic_sens_v_per_pa * gain_linear)

        # ---- グリッドサイズ決定 ----
        n_rows = len(wav_pairs)
        row_nch = []
        for _, wav in wav_pairs:
            w = wav.detach().cpu().numpy()
            if w.ndim == 1:
                C = 1
            elif w.ndim == 2:
                C = w.shape[0]                    # [C,T] 前提
            else:
                raise ValueError(f"wav は 1D/2D 必須, got shape={w.shape}")
            row_nch.append(int(C))
        n_cols = max(row_nch) if n_rows > 0 else 1

        # ---- 図作成 ----
        fig_w = max(3.0 * n_cols, 6.0) + 2.0   # 余白加味
        fig_h = max(2.6 * n_rows, 3.5)
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(fig_w, fig_h),
            squeeze=False, sharex=True, sharey=True
        )       
        # ---- 表示設定 ----
        vmin, vmax = -20.0, 160.0
        cmap = "plasma"
        p_ref = 20e-6
        eps = 1e-20
        last_im = None

        # ---- 各セル描画 ----
        for r, (label, wav_t) in enumerate(wav_pairs):
            w = wav_t.detach().cpu().numpy()
            if w.ndim == 1:
                w2d = w[None, :]
            elif w.ndim == 2:
                w2d = w
            else:
                raise ValueError(f"wav は 1D/2D 必須, got shape={w.shape}")
            C, T = w2d.shape

            # 行ラベル（図全体座標で縦書き、行中央に配置）
            y_pos = (n_rows - r - 0.5) / n_rows
            fig.text(0.003, y_pos, label,
                    va='center', ha='center',
                    rotation=90, fontsize=12, weight="bold")

            for c in range(n_cols):
                ax = axes[r, c]
                if c >= C:
                    ax.axis("off")
                    continue

                # デジタル -> Pa
                wav_pa = (w2d[c] * pa_per_digital).astype(np.float64)

                # STFT & |.|（SPL基準は20µPa）
                spec = librosa.stft(wav_pa, n_fft=self.n_fft, hop_length=self.hop)
                amp_pa = np.abs(spec)
                spec_db_spl = 20.0 * np.log10(np.maximum(amp_pa, eps) / p_ref)

                im = librosa.display.specshow(
                    spec_db_spl, sr=self.sr, hop_length=self.hop,
                    x_axis='time', y_axis='hz', ax=ax,
                    vmin=vmin, vmax=vmax, cmap=cmap
                )
                last_im = im
                if r == 0:
                    ax.set_title(f"Mic {c}")

        # ---- 共通カラーバー ----
        if last_im is not None:
            cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])  
            cbar = fig.colorbar(last_im, cax=cax)
            cbar.set_label("dB SPL")

        # ---- 保存 ----
        fig.suptitle(f"{out_name} Spectrograms (dB SPL)", y=0.995)
        plt.tight_layout(rect=[0, 0, 0.90, 0.98])
        out_path = str(out_root / f"{out_name}.png")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
    
    # メタデータと音声(生音&スペクトログラム)の保存
    def save_metadata_and_audio(self, out_root, theta_l_query, theta_h_query, d_query, 
                            speech_pos, noise_pos, arr, region_indices, q_a, q_d,
                            selected_speech_wavs, selected_noise_wavs, mix, target, use_diffuse, diffuse_noise, n_n_wo_diffuse_noise):
        # --- 各話者の平面極座標(r, θ)を計算 ---
        arr = np.asarray(arr).reshape(1, 3)
        rel_s   = np.asarray(speech_pos) - arr
        speech_pos_d  = np.linalg.norm(rel_s, axis=1)
        speech_pos_az = (np.degrees(np.arctan2(rel_s[:, 1], rel_s[:, 0])) + 360.0) % 360.0
        speech_pos_az = np.where(speech_pos_d == 0.0, 0.0, speech_pos_az)
        
        if len(noise_pos) == 0:
            noise_pos_d = np.array([0], dtype=np.float32)
            noise_pos_az = np.array([0], dtype=np.float32)
        else:
            rel_n = np.asarray(noise_pos) - arr
            noise_pos_d  = np.linalg.norm(rel_n, axis=1)
            noise_pos_az = (np.degrees(np.arctan2(rel_n[:, 1], rel_n[:, 0])) + 360.0) % 360.0
            noise_pos_az = np.where(noise_pos_d == 0.0, 0.0, noise_pos_az)

        metadata = {
            'theta_l': float(theta_l_query), 
            'theta_h': float(theta_h_query), 
            'd_query': float(d_query),
            'use_diffuse': bool(use_diffuse),
            'Q': int(self.Q_val),
            'Q_a': q_a,
            'Q_d': q_d,
            'speech_pos': np.asarray(speech_pos).tolist(),
            'speech_pos_d': speech_pos_d.tolist(),
            'speech_pos_az': speech_pos_az.tolist(),
            'noise_pos': noise_pos,
            'noise_pos_d': noise_pos_d.tolist(),
            'noise_pos_az': noise_pos_az.tolist(),
            'array_pos': np.asarray(arr).tolist(),
            'region_indices': region_indices,
            'room': np.asarray(self.room).tolist(),
            'rt60': self.rt60
        }
        
        metadata = _to_serializable(metadata)
        with open(out_root/"metadata.json","w") as jf:
            json.dump(metadata,jf,indent=2)
            
        # save speech/noise/mix/target wavs and spectrograms
        for i, wav in enumerate(selected_speech_wavs): 
            sf.write(str(out_root/f"speech_{i:02d}.wav"), wav, self.sr)
        
        if n_n_wo_diffuse_noise > 0:
            for i, wav in enumerate(selected_noise_wavs):  
                sf.write(str(out_root/f"noise_{i:02d}.wav"), wav, self.sr)
        
        if use_diffuse and diffuse_noise is not None:
            sf.write(str(out_root/f"diffuse_noise.wav"), diffuse_noise.cpu().numpy().T, self.sr)
        
        generation_spectrogram_list = []
        sf.write(str(out_root/"mix.wav"), mix.cpu().numpy().T, self.sr)
        generation_spectrogram_list.append(("mix", mix))
        sf.write(str(out_root/"target.wav"), target.cpu().numpy().T, self.sr)
        generation_spectrogram_list.append(("target", target))
        self.make_spectrogram(out_root, "spectrogram", generation_spectrogram_list)
    
    # 3D plot
    def simulation_plot_3d(self, out_root, speech_pos, noise_pos, arr, theta_l, theta_h, d_query, region_indices):
        """
        3D plot of the simulation setup.
        arr: microphone array position (x, y, z)
        room: room dimensions (x_max, y_max, z_max)
        speech_pos: positions of speech sources
        noise_pos: positions of noise sources
        theta_l: lower angle limit for query region
        theta_h: upper angle limit for query region
        d_query: distance threshold for query region
        region_indices_A: indices of speech sources inside the query region
        """
        # --- 3Dプロットの初期化 ---
        matplotlib.use('Agg') 
        fig = plt.figure(figsize=(8, 6), dpi=400)
        ax = fig.add_subplot(111, projection='3d')

        # --- 部屋の枠線 ---
        x_max , y_max, z_max = self.room
        for xs in [0, x_max]:
            for ys in [0, y_max]:
                ax.plot([xs, xs], [ys, ys], [0, z_max], 'k--', linewidth=0.5)
        for xs in [0, x_max]:
            for zs in [0, z_max]:
                ax.plot([xs, xs], [0, y_max], [zs, zs], 'k--', linewidth=0.5)
        for ys in [0, y_max]:
            for zs in [0, z_max]:
                ax.plot([0, x_max], [ys, ys], [zs, zs], 'k--', linewidth=0.5)

        # --- 軸範囲・目盛り ---
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)
        ax.set_zlim(0, z_max)
        ax.set_xticks([0, x_max])
        ax.set_yticks([0, y_max])
        ax.set_zticks([0, z_max])
        # パッドを入れて重なりを回避
        ax.tick_params(axis='x', pad=10)
        ax.tick_params(axis='y', pad=10)
        ax.set_box_aspect((x_max, y_max, z_max))

        # --- マイク中心 ---
        mic_np = np.array(arr)
        ax.scatter(*mic_np, c='blue', marker='^', label='mic')
        ax.plot([mic_np[0], mic_np[0]], [mic_np[1], mic_np[1]], [0, mic_np[2]], 'b--', linewidth=0.5)

        # --- speech: inside/outside query ---
        speech_np = np.array(speech_pos)
        # inside
        inside_idx = [i for i in range(len(speech_np)) if i in region_indices]
        outside_idx = [i for i in range(len(speech_np)) if i not in region_indices]
        if inside_idx:
            sp_in = speech_np[inside_idx]
            ax.scatter(sp_in[:,0], sp_in[:,1], sp_in[:,2], c='red', marker='o', label='speech (inside query)')
            for x,y,z in sp_in:
                ax.plot([x, x], [y, y], [0, z], linestyle='--', color='red', linewidth=0.5)
        if outside_idx:
            sp_out = speech_np[outside_idx]
            ax.scatter(sp_out[:,0], sp_out[:,1], sp_out[:,2], c='orange', marker='o', label='speech (outside query)')
            for x,y,z in sp_out:
                ax.plot([x, x], [y, y], [0, z], linestyle='--', color='orange', linewidth=0.5)

        # --- noise ---
        if len(noise_pos) > 0:
            noise_np = np.array(noise_pos)
            ax.scatter(noise_np[:,0], noise_np[:,1], noise_np[:,2], c='green', marker='s', label='noise')
            for x,y,z in noise_np:
                ax.plot([x, x], [y, y], [0, z], 'g--', linewidth=0.5)
        
        # --- クエリ領域の描画（球面ウェッジ＋切断面塗りつぶし＋投影円） ---
        # θ_l>θ_h のときは２区間に分割
        if not self.region_type == 'spherical':
            if theta_l <= theta_h:
                intervals = [(theta_l, theta_h)]
            else:
                intervals = [(theta_l, 360.0), (0.0, theta_h)]
        else:
            intervals = [(0.0, 360.0)]

        # (1) 球面ウェッジ本体
        if not self.region_type == 'angular' and not self.args.query2D:
            φ = np.linspace(0, np.pi, 50)  # 極角
            for start, end in intervals:
                θ = np.deg2rad(np.linspace(start, end, 50))  # 方位角
                Φ, Θ = np.meshgrid(φ, θ, indexing='ij')
                Xw = arr[0] + d_query * np.sin(Φ) * np.cos(Θ)
                Yw = arr[1] + d_query * np.sin(Φ) * np.sin(Θ)
                Zw = arr[2] + d_query * np.cos(Φ)
                # 部屋外は NaN で切り落とし
                m = (Xw>=0)&(Xw<=x_max)&(Yw>=0)&(Yw<=y_max)&(Zw>=0)&(Zw<=z_max)
                Xw[~m]=np.nan; Yw[~m]=np.nan; Zw[~m]=np.nan
                ax.plot_surface(
                    Xw, Yw, Zw,
                    color='gray', alpha=0.5,
                    edgecolor='none',
                    rstride=1, cstride=1, antialiased=True
                )

        # (2) 切断面（θ = θ_l, θ_h の放射面）
        if self.region_type == 'conical' and not self.args.query2D:
            for angle in (theta_l, theta_h):
                tr = math.radians(angle)
                # 放射面（薄い楔形）
                r = np.linspace(0, d_query, 50)
                φ = np.linspace(0, np.pi, 50)
                R, P = np.meshgrid(r, φ, indexing='ij')
                Xp = arr[0] + R * np.sin(P) * math.cos(tr)
                Yp = arr[1] + R * np.sin(P) * math.sin(tr)
                Zp = arr[2] + R * np.cos(P)
                # 部屋外切り落とし
                mask = (Xp>=0)&(Xp<=x_max)&(Yp>=0)&(Yp<=y_max)&(Zp>=0)&(Zp<=z_max)
                Xp[~mask] = np.nan; Yp[~mask] = np.nan; Zp[~mask] = np.nan
                # 塗り潰し
                ax.plot_surface(Xp, Yp, Zp, color='gray', alpha=0.25, edgecolor='none', linewidth=0, rstride=1, cstride=1, antialiased=True)

        # 壁面による切断面（x=0, x=x_max, y=0, y=y_max）
        if not self.region_type == 'angular' and not self.args.query2D:
            for axis, val in [('x',0.0),('x',x_max),('y',0.0),('y',y_max)]:
                if axis=='x':
                    Y = np.linspace(0, y_max, 50)
                    Z = np.linspace(0, z_max, 50)
                    Yg, Zg = np.meshgrid(Y, Z, indexing='ij')
                    Xg = np.full_like(Yg, val)
                else:
                    X = np.linspace(0, x_max, 50)
                    Z = np.linspace(0, z_max, 50)
                    Xg, Zg = np.meshgrid(X, Z, indexing='ij')
                    Yg = np.full_like(Xg, val)
                # 距離＆角度マスク
                vx, vy, vz = Xg-arr[0], Yg-arr[1], Zg-arr[2]
                dist = np.sqrt(vx**2+vy**2+vz**2)
                az   = (np.degrees(np.arctan2(vy,vx))+360.0)%360.0
                mask = (dist<=d_query)
                angle_mask = np.zeros_like(mask)
                for st, ed in intervals:
                    if st<=ed:
                        angle_mask |= (az>=st)&(az<=ed)
                    else:
                        angle_mask |= (az>=st)|(az<=ed)
                mask &= angle_mask
                # 部屋外も切り落とし
                mask &= (Xg>=0)&(Xg<=x_max)&(Yg>=0)&(Yg<=y_max)&(Zg>=0)&(Zg<=z_max)
                Xg[~mask]=np.nan; Yg[~mask]=np.nan; Zg[~mask]=np.nan
                # 塗り潰し
                ax.plot_surface(Xg, Yg, Zg, color='gray', alpha=0.25, edgecolor='none', rstride=1, cstride=1)

        # --- (3) 床面(z=0) と 配列高さ(z=arr[2]) の円周投影 ---
        if not self.region_type == 'angular' and not self.args.query2D:
            x_max, y_max, z_max = self.room
            has_floor_circle = arr[2] <= d_query

            # (3-1) 床面上の共有円 (球と平面 z=0 の交線)
            if has_floor_circle:
                r_floor = math.sqrt(d_query**2 - arr[2]**2)
                theta_circle = np.linspace(0, 2*np.pi, 200)
                xf = arr[0] + r_floor * np.cos(theta_circle)
                yf = arr[1] + r_floor * np.sin(theta_circle)
                zf = np.zeros_like(theta_circle)

                mask_f = (xf >= 0) & (xf <= x_max) & (yf >= 0) & (yf <= y_max)
                edges_f = np.where(np.diff(mask_f.astype(int)) != 0)[0] + 1
                segments_f = np.split(np.arange(len(theta_circle)), edges_f)
                for seg in segments_f:
                    xs, ys, zs = xf[seg], yf[seg], zf[seg]
                    if mask_f[seg][0]:
                        # 部屋内アークを黒線で描画
                        ax.plot(xs, ys, zs, 'k-', linewidth=1.5)
                    else:
                        # 部屋外はクリップして壁に沿う線を描画
                        xs_c = np.clip(xs, 0, x_max)
                        ys_c = np.clip(ys, 0, y_max)
                        ax.plot(xs_c, ys_c, zs, 'k-', linewidth=1.5)

            # (3-2) 配列高さ(z=arr[2]) 上の円周 (半径 d_query)
            theta_circle = np.linspace(0, 2*np.pi, 200)
            xt = arr[0] + d_query * np.cos(theta_circle)
            yt = arr[1] + d_query * np.sin(theta_circle)
            zt = np.full_like(theta_circle, arr[2])

            mask_t = (xt >= 0) & (xt <= x_max) & (yt >= 0) & (yt <= y_max)
            edges_t = np.where(np.diff(mask_t.astype(int)) != 0)[0] + 1
            segments_t = np.split(np.arange(len(theta_circle)), edges_t)
            for seg in segments_t:
                xs, ys, zs = xt[seg], yt[seg], zt[seg]
                if mask_t[seg][0]:
                    # 部屋内アークを黒線で描画
                    ax.plot(xs, ys, zs, 'k-', linewidth=1.5)
                else:
                    # 部屋外はクリップして壁に沿う線を描画
                    xs_c = np.clip(xs, 0, x_max)
                    ys_c = np.clip(ys, 0, y_max)
                    ax.plot(xs_c, ys_c, zs, 'k-', linewidth=1.5)
        elif not self.region_type == 'angular' and self.args.query2D:
            # 床面(z=0) 上の円周 (半径 d_query)
            theta_circle = np.linspace(0, 2*np.pi, 200)
            xf = arr[0] + d_query * np.cos(theta_circle)
            yf = arr[1] + d_query * np.sin(theta_circle)
            zf = np.zeros_like(theta_circle)
            mask_f = (xf >= 0) & (xf <= x_max) & (yf >= 0) & (yf <= y_max)
            edges_f = np.where(np.diff(mask_f.astype(int)) != 0)[0] + 1
            segments_f = np.split(np.arange(len(theta_circle)), edges_f)
            for seg in segments_f:
                xs, ys, zs = xf[seg], yf[seg], zf[seg]
                if mask_f[seg][0]:
                    ax.plot(xs, ys, zs, 'k-', linewidth=1.5)
                else:
                    xs_c = np.clip(xs, 0, x_max)
                    ys_c = np.clip(ys, 0, y_max)
                    ax.plot(xs_c, ys_c, zs, 'k-', linewidth=1.5)

        # --- (4) θ_l, θ_h の放射線 ---
        if not self.region_type == 'spherical':
            for angle in (theta_l, theta_h):
                tr = math.radians(angle)
                ca, sa = math.cos(tr), math.sin(tr)

                # --- 床面(z=0) の放射線 ---
                ts = []
                if ca > 0: ts.append((x_max - arr[0]) / ca)
                elif ca < 0: ts.append(-arr[0] / ca)
                if sa > 0: ts.append((y_max - arr[1]) / sa)
                elif sa < 0: ts.append(-arr[1] / sa)
                t_wall = min(t for t in ts if t > 0)
                x_wall = arr[0] + t_wall * ca
                y_wall = arr[1] + t_wall * sa

                if self.region_type == 'conical':
                    if has_floor_circle:
                        x_int = arr[0] + r_floor * ca
                        y_int = arr[1] + r_floor * sa
                        # 内側を黒破線、外側を赤破線で描画
                        ax.plot([arr[0], x_int], [arr[1], y_int], [0, 0], 'k--', linewidth=1)
                        ax.plot([x_int, x_wall], [y_int, y_wall], [0, 0], 'r--', linewidth=1)
                    else:
                        # 共有円がない場合は赤破線で描画
                        ax.plot([arr[0], x_wall], [arr[1], y_wall], [0, 0], 'r--', linewidth=1)
                    # --- 配列高さ(z=arr[2]) の放射線 (全て黒破線) ---
                    x_top = arr[0] + d_query * ca
                    y_top = arr[1] + d_query * sa
                    x_end = np.clip(x_top, 0, x_max)
                    y_end = np.clip(y_top, 0, y_max)
                    ax.plot([arr[0], x_end], [arr[1], y_end], [arr[2], arr[2]], 'k--', linewidth=1)
                elif self.region_type == 'angular':
                    # 角度領域のみの場合は赤破線で描画
                    ax.plot([arr[0], x_wall], [arr[1], y_wall], [0, 0], 'r--', linewidth=1)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.legend(loc="upper left")
        plt.title("Room Simulation")
        plt.tight_layout()
        
        out_root = Path(out_root)
        plt.savefig(str(out_root/"simulation_3d.png"), dpi=400, bbox_inches='tight')
        plt.close()