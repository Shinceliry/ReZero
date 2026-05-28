import os
import json
import torch
from torch.utils.data import Dataset
import soundfile as sf
import numpy as np
import torch.nn.functional as F
import yaml
import math

class RezeroFreezeDataset(Dataset):
    """
    Directory structure:
        input_dir/
            <roomID1>/
                metadata.json
                mix.wav
                target.wav
            <roomID2>/
                ...

    Returns:
        mix         (Tensor[C, T])      # mixture waveform, C channels
        theta_l     (Tensor[])          # lower angular bound (deg)
        theta_h     (Tensor[])          # upper angular bound (deg)
        d_query     (Tensor[])          # distance threshold
        Q           (Tensor[])          # sources within query region
        target      (Tensor[C, T])      # target query waveform, C channels
    """

    def __init__(self, args, input_dir: str, cpuram: bool = False):
        super().__init__()
        self.args = args
        self.input_dir = input_dir
        self.cpuram = cpuram
        sample_dirs = []
        with os.scandir(self.input_dir) as it:
            for entry in it:
                if entry.is_dir():
                    sample_dirs.append(entry.path)
        self.sample_dirs = sorted(sample_dirs)
        self.region_type = args.region_type
        
        # Load config
        cfg_file = args.config
        with open(cfg_file, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        self.dataset_cfg = cfg['dataset_generation']
        audio_cfg = cfg['audio']
        self.iterations = int(cfg['training']['iterations'])

        # Audio settings
        self.sr = audio_cfg['sample_rate']
        self.n_fft = int(self.sr * audio_cfg['stft']['window_size_ms'] / 1000)
        self.hop = int(self.sr * audio_cfg['stft']['hop_size_ms'] / 1000)
        self.win_type = audio_cfg['stft']['window_type'].lower()
        
        # CPUメモリにデータを格納
        if self.cpuram:
            print("Loading audio files into RAM...")
            samples = []
            sample_counts = 0
            for dir in self.sample_dirs:
                meta_path = os.path.join(dir, "metadata.json")
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                mix_path = os.path.join(dir, "mix.wav")
                if not os.path.isfile(mix_path):
                    print(f"[warn] skip: mix.wav not found in {dir}")
                    continue
                mix_np, _ = sf.read(mix_path)
                mix = self._to_tensor(mix_np)  # (C, T)
                
                target_path = os.path.join(dir, "target.wav")
                if os.path.isfile(target_path):
                    target_np, _ = sf.read(target_path)
                    target = self._to_tensor(target_np)  # (C, T)

                # --- push to RAM cache ---
                samples.append((mix, meta, target))
                sample_counts += 1
            self.samples = samples
            print(f"Loaded {sample_counts} files into RAM.")

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx: int):
        if not self.cpuram:
            sample_dir = self.sample_dirs[idx]
            meta_path = os.path.join(sample_dir, "metadata.json")
            with open(meta_path, "r") as f:
                meta = json.load(f)
                
            mix_path = os.path.join(sample_dir, "mix.wav")
            mix_np, _ = sf.read(mix_path)
            mix = self._to_tensor(mix_np)
            
            target_path = os.path.join(sample_dir, "target.wav")
            target_np, _ = sf.read(target_path)
            target = self._to_tensor(target_np)
        else:
            mix, meta, target = self.samples[idx]

        room = meta["room"]
        min_wall = self.dataset_cfg["min_source_wall_dist_m"]
        if self.region_type == 'angular':
            theta_l_query = torch.tensor(meta["theta_l"], dtype=torch.float32)
            theta_h_query = torch.tensor(meta["theta_h"], dtype=torch.float32)
            d_query = math.sqrt((room[0] - min_wall)**2 + (room[1] - min_wall)**2 + (room[2] - min_wall)**2)
            d_query = torch.tensor(d_query, dtype=torch.float32)
        elif self.region_type == 'spherical':
            theta_l_query = torch.tensor(0.0, dtype=torch.float32)
            theta_h_query = torch.tensor(360.0, dtype=torch.float32)
            d_query = torch.tensor(meta["d_query"], dtype=torch.float32)
        elif self.region_type == 'conical':
            theta_l_query = torch.tensor(meta["theta_l"], dtype=torch.float32)
            theta_h_query = torch.tensor(meta["theta_h"], dtype=torch.float32)
            d_query = torch.tensor(meta["d_query"], dtype=torch.float32)
        Q = torch.tensor(meta["Q"], dtype=torch.long)
        
        return mix, theta_l_query, theta_h_query, d_query, Q, target

    @staticmethod
    def _to_tensor(wav_np: np.ndarray) -> torch.Tensor:
        """
        numpy array -> torch tensor
        """
        if wav_np.ndim == 1:
            wav_np = wav_np[:, None]
        wav_np = wav_np.T
        return torch.from_numpy(wav_np).float()

def collate_fn(batch):
    """
    カスタム collate_fn:
    - waveforms (mix, target) をバッチ中の最長長さに合わせてゼロパディング
    - 固定長テンソル (mic_pos, array_pos, theta, Q) をスタック
    - 可変長テンソル (speech_pos, noise_pos) はリストのまま返却

    Returns dict with keys:
        'mix', 'theta_l', 'theta_h', 'd_query', 'Q', 'target'
    """
    mixes, theta_l_list, theta_h_list, d_query_list, Q_list, targets = zip(*batch)
    lengths = [m.shape[1] for m in mixes]
    max_len = max(lengths)
    mix_batch = torch.stack([F.pad(m, (0, max_len - m.shape[1])) for m in mixes], dim=0)
    target_batch = torch.stack([F.pad(t, (0, max_len - t.shape[1])) for t in targets], dim=0)
    theta_l = torch.stack(theta_l_list)
    theta_h = torch.stack(theta_h_list)
    d_query = torch.stack(d_query_list)
    Q_batch = torch.stack(Q_list)
    return {
        'mix': mix_batch,
        'theta_l': theta_l,
        'theta_h': theta_h,
        'd_query': d_query,
        'Q': Q_batch,
        'target': target_batch
    }