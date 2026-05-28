#!/usr/bin/env python3
import os
import json
import glob
import yaml
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

from src.ADCRezero.models.ARezero import ARezeroModel
from src.ADCRezero.models.CRezero import CRezeroModel
from src.ADCRezero.models.DRezero import DRezeroModel
# from src.ADCRezero.models.DRezero_w_distancefeature import DRezeroModel

class TestDataset(Dataset):
    """
    Dataset for batched inference.
    Each sample returns:
        mix: Tensor[C, T]
        theta_l: Tensor[1] or None
        theta_h: Tensor[1] or None
        d_query: Tensor[1] or None
        basename: str  # used for output
    """
    def __init__(self, input_dir: str, sr: int, region_type: str):
        self.sr = sr
        self.region_type = region_type
        # collect mix and metadata paths
        self.items = []
        mix_paths = glob.glob(os.path.join(input_dir, '**', 'mix.wav'), recursive=True)
        for mix_path in sorted(mix_paths):
            meta_path = os.path.join(os.path.dirname(mix_path), 'metadata.json')
            if not os.path.isfile(meta_path):
                print(f"[Warning] no metadata: {meta_path}")
                continue
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            basename = os.path.basename(os.path.dirname(mix_path))
            self.items.append({'mix': mix_path, 'meta': meta, 'name': basename})
        # mix_paths = glob.glob(os.path.join(input_dir, '**', '**.wav'), recursive=True)
        # for mix_path in sorted(mix_paths):
        #     meta_path = os.path.join(os.path.dirname(mix_path), 'metadata.json')
        #     if not os.path.isfile(meta_path):
        #         print(f"[Warning] no metadata: {meta_path}")
        #         continue
        #     with open(meta_path, 'r') as f:
        #         meta = json.load(f)
        #     basename = os.path.basename(mix_path).replace('.wav','')
        #     self.items.append({'mix': mix_path, 'meta': meta, 'name': basename})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        wav = load_wav(item['mix'], self.sr)
        mix = torch.from_numpy(wav).float()  # [C, T]
        meta = item['meta']
        theta_l = theta_h = d_q = None
        if self.region_type in ['angular', 'conical']:
            theta_l = torch.tensor([meta['theta_l']], dtype=torch.float32)
            theta_h = torch.tensor([meta['theta_h']], dtype=torch.float32)
        if self.region_type in ['spherical', 'conical']:
            d_q = torch.tensor([meta['d_query']], dtype=torch.float32)
        return {
            'mix': mix,
            'theta_l': theta_l,
            'theta_h': theta_h,
            'd_query': d_q,
            'name': item['name']
        }

def collate_fn(batch):
    # determine max time length
    max_T = max(item['mix'].shape[1] for item in batch)
    C = batch[0]['mix'].shape[0]
    mixes = []
    theta_ls, theta_hs, d_qs = [], [], []
    names = []
    for item in batch:
        mix = item['mix']
        T = mix.shape[1]
        if T < max_T:
            pad = torch.zeros(C, max_T - T)
            mix = torch.cat([mix, pad], dim=1)
        mixes.append(mix.unsqueeze(0))  # [1, C, max_T]
        names.append(item['name'])
        # push None as zeros if not used
        theta_ls.append(item['theta_l'] if item['theta_l'] is not None else torch.tensor([0.0]))
        theta_hs.append(item['theta_h'] if item['theta_h'] is not None else torch.tensor([0.0]))
        d_qs.append(item['d_query'] if item['d_query'] is not None else torch.tensor([0.0]))
    batch_mix = torch.cat(mixes, dim=0)  # [B, C, max_T]
    batch_theta_l = torch.cat(theta_ls, dim=0)
    batch_theta_h = torch.cat(theta_hs, dim=0)
    batch_d = torch.cat(d_qs, dim=0)
    return batch_mix, batch_theta_l, batch_theta_h, batch_d, names

def load_wav(path: str, target_sr: int) -> np.ndarray:
    wav, sr = sf.read(path)
    if wav.ndim == 1:
        wav = wav[:, None]
    wav = wav.T
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return wav.astype(np.float32)

def inferenceADC(args):
    os.makedirs(args.est_dir, exist_ok=True)
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    audio_cfg = cfg['audio']
    sr = audio_cfg['sample_rate']
    n_fft = int(sr * audio_cfg['stft']['window_size_ms'] / 1000)
    hop = int(sr * audio_cfg['stft']['hop_size_ms'] / 1000)
    win_type = audio_cfg['stft']['window_type'].lower()
    window = torch.hann_window(n_fft) if win_type=='hann' else torch.ones(n_fft)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if args.region_type=='angular':
        model = ARezeroModel(args, device=device, return_mask=False, complex_as_channel=True)
        state = torch.load(args.ARezero_path, map_location=device)
    elif args.region_type=='spherical':
        model = DRezeroModel(args, device=device, return_mask=False, complex_as_channel=True)
        state = torch.load(args.DRezero_path, map_location=device)
    elif args.region_type=='conical':
        model = CRezeroModel(args, device=device, return_mask=False, complex_as_channel=True)
        state = torch.load(args.CRezero_path, map_location=device)
    else:
        raise ValueError("Unkown region type. args.region_type must be angular, spherical or conical.")
    state = {k: v.float() for k,v in state.items()}
    model.load_state_dict(state, strict=True)
    model = model.float().to(device)
    model.eval()

    # prepare data loader
    dataset = TestDataset(args.test_dir, sr, args.region_type)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        collate_fn=collate_fn)

    with torch.no_grad():
        for batch_mix, batch_theta_l, batch_theta_h, batch_d, names in tqdm(loader):
            batch_mix = batch_mix.to(device)
            batch_theta_l = batch_theta_l.to(device)
            batch_theta_h = batch_theta_h.to(device)
            batch_d = batch_d.to(device)

            if args.region_type=='angular':
                x_hat_spec = model(batch_mix, batch_theta_l, batch_theta_h)
            elif args.region_type=='spherical':
                x_hat_spec = model(batch_mix, batch_d)
            elif args.region_type == "conical":
                x_hat_spec = model(batch_mix, batch_theta_l, batch_theta_h, batch_d)

            # ISTFT and save per sample
            B = x_hat_spec.size(0)
            for i in range(B):
                spec = x_hat_spec[i]
                est = torch.istft(
                                spec,
                                n_fft=n_fft,
                                hop_length=hop,
                                window=window.to(device),
                                length=batch_mix.size(-1)
                                )
                out_path = os.path.join(args.est_dir, f'{names[i]}_est.wav')
                sf.write(out_path, est.cpu().numpy().T, sr)
                print(f'[Saved] {out_path}')