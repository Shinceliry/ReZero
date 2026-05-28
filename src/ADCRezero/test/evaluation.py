import os
import glob
import json
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
from mir_eval.separation import bss_eval_sources
from pystoi import stoi
from pesq import pesq
from tqdm import tqdm
import yaml

# --- Evaluation Metrics ---
def compute_sdr(ref: np.ndarray, est: np.ndarray, eps: float = 1e-12) -> float:
    """
    Scale-Invariant SDR (SI-SDR)
    ref, est: shape [T] もしくは [C, T]（マルチチャネルの場合は各チャネルで平均）
    戻り値: SI-SDR [dB]
    """
    ref = np.asarray(ref, dtype=np.float64)
    est = np.asarray(est, dtype=np.float64)

    if ref.shape != est.shape:
        raise ValueError(f"Shape mismatch: ref {ref.shape} vs est {est.shape}")

    if ref.ndim == 2:
        sisdrs = [compute_sdr(ref[c], est[c], eps) for c in range(ref.shape[0])]
        return float(np.mean(sisdrs))

    if ref.ndim != 1:
        raise ValueError(f"Expected 1D (T) or 2D (C,T), got {ref.ndim}D")

    ref_energy = np.sum(ref ** 2) + eps
    alpha = np.sum(est * ref) / ref_energy
    e_target = alpha * ref
    e_noise = est - e_target

    si_sdr = 10.0 * np.log10((np.sum(e_target ** 2) + eps) / (np.sum(e_noise ** 2) + eps))
    return float(si_sdr)


def compute_stoi(ref: np.ndarray, est: np.ndarray, sr: int) -> float:
    return float(stoi(ref, est, sr, extended=False))


def compute_pesq(ref: np.ndarray, est: np.ndarray, sr: int) -> float:
    try:
        pesq_score = float(pesq(sr, ref, est, 'wb'))
    except Exception as e:
        print(f"Warning: PESQ computation failed with error {e}. Returning 0.0")
        pesq_score = 0.0
    return pesq_score


def compute_energy_decay(est: np.ndarray, mix: np.ndarray) -> float:
    num = np.sum(est ** 2) + 1e-12
    den = np.sum(mix ** 2) + 1e-12
    return 10.0 * np.log10(den / num)

def compute_freq_mae_loss(n_fft, hop, win_type, x_hat_spec: torch.Tensor, weight: float, eps: float = 1e-8) -> torch.Tensor:
    """
    Frequency-domain MAE loss on complex spectrogram.
    L = weight * (|Re(Z)| + |Im(Z)|)_mean
    x_hat_spec: [F, T]
    """
    wav = x_hat_spec.to(torch.float32)
    window = torch.hann_window(n_fft, device=wav.device) if win_type == "hann" else torch.ones(n_fft, device=wav.device)
    x_hat_spec = torch.stft(wav, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
    mae = weight * (x_hat_spec.real.abs().sum() + x_hat_spec.imag.abs().sum())
    return mae

def compute_snr_loss(y: torch.Tensor, y_hat: torch.Tensor, 
                    snr_max_db: float = 30.0, eps: float = 1e-8, weight: float = 1.0) -> torch.Tensor:
    """
    クリップ付きSNR損失
    L = 10*log10((||e||^2 + tau*||y||^2) / (||y||^2)),  tau = 10^(-SNR_max/10)
    y: [T]
    y_hat: [T]
    """
    if y.shape != y_hat.shape:
        raise ValueError(f"shape mismatch: {y.shape} vs {y_hat.shape}")
    
    err = y - y_hat
    power_e = (err ** 2).sum()
    power_y = (y ** 2).sum()
    tau = 10.0 ** (-snr_max_db / 10.0)
    loss = 10.0 * torch.log10(power_e + tau * power_y)
    loss = weight * loss
    return loss

# --- Dataset and Sample ---
class EvaluationSample:
    def __init__(self, base, mix_path, target_path, est_path, info):
        self.id = base
        self.mix_path = mix_path
        self.target_path = target_path
        self.est_path = est_path
        self.info = info

class EvaluationDataset(Dataset):
    def __init__(self, test_dir, est_dir, region_type):
        self.samples = []
        self.region_type = region_type
        meta_paths = sorted(glob.glob(os.path.join(test_dir, '**', 'metadata.json'), recursive=True))
        for meta_path in meta_paths:
            dir_path = os.path.dirname(meta_path)
            base = os.path.basename(dir_path)
            mix_path = os.path.join(dir_path, 'mix.wav')
            est_path = os.path.join(est_dir, f"{base}_est.wav")
            target_path = os.path.join(dir_path, 'target.wav')
            if not (os.path.exists(mix_path) and os.path.exists(target_path) and os.path.exists(est_path)):
                print(f"Warning: Missing files for {base}, skipping.")
                continue
            with open(meta_path, 'r') as f:
                info = json.load(f)
            # assume metadata contains 'Qa' and 'Qd'
            self.samples.append(EvaluationSample(base, mix_path, target_path, est_path, info))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        mix, sr1 = sf.read(s.mix_path)
        target, sr2 = sf.read(s.target_path)
        est, sr3 = sf.read(s.est_path)
        if not (sr1 == sr2 == sr3):
            raise ValueError(f"Sample rate mismatch for {s.id}")
        mix = mix[:, 0]
        target = target[:, 0]
        est = est.mean(axis=1) if est.ndim>1 else est
        info = s.info
        speech_pos = info['speech_pos']
        Q = info['Q']
        
        if self.region_type == 'angular':
            Qd = int(len(speech_pos))
        else:
            Qd = info['Q_d']
        
        if self.region_type == 'spherical':
            Qa = int(len(speech_pos))
        else:
            Qa = info['Q_a']
        
        return {
            'id': s.id,
            'mix': mix, 
            'target': target, 
            'est': est,
            'Q': Q, 
            'Qa': Qa, 
            'Qd': Qd,
            'sr': sr1
        }

def collate_fn(batch):
    collated = {}
    for key in batch[0]:
        if key in ['mix', 'target', 'est']:
            collated[key] = [b[key] for b in batch]
        elif key == 'id':
            collated[key] = [b[key] for b in batch]
        else:
            collated[key] = torch.tensor([b[key] for b in batch])
    return collated

# --- Evaluation Loop ---
def evaluate_dataset(config, dataset, batch_size, num_workers, collate_fn=None):
    with open(config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    sr = cfg['audio']['sample_rate']
    n_fft = int(sr * cfg['audio']['stft']['window_size_ms'] / 1000)
    hop = int(sr * cfg['audio']['stft']['hop_size_ms'] / 1000)
    win_type = cfg['audio']['stft']['window_type'].lower()
    mae_weight = cfg['training']['loss']['Q_eq_0']['lambda']
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)
    results = []
    mae_losses = []
    snr_losses = []
    for batch in tqdm(loader, total=len(loader)):
        for i in range(len(batch['id'])):
            sid = batch['id'][i]
            Q = int(batch['Q'][i].item())
            mix = batch['mix'][i]
            target = batch['target'][i]
            est = batch['est'][i]
            sr = int(batch['sr'][i].item())
            Qa = int(batch['Qa'][i].item())
            Qd = int(batch['Qd'][i].item())
            entry = {'id': sid, 'Q': Q, 'Qa': Qa, 'Qd': Qd}
            if Q == 0:
                entry['EnergyDecay'] = compute_energy_decay(est, mix)
                mae = compute_freq_mae_loss(n_fft, hop, win_type, torch.tensor(est), weight=mae_weight).item()
                mae_losses.append(mae)
            else:
                min_len = min(len(target), len(est))
                target = target[:min_len]
                est = est[:min_len]
                entry['SDR'] = compute_sdr(target, est)
                snr = compute_snr_loss(torch.tensor(target), torch.tensor(est), snr_max_db=30.0, weight=1.0).item()
                snr_losses.append(snr)
                if Q == 1:
                    entry['STOI'] = compute_stoi(target, est, sr)
                    entry['PESQ'] = compute_pesq(target, est, sr)
            results.append(entry)
            print(f"Evaluated {sid}: {entry}")
    if len(mae_losses) > 0:
        avg_mae = np.mean(mae_losses)
    else:
        avg_mae = 100.0
    if len(snr_losses) > 0:
        avg_snr = np.mean(snr_losses)
    else:
        avg_snr = 100.0
    return results, avg_mae, avg_snr

# --- Summarize and Save ---
def summarize_and_save(results, output_txt, mae, snr):
    total_count = len(results)
    neg_sdr_count = sum(1 for r in results if ('SDR' in r) and (r['SDR'] < 0.0))
    low_ed_count = sum(1 for r in results if ('EnergyDecay' in r) and (r['EnergyDecay'] <= 15.0))
    rel_groups = {
        'Q=0/Qa=Qd=0': [],
        'Q=0/Qa>Qd=0': [],
        'Q=0/Qd>Qa=0': [],
        'Q=0/Qa=Qd=1': [],
        'Q=1/Qa=Qd=1': [],
        'Q=1/Qa>Qd=1': [],
        'Q=1/Qd>Qa=1': [],
        'Q=2/Qa=Qd=2': [],
    }
    q_groups = {
        'Q=0': [],
        'Q=1': [],
        'Q=2': [],
    }
    for r in results:
        Q = r['Q']
        Qa = r['Qa']
        Qd = r['Qd']
        if Q in (0, 1, 2):
            q_groups[f'Q={Q}'].append(r)
        if Q == 0:
            if Qa == Qd:
                if Qa == 0:
                    key = f"Q=0/Qa=Qd=0"
                elif Qa == 1:
                    key = f"Q=0/Qa=Qd=1"
            elif Qa > Qd:
                key = f"Q=0/Qa>Qd={Q}"
            else:
                key = f"Q=0/Qd>Qa={Q}"
        elif Q == 1:
            if Qa == Qd:
                if Qa == 1:
                    key = f"Q=1/Qa=Qd=1"
                elif Qa == 2:
                    key = f"Q=1/Qa=Qd=2"
            elif Qa > Qd:
                key = f"Q=1/Qa>Qd={Q}"
            else:
                key = f"Q=1/Qd>Qa={Q}"
        elif Q == 2 and Qa == Qd:
            key = "Q=2/Qa=Qd=2"
        else:
            print(f"ID:{r[id]} does not belong to any group.")
        rel_groups[key].append(r)

    with open(output_txt, 'w') as f:
        f.write("=== Global Stats ===\n")
        f.write(f"Total samples: {total_count}\n")
        f.write(f"Num samples with SDR < 0 dB: {neg_sdr_count}\n")
        f.write(f"Num samples with EnergyDecay ≤ 15 dB: {low_ed_count}\n\n")
        f.write(f"Average MAE Loss (Q=0): {mae:.4f}\n")
        f.write(f"Average SNR Loss (Q=1,2): {snr:.4f}\n\n")
        
        f.write("=== Quantitative Evaluation (Averages) per QaQd ===\n")
        for grp, items in rel_groups.items():
            if not items:
                continue
            f.write(f"-- {grp} (Count: {len(items)}) --\n")
            metrics = {}
            for item in items:
                for k, v in item.items():
                    if k in ['id','Q','Qa','Qd']:
                        continue
                    metrics.setdefault(k, []).append(v)
            for m, vals in metrics.items():
                avg = np.mean(vals)
                f.write(f" {m}: {avg:.4f}\n")
        
        f.write("\n=== Quantitative Evaluation (Averages) per Q ===\n")
        for grp, items in q_groups.items():
            if not items:
                continue
            f.write(f"-- {grp} (Count: {len(items)}) --\n")
            metrics = {}
            for item in items:
                for k, v in item.items():
                    if k in ['id','Q','Qa','Qd']:
                        continue
                    metrics.setdefault(k, []).append(v)
            for m, vals in metrics.items():
                avg = np.mean(vals)
                f.write(f" {m}: {avg:.4f}\n")

        f.write("\n=== Best/Worst Samples per QaQd ===\n")
        for grp, items in rel_groups.items():
            if not items:
                continue
            key = 'EnergyDecay' if '0' in grp else 'SDR'
            sorted_items = sorted(items, key=lambda x: x.get(key, -np.inf))
            worst = [r['id'] for r in sorted_items[:3]]
            best = [r['id'] for r in sorted_items[-3:]]
            f.write(f"{grp} Best3({key}): {best}\n")
            f.write(f"{grp} Worst3({key}): {worst}\n")
        
        f.write("\n=== Best/Worst Samples per Q ===\n")
        for grp, items in q_groups.items():
            if not items:
                continue
            key = 'EnergyDecay' if '0' in grp else 'SDR'
            sorted_items = sorted(items, key=lambda x: x.get(key, -np.inf))
            worst = [r['id'] for r in sorted_items[:3]]
            best = [r['id'] for r in sorted_items[-3:]]
            f.write(f"{grp} Best3({key}): {best}\n")
            f.write(f"{grp} Worst3({key}): {worst}\n")

# --- Main ---
def evaluation(args):
    output_eval_dir = os.path.join(args.output_dir, "evaluation", f"{args.est_dir.split('/')[-2]}")
    os.makedirs(output_eval_dir, exist_ok=True)
    output_txt_path = os.path.join(output_eval_dir, f"{args.est_dir.split('/')[-1]}.txt")
    dataset = EvaluationDataset(args.test_dir, args.est_dir, args.region_type)
    results, mae, snr = evaluate_dataset(args.config, dataset, args.batch_size, args.num_workers, collate_fn=collate_fn)
    summarize_and_save(results, output_txt_path, mae, snr)