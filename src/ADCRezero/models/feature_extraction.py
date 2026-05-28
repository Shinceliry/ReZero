import os
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import math

class FeatureExtraction(nn.Module):
    """Extract STFT-based spatial features Complex Spectrogram, IPD, ILD, V(θ) for ADCReZero."""
    def __init__(self, args, device: str = "cuda") -> None:
        super().__init__()
        self.region_type = args.region_type
        self.device = device

        cfg_file = args.config
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.dataset_cfg = cfg["dataset_generation"]
        self.audio_cfg = cfg["audio"]
        model_cfg = cfg["model"]

        # STFT
        self.sr = int(self.audio_cfg["sample_rate"])
        win_ms = self.audio_cfg["stft"]["window_size_ms"]
        hop_ms = self.audio_cfg["stft"]["hop_size_ms"]
        self.n_fft = int(self.sr * win_ms / 1000)
        self.hop = int(self.sr * hop_ms / 1000)

        if self.audio_cfg["stft"]["window_type"].lower() == "hann":
            self.window = torch.hann_window(self.n_fft, device=self.device)
        else:
            self.window = torch.ones(self.n_fft, device=self.device)

        # Microphone pairs
        if args.mic_arch == 'circular':
            self.n_mics = cfg["microphone_array"]["circular"]["channels"]
        elif args.mic_arch == 'linear':
            self.n_mics = cfg["microphone_array"]["linear"]["channels"]
        self.pairs: List[Tuple[int, int]] = [
            (i, j)
            for i in range(self.n_mics)
            for j in range(i + 1, self.n_mics)
        ]
        
        if args.mic_arch == 'circular':
            radius = cfg["microphone_array"]['circular']['diameter_m'] / 2
            coords = []
            for i in range(self.n_mics):
                angle = 2 * math.pi * i / self.n_mics
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                coords.append([x, y, 0.0])
            self.mic_array = torch.tensor(coords, device=self.device, dtype=torch.float)
        elif args.mic_arch == 'linear':
            aperture = cfg["microphone_array"]['linear']['aperture_m']
            coords = []
            for i in range(self.n_mics):
                x = i * aperture / (self.n_mics - 1)
                coords.append([x, 0.0, 0.0])
            self.mic_array = torch.tensor(coords, device=self.device, dtype=torch.float)

        # Subbands
        self.subbands = self._make_subbands(model_cfg["subband_scheme"])
        self.Nviews = model_cfg["aggregation"]["fixed_number"]["spatial_views"]
        self.ang_min, self.ang_max = self.dataset_cfg["query_region"]["angular_width_deg"]

    def forward(self,
                mix: torch.Tensor,
                theta_l: torch.Tensor = None,
                theta_h: torch.Tensor = None,) -> dict:
        """
        inputs:
            mix         : [B, C, T]  B: バッチ数, C: チャネル数, T: 時間サンプル数
            theta_l     (Tensor[B])
            theta_h     (Tensor[B])

        Returns dict with:
            spec         : [B, C, F, T_spec]               F: 周波数ビン数
            spec_bands   : list of [B, C, Bw, T_spec]      Bw: サブバンド幅ビン数
            ipd_bands    : list of [B, Mp, Bw, T_spec]      Mp: マイクペア数
            ild_bands    : list of [B, Mp, Bw, T_spec]
            V_theta_bands: list of [B, Nviews, Bw, T_spec] Nviews: 角度サンプリング数
        """
        # STFT
        mix = mix.to(self.device)
        B, C, T = mix.shape
        mix_flat = mix.reshape(B * C, T)             # [B*C, T]
        spec_flat = torch.stft(
            mix_flat,
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.window,
            center=True,
            return_complex=True,
        )                                            # [B*C, F, T_spec]
        F_bins, T_spec = spec_flat.shape[1], spec_flat.shape[2]
        spec = spec_flat.view(B, C, F_bins, T_spec)  # [B, C, F, T_spec]

        # IPD / ILD （サブバンドごと）
        spec_bands = [spec[:, :, s:e, :] for s, e in self.subbands]
        ipd_bands = [self._compute_ipd(sb) for sb in spec_bands]
        ild_bands = [self._compute_ild(sb) for sb in spec_bands]

        # 方向特徴 V(θ)（サブバンドごと）
        if not self.region_type == "spherical":
            mic_array = self.mic_array.unsqueeze(0).repeat(B, 1, 1)
            V_theta_bands = self.compute_V_theta_bands(
                ipd_bands, mic_array, theta_l, theta_h
            )
            
            if self.region_type == "conical":
                return {
                    "spec": spec,
                    "spec_bands": spec_bands,
                    "ipd_bands": [x.float() for x in ipd_bands],
                    "ild_bands": [x.float() for x in ild_bands],
                    "V_theta_bands": [x.float() for x in V_theta_bands],
                }
            elif self.region_type == "angular":
                return {
                    "spec": spec,
                    "spec_bands": spec_bands,
                    "ipd_bands": [x.float() for x in ipd_bands],
                    "V_theta_bands": [x.float() for x in V_theta_bands],
                }
        elif self.region_type == "spherical":
            return {
                "spec": spec,
                "spec_bands": spec_bands,
                "ild_bands": [x.float() for x in ild_bands],
            }

    def _make_subbands(self, scheme_cfg: list) -> List[Tuple[int, int]]:
        F_bins  = self.n_fft // 2 + 1
        freq_res = (self.sr / 2) / (F_bins - 1)

        subbands, used = [], 0
        for blk in scheme_cfg:
            if "bandwidth_hz" in blk:
                width = max(1, int(blk["bandwidth_hz"] / freq_res))
                for _ in range(blk["count"]):
                    start, end = used, min(used + width, F_bins)
                    subbands.append((start, end))
                    used = end
            else:
                for _ in range(blk["count"]):
                    if used < F_bins:
                        subbands.append((used, F_bins))
                    used = F_bins
        return subbands

    def _compute_ipd(self, spec_band: torch.Tensor) -> torch.Tensor:
        """
        Compute IPD per subband.
        spec_band: [B, C, Bw, T_spec] → returns [B, Mp, Bw, T_spec]
        """
        phases = torch.angle(spec_band)
        diffs = [
            phases[:, i] - phases[:, j]
            for i, j in self.pairs
        ]
        ipd = torch.stack(diffs, dim=0)                                 # [Mp, B, Bw, T_spec]
        ipd = ipd.permute(1, 0, 2, 3)                                   # [B, Mp, Bw, T_spec]
        ipd = torch.remainder(ipd + torch.pi, 2 * torch.pi) - torch.pi  # wrap to [-π, π]
        return ipd

    def _compute_ild(self, spec_band: torch.Tensor) -> torch.Tensor:
        """
        Compute ILD per subband.
        spec_band: [B, C, Bw, T_spec] → returns [B, Mp, Bw, T_spec]
        """
        mag   = spec_band.abs()
        ilogs = [
            20 * torch.log10((mag[:, i] + 1e-6) / (mag[:, j] + 1e-6))
            for i, j in self.pairs
        ]
        ild = torch.stack(ilogs, dim=0)                 # [Mp, B, Bw, T_spec]
        return ild.permute(1, 0, 2, 3)                  # [B, Mp, Bw, T_spec]

    # TPD
    def compute_tpd(self,
                    mic_array: torch.Tensor,
                    theta: torch.Tensor,
                    phi: torch.Tensor,
                    n_fft: int,
                    sr: int,
                    speed_of_sound: float = 343.0
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batch & view aware TDOA calculation.
        mic_array: [B, C, 3]
        theta, phi: [B, Nviews]
        returns:
            tpd: [B, Nviews, Mp, F]
        """
        # direction unit vectors u: [B, Nviews, 3]
        theta_rad = torch.deg2rad(theta)
        phi_rad = torch.deg2rad(phi)
        ux = torch.cos(phi_rad) * torch.cos(theta_rad)
        uy = torch.cos(phi_rad) * torch.sin(theta_rad)
        uz = torch.sin(phi_rad)
        u = torch.stack([ux, uy, uz], dim=-1)                       # [B, Nviews, 3]

        # baseline for each mic-pair: [B, Mp, 3]
        idx_i = torch.tensor([i for i, _ in self.pairs], device=self.device)
        idx_j = torch.tensor([j for _, j in self.pairs], device=self.device)
        # baseline = mic_array[:, idx_i, :] - mic_array[:, idx_j, :]  # [B, Mp, 3]
        baseline = torch.abs(mic_array[:, idx_i, :] - mic_array[:, idx_j, :])

        # projected distances d_p: [B, Nviews, Mp]
        d_p = (baseline.unsqueeze(1) * u.unsqueeze(2)).sum(dim=-1) # [B, Nviews, Mp]: TDOA
        tau_p = d_p * sr / speed_of_sound                          # [B, Nviews, Mp]

        k = torch.arange(n_fft // 2 + 1, device=self.device)       # [F]
        tpd = 2 * torch.pi * (tau_p.unsqueeze(-1) * k.unsqueeze(0).unsqueeze(0) / n_fft)  # [B, Nviews, Mp, F]
        return tpd

    # V(θ) (bandwise)
    def compute_V_theta_bands(self,
                            ipd_bands: List[torch.Tensor],
                            mic_array: torch.Tensor,
                            theta_l: torch.Tensor,
                            theta_h: torch.Tensor
                            ) -> List[torch.Tensor]:
        """
        Compute directional feature V(theta) per subband.
        ipd_bands: list of [B, Mp, Bw, T_spec]
        mic_pos:   [B, C, 3]
        theta_l/h: [B]
        returns:   list of [B, Nviews, Bw, T_spec]
        """
        # Angle sampling: [B, Nviews]
        angles = self.sample_fixed_number(self.Nviews, theta_l, theta_h)
        phi = torch.zeros_like(angles)                          # [B, Nviews], φ = 0で固定

        # compute full TDOA: [B, Nviews, Mp, F]
        tpd = self.compute_tpd(
            mic_array, angles, phi,
            n_fft=self.n_fft, sr=self.sr
        )

        V_bands = []
        for band_idx, (s, e) in enumerate(self.subbands):
            ipd = ipd_bands[band_idx]                           # [B, Mp, Bw, T_spec]
            tpd_band = tpd[:, :, :, s:e]                        # [B, Nviews, Mp, Bw]
            theo = torch.exp(-1j * tpd_band.unsqueeze(-1))      # [B, Nviews, Mp, Bw, 1]
            obs = torch.exp(1j * ipd).unsqueeze(1)              # [B, 1, Mp, Bw, T_spec]
            V = (obs * theo).real                               # [B, Nviews, Mp, Bw, T_spec]
            V = V.sum(dim=2, keepdim=False)                     # [B, Nviews, Bw, T_spec]
            
            # 代表ペア: 最長基線ペア
            idx_i = torch.tensor([i for i, _ in self.pairs], device=self.device)
            idx_j = torch.tensor([j for _, j in self.pairs], device=self.device)
            baseline = torch.abs(mic_array[:, idx_i, :] - mic_array[:, idx_j, :])    # [B, Mp, 3]
            ref_pair = torch.argmax(torch.norm(baseline[0], dim=-1)).item()
            f_center = (s + e - 1) // 2
            
            # TPD をキーに Nviews を昇順ソート（TPDはτに単調なのでTDOA順と同等）
            tpd_ref = tpd[:, :, ref_pair, f_center]                            # [B, Nviews]
            sort_idx = torch.argsort(tpd_ref, dim=1)                           # [B, Nviews]
            V = V.gather(
                dim=1,
                index=sort_idx[:, :, None, None].expand(V.size(0), V.size(1), V.size(2), V.size(3))
            ) 
            V_bands.append(V)

        return V_bands

    # Angle sampling utilities (Fixed number)
    def sample_fixed_number(self, num_views: int,
                        theta_l: torch.Tensor,
                        theta_h: torch.Tensor) -> torch.Tensor:
        """
        theta_l, theta_h: [B] (deg)
        return: angles [B, num_views] (deg, in 0..360)
        """
        if num_views < 2:
            raise ValueError("num_views must be 2 or greater.")
        device, dtype = theta_l.device, theta_l.dtype
        theta_h_wrapped = torch.where(theta_l > theta_h, theta_h + 360.0, theta_h)   # [B]
        total = (theta_h_wrapped - theta_l) % 360.0                                  # [B]
        step = total / (num_views - 1)                                               # [B]
        idx  = torch.arange(num_views, device=device, dtype=dtype).unsqueeze(0)      # [1, N]
        angles = (theta_l.unsqueeze(1) + idx * step.unsqueeze(1)) % 360.0            # [B, N]
        return angles