import os
import yaml
import torch
import torch.nn as nn
from src.ADCRezero.models.aggregation import build_aggregator
from src.ADCRezero.models.feature_extraction import FeatureExtraction
from src.ADCRezero.models.BandSplitRNN.bandsplitLSTM import BandSequenceModelModule
from src.ADCRezero.models.BandSplitRNN.maskestimation import MaskEstimationModule

class ARezeroModel(nn.Module):
    """
    Conical Region ReZero model combining feature extraction, subband processing,
    region aggregation, and multi-channel BSRNN.
    """
    def __init__(self, args, device: str = "cuda", return_mask: bool = False, complex_as_channel: bool = True):
        super().__init__()
        if not args.region_type == 'angular':
            raise ValueError("AReZero model only supports 'angular' region type.")
        
        # Load config
        cfg_file = args.config
        with open(cfg_file, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        model_cfg = cfg['model']
        audio_cfg = cfg['audio']

        # Feature extraction module
        self.fe = FeatureExtraction(args, device)

        # Subband definitions
        self.subbands = self.fe.subbands
        K = len(self.subbands)
        fc_dim = model_cfg['feature_dim']
        region_dim = model_cfg['region_feature_dim'] # P

        # Spectrogram branch: Norm + FC per subband (real+imag)
        self.spec_norms = nn.ModuleList([
            nn.LayerNorm([2 * (e - s)]) for s, e in self.subbands
        ])
        self.spec_fcs = nn.ModuleList([
            nn.Linear(2 * (e - s), fc_dim) for s, e in self.subbands
        ])

        # IPD branch: Norm + FC per subband
        Mp = len(self.fe.pairs)
        self.ipd_norms = nn.ModuleList([
            nn.LayerNorm([Mp * (e - s)]) for s, e in self.subbands
        ])
        self.ipd_fcs = nn.ModuleList([
            nn.Linear(Mp * (e - s), fc_dim) for s, e in self.subbands
        ])
        
        # Region aggregator, Norm + FC per subband
        agg_cfg = model_cfg['aggregation']
        agg_type = agg_cfg['method'].lower()
        Nviews = self.fe.Nviews
        self.region_aggs = nn.ModuleList([
            build_aggregator(agg_type, input_dim=(e - s), hidden_dim=region_dim)
            for (s, e) in self.subbands
        ])
        
        if agg_type == 'rnn-loop':
            N_agg = 2
        elif agg_type == 'rnn' or agg_type == 'taa':
            N_agg = 1
        else:
            N_agg = Nviews
        
        self.region_norms = nn.ModuleList([
            nn.LayerNorm(N_agg * region_dim) for _ in range(K)
        ])
        self.region_fcs = nn.ModuleList([
            nn.Linear(N_agg * region_dim, fc_dim) for _ in range(K)
        ])

        # Band-split & sequence modeling (BSRNN)
        self.band_sequence = BandSequenceModelModule(
            input_dim_size=fc_dim,
            hidden_dim_size=model_cfg['bsrnn_hidden_dim'],
            rnn_type='LSTM',
            bidirectional_T=False,
            bidirectional_K=True,
            num_layers=model_cfg['blocks']
        )
        
        # Mask estimation
        self.mask_estimation = MaskEstimationModule(
            sr=audio_cfg['sample_rate'],
            n_fft=self.fe.n_fft,
            bandsplits=self.subbands,
            fc_dim=fc_dim,
            mlp_dim=fc_dim*2,
            num_channels=1
        )
        
        self.return_mask = return_mask
        self.cac = complex_as_channel

    def forward(self, mix: torch.Tensor,
                        theta_l: torch.Tensor,
                        theta_h: torch.Tensor,
                        ) -> torch.Tensor:
        # Feature extraction
        merged = self.fe(mix, theta_l, theta_h)
        spec_full = merged['spec']               # [B, C, F, T]
        spec_bands = merged['spec_bands']        # List of [B, C, (e-s), T]
        ipd_bands = merged['ipd_bands']          # List of [B, Mp, (e-s), T]
        V_bands = merged['V_theta_bands']        # List of [B, Nviews, (e-s), T]

        B = mix.size(0)
        T_spec = spec_bands[0].shape[-1]
        band_feats = []
        
        # Process each subband
        for k, (s, e) in enumerate(self.subbands):
            # --- Spectrogram branch ---
            spec_band = spec_bands[k]                                                      # [B, C, (e-s), T_spec]
            spec_band_ref = spec_band[:, 0, :, :].squeeze(1)                               # [B, (e-s), T_spec]
            real = spec_band_ref.real                                                      # [B, (e-s), T_spec]
            imag = spec_band_ref.imag                                                      # [B, (e-s), T_spec]
            feat_spec_band_ref = torch.cat([real, imag], dim=1)                            # [B, 2*(e-s), T_spec]
            feat_spec_band_ref = feat_spec_band_ref.permute(0, 2, 1)                       # [B, T_spec, 2*(e-s)]
            feat_spec_band_ref = self.spec_fcs[k](self.spec_norms[k](feat_spec_band_ref))  # [B, T_spec, fc_dim]

            # --- IPD branch ---
            ipd = ipd_bands[k]
            feat_ipd = ipd.permute(0, 3, 1, 2).reshape(B, T_spec, -1)   # [B, T_spec, Mp*(e-s)]
            feat_ipd = self.ipd_fcs[k](self.ipd_norms[k](feat_ipd))     # [B, T_spec, fc_dim]

            # --- Region (direction) branch ---
            V = V_bands[k].permute(0, 3, 1, 2)
            B, T_spec, Nviews, Bw = V.shape
            V = V.reshape(B * T_spec, Nviews, Bw)                       # [B*T_spec, Nviews, Bw]
            ragg = self.region_aggs[k](V)                               # [B*T_spec, N_agg*P] (P=region_dim)
            ragg = self.region_norms[k](ragg)                           # [B*T_spec, P]
            _, P = ragg.shape
            ragg = ragg.view(B, T_spec, P)                              # [B, T_spec, P]
            feat_reg = self.region_fcs[k](ragg)                         # [B, T_spec, fc_dim]

            # --- Merge subband features ---
            merged = feat_spec_band_ref + feat_ipd + feat_reg
            band_feats.append(merged.unsqueeze(1))

        # Stack all subbands and process
        x = torch.cat(band_feats, dim=1)                                      # [B, K, T_spec, fc_dim]
        x = self.band_sequence(x)                                             # [B, K, T_spec, fc_dim]
        mask = self.mask_estimation(x)                                        # [B, 1, F, T_spec]
        x_hat = mask if self.return_mask else mask * spec_full[:, 0:1, :, :]  # [B, 1, F, T]
        return x_hat