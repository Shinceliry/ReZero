import os
import yaml
import torch
import torch.nn as nn
from src.ADCRezero.models.feature_extraction import FeatureExtraction
from src.ADCRezero.models.BandSplitRNN.bandsplitLSTM import BandSequenceModelModule
from src.ADCRezero.models.BandSplitRNN.maskestimation import MaskEstimationModule

class DRezeroModel(nn.Module):
    """
    Conical Region ReZero model combining feature extraction, subband processing,
    region aggregation, and multi-channel BSRNN.
    """
    def __init__(self, args, device: str = "cuda", return_mask: bool = False, complex_as_channel: bool = True):
        super().__init__()
        if not args.region_type == 'spherical':
            raise ValueError("DReZero model only supports 'spherical' region type.")
        
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
        
        # ILD branch: same dims
        Mp = len(self.fe.pairs)
        self.ild_norms = nn.ModuleList([
            nn.LayerNorm([Mp * (e - s)]) for s, e in self.subbands
        ])
        self.ild_fcs = nn.ModuleList([
            nn.Linear(Mp * (e - s), fc_dim) for s, e in self.subbands
        ])

        # Distance Embedding Generator (DEG)
        deg_cfg = cfg['model']['distance_embedding']
        embed_dim = deg_cfg['embedding_dim']
        self.degs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, region_dim), nn.Tanh(),
                nn.Linear(region_dim, region_dim), nn.Tanh(),
                nn.Linear(region_dim, embed_dim), nn.Tanh()
            ) for _ in range(K)
        ])
        self.dist_fcs = nn.ModuleList([
            nn.Linear(embed_dim, fc_dim) for _ in range(K)
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
                        d_query: torch.Tensor,
                        ) -> torch.Tensor:
        # Feature extraction
        feats = self.fe(mix)
        spec_full = feats['spec']          # [B, C, F, T]
        spec_bands = feats['spec_bands']   # List of [B, C, (e-s), T]
        ild_bands = feats['ild_bands']     # List of [B, Mp, (e-s), T]

        B = mix.size(0)
        d_query = d_query.unsqueeze(-1)
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

            # --- ILD branch ---
            ild = ild_bands[k]
            feat_ild = ild.permute(0, 3, 1, 2).reshape(B, T_spec, -1)   # [B, T_spec, Mp*(e-s)]
            feat_ild = self.ild_fcs[k](self.ild_norms[k](feat_ild))     # [B, T_spec, fc_dim]

            # --- Distance branch ---
            dist_emb = self.degs[k](d_query)
            feat_dist = self.dist_fcs[k](dist_emb)
            feat_dist = feat_dist.unsqueeze(1).expand(-1, T_spec, -1)   # [B, T_spec, fc_dim]

            # --- Merge subband features ---
            merged = feat_spec_band_ref + feat_ild + feat_dist
            band_feats.append(merged.unsqueeze(1))

        # Stack all subbands and process
        x = torch.cat(band_feats, dim=1)                                      # [B, K, T_spec, fc_dim]
        x = self.band_sequence(x)                                             # [B, K, T_spec, fc_dim]
        mask = self.mask_estimation(x)                                        # [B, 1, F, T_spec]
        x_hat = mask if self.return_mask else mask * spec_full[:, 0:1, :, :]  # [B, 1, F, T]
        return x_hat   