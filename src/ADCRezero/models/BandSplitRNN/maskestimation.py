import torch
import torch.nn as nn
import typing as tp

class GLU(nn.Module):
    """
    GLU Activation Module.
    """
    def __init__(self, input_dim: int):
        super(GLU, self).__init__()
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, input_dim * 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        return x[..., :self.input_dim] * self.sigmoid(x[..., self.input_dim:])

class MLP(nn.Module):
    """
    Simple MLP with tanh activation and GLU output.
    """
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            activation_type: str = 'tanh',
    ):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            self.select_activation(activation_type)(),
            nn.Linear(hidden_dim, output_dim),
            GLU(output_dim)
        )

    @staticmethod
    def select_activation(activation_type: str) -> nn.modules.activation:
        if activation_type == 'tanh':
            return nn.Tanh
        elif activation_type == 'relu':
            return nn.ReLU
        elif activation_type == 'gelu':
            return nn.GELU
        else:
            raise ValueError(f"Unsupported activation: {activation_type}")

    def forward(self, x: torch.Tensor):
        return self.mlp(x)

class MaskEstimationModule(nn.Module):
    """
    MaskEstimation Module of BandSplitRNN.
    Forms the T-F output (complex or real) per subband and concatenates.
    """
    def __init__(
            self,
            sr: int,
            n_fft: int,
            bandsplits: tp.List[tp.Tuple[int, int]],
            fc_dim: int = 128,
            mlp_dim: int = 512,
            complex_as_channel: bool = True,
            num_channels: int = 1,
    ):
        super(MaskEstimationModule, self).__init__()
        frequency_mul = num_channels
        if complex_as_channel:
            frequency_mul *= 2
        self.cac = complex_as_channel
        self.frequency_mul = frequency_mul
        self.bandwidths = [e - s for (s, e) in bandsplits]
        
        self.layernorms = nn.ModuleList([
            nn.LayerNorm(fc_dim)
            for _ in self.bandwidths
        ])
        self.mlp = nn.ModuleList([
            MLP(fc_dim, mlp_dim, bw * self.frequency_mul, activation_type='tanh')
            for bw in self.bandwidths
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, k_subbands, time, fc_dim]
        Returns:
            Tensor of shape [batch_size, C(n_channels), freq, time] (complex if cac=True)
        """
        outs = []
        for i, bw in enumerate(self.bandwidths):
            out = self.layernorms[i](x[:, i])        # [B, T, fc_dim]
            out = self.mlp[i](out)                   # [B, T, bw*frequency_mul]
            B, T, FM = out.shape
            if self.cac:
                # Rearrange to complex tensor
                out = out.permute(0, 2, 1).contiguous()               # [B, FM, T]
                out = out.view(B, -1, 2, FM // self.frequency_mul, T) # [B, C, 2, F, T]
                out = out.permute(0, 1, 3, 4, 2)                      # [B, C, F, T, 2]
                out = out.contiguous()                                # [B, C, F, T, 2]
                out = torch.view_as_complex(out)                      # [B, C, F, T]
            else:
                out = out.view(B, -1, FM // self.frequency_mul, T)
            outs.append(out)
        return torch.cat(outs, dim=-2)