import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Literal

# =================================================================================
# Feature‑aggregation blocks referenced in ReZero (Concat, TAC, TAA, RNN, RNN‑Loop)
# =================================================================================

# x: [B * T_spec, N, Bw]

class ConcatAggregator(nn.Module):
    """
    Concatenate features along the feature dimension.
    Input: x of shape [B * T_spec, N, Bw]
    Output: [B*T_spec, N*Bw]
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        BT_spec, N, Bw = x.shape
        return x.reshape(BT_spec, N * Bw)

class TAConcatenateAggregator(nn.Module):
    """
    Transform each view with a shared 3-layer MLP, then concatenate.
    Input: x of shape [B*T_spec, N, Bw]
    Output: [B*T_spec, N*P]
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.fc1(x))       # [B*T_spec, N, hidden_dim]
        out = F.relu(self.fc2(out))     # [B*T_spec, N, hidden_dim]
        out = F.relu(self.fc3(out))     # [B*T_spec, N, hidden_dim]
        BT_spec, N, P = out.shape
        return out.reshape(BT_spec, N * P)

class TAAverageAggregator(nn.Module):
    """
    Transform each view with a shared 3-layer MLP, then average.
    Input: x of shape [B * T_spec, N, Bw]
    Output: [B*T_spec, P]
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.fc1(x))        # [B*T_spec, N, hidden_dim]
        out = F.relu(self.fc2(out))      # [B*T_spec, N, hidden_dim]
        out = F.relu(self.fc3(out))      # [B*T_spec, N, hidden_dim]
        return out.mean(dim=1)           # [B*T_spec, hidden_dim]

class RNNAggregator(nn.Module):
    """
    Sequence modeling over views with an LSTM, take last output.
    Input:  x: [B*T_spec, N, Bw]
    Output: [B*T_spec, hidden_dim]
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc  = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)          # [B*T_spec, N, hidden_dim]
        out = out[:, -1, :]           
        out = self.fc(out)            # [B*T_spec, hidden_dim]
        return out

class RNNLoopAggregator(nn.Module):
    """
    Sequence modeling over views with an LSTM on a closed loop.
    Dynamically adapts to feature width P at runtime.
    Input: x of shape [B * T_spec, N, Bw]
    Output: [B*T_spec, 2*P] (P=hidden_dim)
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Create closed loop by appending first view to the end
        first = x[:, :1, :]                  # [B*T_spec, 1, Bw]
        loop = torch.cat([x, first], dim=1)  # [B*T_spec, N+1, Bw]

        # RNN across (N+1) views
        out, _ = self.rnn(loop)              # [B*T_spec, N+1, hidden_dim]
        out = self.fc(out)                   # [B*T_spec, N+1, hidden_dim]
        out = out[:, -2:, :]                 # [B*T_spec, 2, hidden_dim]
        BT_spec, N, P = out.shape

        return out.reshape(BT_spec, N * P)   # [B*T_spec, 2*hidden_dim]


def build_aggregator(method: Literal["concat", "tac", "taa", "rnn", "rnn-loop"], input_dim: int = None, hidden_dim: int = None) -> nn.Module:
    """Return aggregator module according to *method* string."""
    m = method.lower()
    if m in ("tac", "taa"):
        assert input_dim is not None and hidden_dim is not None, "TAC/TAA には input_dim と hidden_dim が必要"
    if m in ("rnn", "rnn-loop"):
        assert hidden_dim is not None, "RNN/RNN-Loop には hidden_dim が必要"
        
    if m == "concat":
        return ConcatAggregator()
    elif m == "tac":
        return TAConcatenateAggregator(input_dim, hidden_dim)
    elif m == "taa":
        return TAAverageAggregator(input_dim, hidden_dim)
    elif m == "rnn":
        return RNNAggregator(input_dim, hidden_dim)
    elif m == "rnn-loop":
        return RNNLoopAggregator(input_dim, hidden_dim)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")