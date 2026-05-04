"""
Baseline models for comparison with PatchTST.
1. DLinear - Decomposition Linear model (Zeng et al., 2022)
2. Vanilla Transformer - Standard Transformer without patching
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# DLinear: Decomposition + Linear
# ============================================================

class MovingAvg(nn.Module):
    """Moving average block for trend decomposition."""

    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        """x: (batch, seq_len, channels)"""
        # Pad front and end
        front = x[:, :1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):
    """Series decomposition into trend and residual."""

    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size)

    def forward(self, x):
        trend = self.moving_avg(x)
        residual = x - trend
        return residual, trend


class DLinear(nn.Module):
    """
    DLinear: Decomposition-Linear model (Zeng et al., 2022).

    Decomposes input into trend and seasonal (residual) components, then applies
    linear layers to each. Matches the official DLinear in LTSF-Linear / PatchTST
    repos.

    Args:
        enc_in: number of channels
        seq_len: input length
        pred_len: prediction horizon
        kernel_size: moving average kernel for trend decomposition (default 25)
        individual: if True, use separate linear layers per channel (more params).
                    Default False = shared weights, matching official DLinear.
    """

    def __init__(self, enc_in=7, seq_len=336, pred_len=96, kernel_size=25,
                 individual=False):
        super().__init__()
        self.enc_in = enc_in
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual

        self.decomp = SeriesDecomp(kernel_size)

        if individual:
            # Per-channel linear layers
            self.linear_seasonal = nn.ModuleList([
                nn.Linear(seq_len, pred_len) for _ in range(enc_in)
            ])
            self.linear_trend = nn.ModuleList([
                nn.Linear(seq_len, pred_len) for _ in range(enc_in)
            ])
        else:
            # Shared linear layers (official default)
            self.linear_seasonal = nn.Linear(seq_len, pred_len)
            self.linear_trend = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        """
        x: (batch, seq_len, enc_in)
        returns: (batch, pred_len, enc_in)
        """
        # Decompose
        seasonal_init, trend_init = self.decomp(x)

        # Permute to (batch, enc_in, seq_len) for linear over time dim
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        if self.individual:
            seasonal_out = torch.zeros(seasonal_init.size(0), self.enc_in, self.pred_len,
                                        dtype=seasonal_init.dtype, device=seasonal_init.device)
            trend_out = torch.zeros_like(seasonal_out)
            for i in range(self.enc_in):
                seasonal_out[:, i, :] = self.linear_seasonal[i](seasonal_init[:, i, :])
                trend_out[:, i, :] = self.linear_trend[i](trend_init[:, i, :])
        else:
            seasonal_out = self.linear_seasonal(seasonal_init)
            trend_out = self.linear_trend(trend_init)

        x = seasonal_out + trend_out
        return x.permute(0, 2, 1)  # (batch, pred_len, enc_in)


# ============================================================
# Vanilla Transformer (without patching - point-wise tokens)
# ============================================================

class TokenEmbedding(nn.Module):
    """Project each time step to d_model dimension."""

    def __init__(self, c_in, d_model):
        super().__init__()
        self.projection = nn.Linear(c_in, d_model)

    def forward(self, x):
        return self.projection(x)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class VanillaTransformer(nn.Module):
    """
    Standard Transformer encoder for time series forecasting.
    Uses point-wise tokens (no patching) for comparison with PatchTST.
    This is the channel-mixing version where all variables are embedded together.
    """

    def __init__(
        self,
        enc_in=7,
        seq_len=96,
        pred_len=96,
        d_model=128,
        n_heads=8,
        e_layers=2,
        d_ff=256,
        dropout=0.2,
    ):
        super().__init__()
        self.enc_in = enc_in
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Embedding
        self.embedding = TokenEmbedding(enc_in, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len + 10)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)

        # Prediction head
        self.head = nn.Linear(seq_len * d_model, pred_len * enc_in)

    def forward(self, x):
        """
        x: (batch, seq_len, enc_in)
        returns: (batch, pred_len, enc_in)
        """
        batch_size = x.shape[0]

        # Embed + positional encoding
        x = self.embedding(x)          # (batch, seq_len, d_model)
        x = self.pos_enc(x)

        # Transformer encode
        x = self.encoder(x)            # (batch, seq_len, d_model)

        # Flatten and project
        x = x.reshape(batch_size, -1)  # (batch, seq_len * d_model)
        x = self.head(x)               # (batch, pred_len * enc_in)
        x = x.reshape(batch_size, self.pred_len, self.enc_in)

        return x


if __name__ == '__main__':
    # Test DLinear
    model = DLinear(enc_in=7, seq_len=336, pred_len=96)
    x = torch.randn(32, 336, 7)
    out = model(x)
    print(f"DLinear - Input: {x.shape}, Output: {out.shape}")
    print(f"DLinear params: {sum(p.numel() for p in model.parameters()):,}")

    # Test Vanilla Transformer (smaller seq_len due to memory)
    model2 = VanillaTransformer(enc_in=7, seq_len=96, pred_len=96)
    x2 = torch.randn(32, 96, 7)
    out2 = model2(x2)
    print(f"Transformer - Input: {x2.shape}, Output: {out2.shape}")
    print(f"Transformer params: {sum(p.numel() for p in model2.parameters()):,}")
