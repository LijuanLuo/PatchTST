"""
PatchTST: A Time Series is Worth 64 Words (ICLR 2023)
Re-implementation of the core PatchTST model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RevIN(nn.Module):
    """Reversible Instance Normalization for time series."""

    def __init__(self, num_features, eps=1e-5, affine=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode='norm'):
        """
        x: (batch, seq_len, channels)
        mode: 'norm' or 'denorm'
        """
        if mode == 'norm':
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.stdev = (x.var(dim=1, keepdim=True, unbiased=False) + self.eps).sqrt().detach()
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            x = x * self.stdev + self.mean
        return x


class PatchEmbedding(nn.Module):
    """
    Segment time series into patches and project to embedding space.

    Matches official PatchTST behavior: always pads `stride` repeated copies of
    the last value at the end (via ReplicationPad1d). For L=336, P=16, S=8 this
    yields 42 patches (the "PatchTST/42" config from the paper).
    """

    def __init__(self, patch_len, stride, d_model, padding='end'):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding = padding
        # Always pad `stride` repeated copies of the last value at the end
        self.padder = nn.ReplicationPad1d((0, stride))
        self.projection = nn.Linear(patch_len, d_model)

    def forward(self, x):
        """
        x: (batch * n_vars, seq_len) -> (batch * n_vars, num_patches, d_model)

        After ReplicationPad1d((0, stride)) padding, sequence length becomes L+S,
        and num_patches = (L - P)/S + 1 + 1 (the "+1" comes from the pad).
        """
        # x: (batch * n_vars, seq_len)
        if x.dim() == 3:
            x = x.squeeze(1)

        # ReplicationPad1d expects (B, C, L) — add a channel dim then remove
        if self.padding == 'end':
            x = self.padder(x.unsqueeze(1)).squeeze(1)

        # Unfold into patches: (batch * n_vars, num_patches, patch_len)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        # Project patches to d_model
        x = self.projection(x)  # (batch * n_vars, num_patches, d_model)
        return x


class PositionalEncoding(nn.Module):
    """Learnable positional encoding for patches."""

    def __init__(self, d_model, max_len=128):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x):
        """x: (batch, num_patches, d_model)"""
        return x + self.pos_embed[:, :x.size(1), :]


class _BatchNorm1dWrapper(nn.Module):
    """
    Wraps BatchNorm1d to operate on (B, L, D) tensors by transposing internally.
    Matches the official PatchTST encoder, which uses BatchNorm with Transpose
    wrappers (paper footnote: BatchNorm outperforms LayerNorm for time series).
    """

    def __init__(self, d_model):
        super().__init__()
        self.bn = nn.BatchNorm1d(d_model)

    def forward(self, x):
        # x: (B, L, D) -> (B, D, L) -> BN -> (B, L, D)
        return self.bn(x.transpose(1, 2)).transpose(1, 2)


class AttentionEncoderLayer(nn.Module):
    """
    Custom Transformer encoder layer that exposes attention weights.

    Matches official PatchTST encoder defaults:
    - BatchNorm (not LayerNorm) — paper specifies BatchNorm via Transpose wrappers
    - Post-norm (norm AFTER residual add)
    - GELU activation
    - Dropout after attention output and after FFN
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, activation='gelu',
                 norm='BatchNorm'):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                                batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Paper defaults to BatchNorm; allow LayerNorm as fallback
        if 'batch' in norm.lower():
            self.norm1 = _BatchNorm1dWrapper(d_model)
            self.norm2 = _BatchNorm1dWrapper(d_model)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU() if activation == 'gelu' else nn.ReLU()

    def forward(self, x, return_attention=False):
        # Self-attention with optional weight extraction
        attn_out, attn_weights = self.self_attn(
            x, x, x,
            need_weights=return_attention,
            average_attn_weights=False,  # keep per-head weights for richer visualization
        )
        x = self.norm1(x + self.dropout1(attn_out))

        # Feed-forward
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm2(x + self.dropout2(ff_out))

        if return_attention:
            return x, attn_weights
        return x


class AttentionEncoder(nn.Module):
    """Stack of AttentionEncoderLayers that can return attention from all layers."""

    def __init__(self, d_model, nhead, dim_feedforward, num_layers,
                 dropout=0.1, activation='gelu', norm='BatchNorm'):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                   activation, norm=norm)
            for _ in range(num_layers)
        ])

    def forward(self, x, return_attention=False):
        all_attn = []
        for layer in self.layers:
            if return_attention:
                x, attn = layer(x, return_attention=True)
                all_attn.append(attn)
            else:
                x = layer(x)
        if return_attention:
            return x, all_attn  # list of (batch, n_heads, N, N)
        return x


class FlattenHead(nn.Module):
    """Flatten encoded patches and project to prediction horizon."""

    def __init__(self, d_model, num_patches, pred_len, head_dropout=0.0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(num_patches * d_model, pred_len)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        """
        x: (batch * n_vars, num_patches, d_model)
        returns: (batch * n_vars, pred_len)
        """
        x = self.flatten(x)       # (batch * n_vars, num_patches * d_model)
        x = self.linear(x)        # (batch * n_vars, pred_len)
        x = self.dropout(x)
        return x


class PatchTST(nn.Module):
    """
    PatchTST: Channel-Independent Patch Time Series Transformer.

    Key innovations:
    1. Patching: segments time series into subseries-level patches as Transformer tokens
    2. Channel-independence: each univariate series shares the same Transformer backbone

    Args:
        enc_in: number of input channels/variables (M)
        seq_len: look-back window length (L)
        pred_len: prediction horizon (T)
        patch_len: length of each patch (P), default 16
        stride: stride between patches (S), default 8
        d_model: Transformer latent dimension (D), default 128
        n_heads: number of attention heads (H), default 16
        e_layers: number of encoder layers, default 3
        d_ff: feed-forward dimension (F), default 256
        dropout: dropout rate, default 0.2
        head_dropout: prediction head dropout, default 0.0
        use_revin: whether to use RevIN, default True
        affine_revin: whether RevIN has learnable affine, default False
    """

    def __init__(
        self,
        enc_in=7,
        seq_len=336,
        pred_len=96,
        patch_len=16,
        stride=8,
        d_model=128,
        n_heads=16,
        e_layers=3,
        d_ff=256,
        dropout=0.2,
        head_dropout=0.0,
        use_revin=True,
        affine_revin=False,
    ):
        super().__init__()

        self.enc_in = enc_in
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.use_revin = use_revin

        # Calculate number of patches (matches official PatchTST/42 config)
        # After ReplicationPad1d((0, stride)) padding, L becomes L+S, so:
        #   num_patches = (L - P) / S + 1 + 1
        # For L=336, P=16, S=8: (336-16)/8 + 1 + 1 = 42
        self.num_patches = (seq_len - patch_len) // stride + 2

        # RevIN
        if use_revin:
            self.revin = RevIN(enc_in, affine=affine_revin)

        # Patch Embedding + Positional Encoding
        self.patch_embed = PatchEmbedding(patch_len, stride, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=self.num_patches + 10)

        # Transformer Encoder — custom layer that supports attention extraction
        self.encoder = AttentionEncoder(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            num_layers=e_layers,
            dropout=dropout,
            activation='gelu',
        )

        # Prediction Head
        self.head = FlattenHead(d_model, self.num_patches, pred_len, head_dropout)

    def forward(self, x, return_attention=False):
        """
        x: (batch, seq_len, enc_in) - multivariate input
        return_attention: if True, also returns list of attention weights per encoder layer

        Returns:
            (batch, pred_len, enc_in) - multivariate prediction
            (if return_attention) attention_maps: list of (batch*n_vars, n_heads, N, N) tensors
        """
        batch_size = x.shape[0]
        n_vars = self.enc_in

        # 1. RevIN normalization
        if self.use_revin:
            x = self.revin(x, mode='norm')  # (batch, seq_len, n_vars)

        # 2. Channel-independence: reshape to process each variable independently
        # (batch, seq_len, n_vars) -> (batch * n_vars, seq_len)
        x = x.permute(0, 2, 1)                     # (batch, n_vars, seq_len)
        x = x.reshape(batch_size * n_vars, self.seq_len)  # (batch * n_vars, seq_len)

        # 3. Patching + Projection
        x = self.patch_embed(x)     # (batch * n_vars, num_patches, d_model)

        # 4. Add positional encoding
        x = self.pos_enc(x)         # (batch * n_vars, num_patches, d_model)

        # 5. Transformer Encoder (with optional attention extraction)
        if return_attention:
            x, attn_maps = self.encoder(x, return_attention=True)
        else:
            x = self.encoder(x)     # (batch * n_vars, num_patches, d_model)

        # 6. Flatten + Linear Head
        x = self.head(x)            # (batch * n_vars, pred_len)

        # 7. Reshape back to multivariate
        x = x.reshape(batch_size, n_vars, self.pred_len)  # (batch, n_vars, pred_len)
        x = x.permute(0, 2, 1)     # (batch, pred_len, n_vars)

        # 8. RevIN denormalization
        if self.use_revin:
            x = self.revin(x, mode='denorm')

        if return_attention:
            return x, attn_maps
        return x


class PatchTST_CI_Only(nn.Module):
    """
    Ablation: Channel-Independence ONLY (no patching).
    Each time step is a token (P=1, S=1). Same as original TST with CI.
    Used to isolate the effect of channel-independence.
    """

    def __init__(self, enc_in=7, seq_len=336, pred_len=96,
                 d_model=128, n_heads=16, e_layers=3, d_ff=256,
                 dropout=0.2, use_revin=True):
        super().__init__()
        self.enc_in = enc_in
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.use_revin = use_revin

        if use_revin:
            self.revin = RevIN(enc_in)

        # Point-wise: project each single time step to d_model
        self.input_proj = nn.Linear(1, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len + 10)

        # Same BatchNorm-based encoder as full PatchTST for fair comparison
        self.encoder = AttentionEncoder(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            num_layers=e_layers, dropout=dropout, activation='gelu',
            norm='BatchNorm',
        )
        self.head = nn.Linear(seq_len * d_model, pred_len)

    def forward(self, x):
        batch_size = x.shape[0]
        n_vars = self.enc_in

        if self.use_revin:
            x = self.revin(x, mode='norm')

        # Channel-independence: (batch, seq, n_vars) -> (batch*n_vars, seq, 1)
        x = x.permute(0, 2, 1).reshape(batch_size * n_vars, self.seq_len, 1)

        x = self.input_proj(x)      # (B*M, L, D)
        x = self.pos_enc(x)
        x = self.encoder(x)         # (B*M, L, D)
        x = x.reshape(batch_size * n_vars, -1)
        x = self.head(x)            # (B*M, T)

        x = x.reshape(batch_size, n_vars, self.pred_len).permute(0, 2, 1)

        if self.use_revin:
            x = self.revin(x, mode='denorm')
        return x


class PatchTST_P_Only(nn.Module):
    """
    Ablation: Patching ONLY (no channel-independence = channel-mixing).
    All variables are concatenated in the patch dimension.
    Used to isolate the effect of patching.
    """

    def __init__(self, enc_in=7, seq_len=336, pred_len=96,
                 patch_len=16, stride=8,
                 d_model=128, n_heads=16, e_layers=3, d_ff=256,
                 dropout=0.2, use_revin=True):
        super().__init__()
        self.enc_in = enc_in
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.use_revin = use_revin

        if use_revin:
            self.revin = RevIN(enc_in)

        # Same patching scheme as full PatchTST for fair comparison:
        # always pad `stride` repeated copies at the end -> num_patches = (L-P)/S + 2
        self.num_patches = (seq_len - patch_len) // stride + 2

        # Channel-mixing: each patch token contains ALL variables
        # patch dim = patch_len * enc_in
        self.patch_embed = nn.Linear(patch_len * enc_in, d_model)
        self.patch_len = patch_len
        self.stride = stride
        self.padder = nn.ReplicationPad1d((0, stride))
        self.pos_enc = PositionalEncoding(d_model, max_len=self.num_patches + 10)

        # Same BatchNorm-based encoder as full PatchTST
        self.encoder = AttentionEncoder(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            num_layers=e_layers, dropout=dropout, activation='gelu',
            norm='BatchNorm',
        )
        self.head = nn.Linear(self.num_patches * d_model, pred_len * enc_in)

    def forward(self, x):
        batch_size = x.shape[0]

        if self.use_revin:
            x = self.revin(x, mode='norm')

        # x: (B, L, M) — keep all channels together (channel-mixing)
        # Pad: ReplicationPad1d expects (B, C, L), so transpose then transpose back
        x = x.permute(0, 2, 1)              # (B, M, L)
        x = self.padder(x)                  # (B, M, L+S)
        x = x.permute(0, 2, 1)              # (B, L+S, M)

        # Unfold using torch.unfold for efficiency
        # (B, L+S, M) -> patches of (B, num_patches, P, M)
        # Use as_strided/manual indexing
        patches = []
        for i in range(0, self.num_patches):
            start = i * self.stride
            patch = x[:, start:start + self.patch_len, :]   # (B, P, M)
            patches.append(patch.reshape(batch_size, -1))    # (B, P*M)

        x = torch.stack(patches, dim=1)  # (B, num_patches, P*M)
        x = self.patch_embed(x)          # (B, num_patches, D)
        x = self.pos_enc(x)
        x = self.encoder(x)              # (B, num_patches, D)

        x = x.reshape(batch_size, -1)
        x = self.head(x)                 # (B, T*M)
        x = x.reshape(batch_size, self.pred_len, self.enc_in)

        if self.use_revin:
            x = self.revin(x, mode='denorm')
        return x


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test all variants
    configs = [
        ('PatchTST (P+CI)', PatchTST(enc_in=7, seq_len=336, pred_len=96, d_model=128, n_heads=16, e_layers=3, d_ff=256)),
        ('CI Only',         PatchTST_CI_Only(enc_in=7, seq_len=96, pred_len=96, d_model=128, n_heads=16, e_layers=3, d_ff=256)),
        ('P Only',          PatchTST_P_Only(enc_in=7, seq_len=336, pred_len=96, d_model=128, n_heads=16, e_layers=3, d_ff=256)),
    ]

    for name, model in configs:
        sl = model.seq_len
        x = torch.randn(4, sl, 7)
        out = model(x)
        print(f"{name:20s}: ({4},{sl},7) -> {tuple(out.shape)}, params={count_parameters(model):,}")
