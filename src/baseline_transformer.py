"""
Standard Transformer baseline for battery capacity prediction.

A clean vanilla Transformer encoder that maps a capacity sequence window
to a single next-step capacity prediction.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model)"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class StandardTransformer(nn.Module):
    """
    Standard Transformer encoder for univariate time series forecasting.

    Architecture:
        Linear embedding → Positional Encoding → N encoder layers → Flatten → MLP → scalar
    """

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        input_dim: int = 1,
        window_size: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size

        # Input projection: scalar → d_model
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=window_size, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head
        self.flatten = nn.Flatten()
        self.head = nn.Sequential(
            nn.Linear(window_size * d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, 1) — univariate capacity sequence
        Returns: (B, 1) — next-step capacity prediction
        """
        B, L, _ = x.shape

        # Embed
        x = self.input_proj(x)              # (B, L, d_model)
        x = self.pos_encoder(x)             # (B, L, d_model)

        # Encode
        x = self.encoder(x)                 # (B, L, d_model)

        # Predict
        x = self.flatten(x)                 # (B, L*d_model)
        x = self.head(x)                    # (B, 1)

        return x


def build_transformer_baseline(
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
    window_size: int = 64,
    **kwargs,
) -> StandardTransformer:
    """Factory function for the standard Transformer baseline."""
    return StandardTransformer(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        input_dim=1,
        window_size=window_size,
    )


if __name__ == "__main__":
    # Quick sanity check
    model = build_transformer_baseline(window_size=64, d_model=64, nhead=4)
    x = torch.randn(8, 64, 1)  # (batch=8, seq_len=64, features=1)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
