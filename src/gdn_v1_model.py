"""
Gated DeltaNet v1 (ICLR 2025, NVIDIA) — pure PyTorch.
Single scalar gate beta controls both erase and write.
Architecture follows NVIDIA ref_gated_deltanet/lit_gpt/gated_delta_net.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CausalConv1d(nn.Module):
    def __init__(self, dim, kernel_size=4, bias=False):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size, groups=dim, padding=kernel_size - 1, bias=bias)
        self.act = nn.SiLU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)[..., : -self.conv.kernel_size[0] + 1]
        return self.act(x).transpose(1, 2)


class GatedDeltaNet_v1(nn.Module):
    """Single GDN-v1 token-mixing layer."""

    def __init__(self, d_model=64, expand_k=1.0, expand_v=2.0, num_heads=4, conv_size=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        key_dim = int(d_model * expand_k)
        value_dim = int(d_model * expand_v)

        self.q_proj = nn.Linear(d_model, key_dim, bias=False)
        self.k_proj = nn.Linear(d_model, key_dim, bias=False)
        self.v_proj = nn.Linear(d_model, value_dim, bias=False)
        self.q_conv = CausalConv1d(key_dim, conv_size)
        self.k_conv = CausalConv1d(key_dim, conv_size)
        self.v_conv = CausalConv1d(value_dim, conv_size)

        self.gk_proj = nn.Linear(d_model, num_heads)
        self.b_proj = nn.Linear(d_model, num_heads)

        A = torch.empty(num_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        dt_min, dt_max = 0.001, 0.1
        dt = torch.exp(torch.rand(num_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        self.dt_bias = nn.Parameter(torch.log(-torch.expm1(-dt)) + dt)

        self.g_proj = nn.Linear(d_model, value_dim)
        self.out_norm = nn.RMSNorm(value_dim // num_heads)
        self.out_proj = nn.Linear(value_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape
        H = self.num_heads

        q = self.q_conv(self.q_proj(x))
        k = self.k_conv(self.k_proj(x))
        v = self.v_conv(self.v_proj(x))

        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        forget = -self.A_log.exp() * F.softplus(self.gk_proj(x) + self.dt_bias)
        beta = self.b_proj(x).sigmoid()

        Dk = q.size(-1) // H
        Dv = v.size(-1) // H
        q = q.view(B, L, H, Dk).transpose(1, 2)
        k = k.view(B, L, H, Dk).transpose(1, 2)
        v = v.view(B, L, H, Dv).transpose(1, 2)
        forget = forget.transpose(1, 2)
        beta = beta.transpose(1, 2)

        decay = torch.exp(forget)
        I = torch.eye(Dk, device=q.device, dtype=q.dtype).view(1, 1, 1, Dk, Dk)
        S = torch.zeros(B, H, Dv, Dk, device=q.device, dtype=q.dtype)
        out = []

        for t in range(L):
            d, b = decay[:, :, t], beta[:, :, t]
            kt, vt, qt = k[:, :, t], v[:, :, t], q[:, :, t]
            kk = kt.unsqueeze(-1) * kt.unsqueeze(-2)
            vk = vt.unsqueeze(-1) * kt.unsqueeze(-2)
            S = S @ (d.unsqueeze(-1).unsqueeze(-1) * (I - b.unsqueeze(-1).unsqueeze(-1) * kk)) \
                + b.unsqueeze(-1).unsqueeze(-1) * vk
            out.append((S @ qt.unsqueeze(-1)).squeeze(-1))

        o = torch.stack(out, dim=2).transpose(1, 2).contiguous().view(B, L, H, Dv)
        o = self.out_norm(o)
        g = self.g_proj(x).view(B, L, H, Dv)
        o = o * F.silu(g)
        return self.dropout(self.out_proj(o.view(B, L, -1)))


def build_gdn_v1(window_size=64, **kw):
    """Build a GDN v1 backbone for time series."""
    d_model = kw.get('d_model', 64)
    num_layers = kw.get('num_layers', 2)
    num_heads = kw.get('num_heads', 4)
    expand_k = kw.get('expand_k', 1.0)
    expand_v = kw.get('expand_v', 2.0)
    conv_size = kw.get('conv_size', 4)
    dropout = kw.get('dropout', 0.1)

    class GDNv1Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Linear(1, d_model)
            self.pos_embed = nn.Parameter(torch.randn(1, window_size, d_model) * 0.02)
            self.layers = nn.ModuleList([
                nn.ModuleDict({
                    'gdn': GatedDeltaNet_v1(d_model, expand_k, expand_v, num_heads, conv_size, dropout),
                    'norm': nn.RMSNorm(d_model),
                }) for _ in range(num_layers)
            ])
            self.head = nn.Sequential(
                nn.RMSNorm(d_model), nn.Flatten(),
                nn.Linear(window_size * d_model, 128), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(128, 1),
            )

        def forward(self, x):
            B, L, _ = x.shape
            x = self.input_proj(x) + self.pos_embed[:, :L, :]
            for layer in self.layers:
                x = layer['norm'](layer['gdn'](x) + x)
            return self.head(x)

    return GDNv1Backbone()
