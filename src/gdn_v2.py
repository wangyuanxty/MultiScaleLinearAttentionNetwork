"""
Gated DeltaNet-2 (NVIDIA, 2026) — pure PyTorch, faithful to official code.
https://github.com/NVlabs/GatedDeltaNet-2

Recurrence:
  S_t = (I - k_erase · k^T) · diag(decay) · S_{t-1} + k · v_write^T   [S: Dk×Dv]
  o_t = S_t^T · q_t                                                      [o: Dv]

  k_erase = k ⊙ b   (channel-wise erase)
  v_write = v ⊙ w   (channel-wise write)
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


class GDN2Block(nn.Module):
    """GDN-2 token-mixing layer — exact NVIDIA architecture, pure PyTorch."""

    def __init__(self, d_model=64, head_dim=16, num_heads=4, expand_v=2.0, conv_size=4, dropout=0.1,
                 use_physics=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_k_dim = head_dim
        self.head_v_dim = int(head_dim * expand_v)
        self.key_dim = num_heads * self.head_k_dim
        self.value_dim = num_heads * self.head_v_dim

        self.use_physics = use_physics
        if use_physics:
            # gamma = softplus(raw): always positive (IR/T can only accelerate decay)
            self.gamma_ir = nn.Parameter(torch.full((1,), -4.0))  # softplus -> ~0.018
            self.gamma_t = nn.Parameter(torch.full((1,), -4.0))
            self.Ea_log = nn.Parameter(torch.tensor(4.0))  # Ea ~55 kJ/mol

        self.q_proj = nn.Linear(d_model, self.key_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.key_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.value_dim, bias=False)
        self.q_conv = CausalConv1d(self.key_dim, conv_size)
        self.k_conv = CausalConv1d(self.key_dim, conv_size)
        self.v_conv = CausalConv1d(self.value_dim, conv_size)

        self.f_proj = nn.Sequential(
            nn.Linear(d_model, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.key_dim, bias=False),
        )
        self.b_proj = nn.Linear(d_model, self.key_dim, bias=False)
        self.w_proj = nn.Linear(d_model, self.value_dim, bias=False)

        self.A_log = nn.Parameter(torch.log(torch.empty(num_heads).uniform_(1, 16)))
        dt = torch.exp(torch.rand(self.key_dim)*(math.log(.1)-math.log(.001))+math.log(.001)).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(torch.log(-torch.expm1(-dt)) + dt)

        self.g_proj = nn.Sequential(
            nn.Linear(d_model, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.value_dim, bias=False),
        )
        self.out_norm = nn.RMSNorm(self.head_v_dim)
        self.out_proj = nn.Linear(self.value_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, phys_mod=None, return_state=False):
        B, L, D = x.shape
        H = self.num_heads

        q = self.q_conv(self.q_proj(x))
        k = self.k_conv(self.k_proj(x))
        v = self.v_conv(self.v_proj(x))
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        raw_g = self.f_proj(x).view(B, L, H, self.head_k_dim)
        g = -self.A_log.exp().view(1, 1, H, 1) * F.softplus(raw_g + self.dt_bias.view(1, 1, H, self.head_k_dim))
        if phys_mod is not None:
            gamma_ir = F.softplus(self.gamma_ir)  # >= 0: IR only accelerates decay
            ir = phys_mod[:, :, 0:1]
            modulation = gamma_ir * ir
            if phys_mod.shape[-1] >= 2:
                gamma_t = F.softplus(self.gamma_t)
                t = phys_mod[:, :, 1:2]
                R_gas = 8.314
                # Arrhenius term relative to T_ref=298K: O(1) at room temp, ~4x at 45C.
                # Absolute exp(-Ea/RT) is ~1e-10 and contributes nothing to decay.
                ea = torch.exp(self.Ea_log) * 1000.0  # kJ/mol -> J/mol
                arrhenius = torch.exp(-ea / R_gas * (1.0 / (t + 273.15) - 1.0 / 298.0))
                modulation = modulation + gamma_t * arrhenius
            g = g + modulation.view(B, -1, 1, 1)

        b = self.b_proj(x).sigmoid()
        w = self.w_proj(x).sigmoid()

        Dk, Dv = self.head_k_dim, self.head_v_dim
        q = q.view(B, L, H, Dk).transpose(1, 2)
        k = k.view(B, L, H, Dk).transpose(1, 2)
        v = v.view(B, L, H, Dv).transpose(1, 2)
        g = g.transpose(1, 2)
        b = b.view(B, L, H, Dk).transpose(1, 2)
        w = w.view(B, L, H, Dv).transpose(1, 2)

        o, S = self._scan(q, k, v, b, w, g, return_state=True)

        o = o.transpose(1, 2).contiguous().view(B, L, H, Dv)
        o = self.out_norm(o)
        g_out = self.g_proj(x).view(B, L, H, Dv)
        o = o * F.silu(g_out)
        out = self.dropout(self.out_proj(o.view(B, L, self.value_dim)))
        return (out, S) if return_state else out

    def _scan(self, q, k, v, b, w, g, return_state=False):
        B, H, L, Dk = k.shape
        Dv = v.shape[-1]
        decay = torch.exp(g)
        I = torch.eye(Dk, device=k.device, dtype=k.dtype).view(1, 1, Dk, Dk)
        S = torch.zeros(B, H, Dk, Dv, device=k.device, dtype=k.dtype)
        out = []
        for t in range(L):
            kt, vt, qt = k[:, :, t], v[:, :, t], q[:, :, t]
            bt, wt, dt = b[:, :, t], w[:, :, t], decay[:, :, t]
            erase = I - (kt * bt).unsqueeze(-1) * kt.unsqueeze(-2)
            write = kt.unsqueeze(-1) * (vt * wt).unsqueeze(-2)
            S = erase @ (dt.unsqueeze(-1) * S) + write
            out.append((S.transpose(-1, -2) @ qt.unsqueeze(-1)).squeeze(-1))
        out = torch.stack(out, dim=2)
        return (out, S) if return_state else out


class GDNv2Backbone(nn.Module):
    def __init__(self, d_model=64, num_layers=2, num_heads=4, head_dim=16,
                 expand_v=2.0, conv_size=4, dropout=0.1, input_dim=1, window_size=64):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'gdn': GDN2Block(d_model, head_dim, num_heads, expand_v, conv_size, dropout),
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
        x = self.input_proj(x)
        for l in self.layers:
            x = l['norm'](l['gdn'](x) + x)
        return self.head(x)


def build_gdn_v2(window_size=64, **kw):
    return GDNv2Backbone(
        d_model=kw.get('d_model', 64), num_layers=kw.get('num_layers', 2),
        num_heads=kw.get('num_heads', 4), head_dim=kw.get('head_dim', 16),
        expand_v=kw.get('expand_v', 2.0), conv_size=kw.get('conv_size', 4),
        dropout=kw.get('dropout', 0.1), input_dim=1, window_size=window_size,
    )
