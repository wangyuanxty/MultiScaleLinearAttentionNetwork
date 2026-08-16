"""
Gated DeltaFormer — Multi-scale seq2vec Transformer for battery RUL prediction.

Configurable components (toggleable for ablation):
  --multiscale:       3-branch capacity path (patch 2/8/64)
  --cross_exchange:   Gated cross-scale exchange (requires --multiscale)
  --learnable_ps:     STE-learnable patch sizes
  --num_quantiles:    1 for point, 3 for P10/P50/P90 (default 1)
  --output_len:       Seq2vec output length K (default 64)

Multi-variable: concatenate [C, V, I, T, IR] as input channels.
No dual-path. No cross-attention. No autoregression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from gdn_v2 import GDN2Block


# ─── Cross-scale exchange ─────────────────────────────────

class CrossScaleExchange(nn.Module):
    """Gated cross-scale exchange between multi-scale branches (scalar gate)."""
    def __init__(self, d_model=64):
        super().__init__()
        self.gate_fn = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 1), nn.Sigmoid())
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, h_a, h_b):
        p_a = h_a.mean(dim=1)
        p_b = h_b.mean(dim=1)
        gate = self.gate_fn(torch.cat([p_a, p_b], dim=-1)).unsqueeze(1)
        return gate * h_a + (1 - gate) * self.proj(h_b)


class StageQueryCrossExchange(nn.Module):
    """Stage-aware cross-scale interaction, V3: GDN state as query.
    The coarse branch's GDN-2 state S (the compressed degradation
    history = stage information) is projected to a single query; the
    fine/mid hidden reps provide K/V. Single-query attention → O(L),
    fully linear, no gating, additive residual.

    state_dim = H*Dk*Dv (flattened GDN state)."""
    def __init__(self, d_model=64, d_k=32, state_dim=2048):
        super().__init__()
        self.W_q = nn.Linear(state_dim, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

    def forward(self, S_coarse, h_fine, h_mid):
        """S_coarse: (B, state_dim) flattened GDN state.
        h_fine/h_mid: (B, T, d) aligned. Returns (h_fine', h_mid')."""
        q = self.W_q(S_coarse)  # (B, d_k) single stage query

        def attend(h_in):
            k = self.W_k(h_in)   # (B, T, d_k)
            v = self.W_v(h_in)   # (B, T, d)
            scores = torch.bmm(k, q.unsqueeze(-1)).squeeze(-1) \
                / (self.W_k.out_features ** 0.5)  # (B, T)
            attn = F.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
            return attn * v  # per-position weighted values (O(L))

        c_f = attend(h_fine)
        c_m = attend(h_mid)
        h_f = h_fine + c_f  # additive residual, no gate
        h_m = h_mid + c_m
        return h_f, h_m


# ─── Single GDN branch ────────────────────────────────────

class SingleBranch(nn.Module):
    """Single GDN v2 branch with optional patching and physics gate."""
    def __init__(self, patch_size, input_dim, d_model, num_layers,
                 num_heads, head_dim, expand_v, conv_size, dropout,
                 use_physics=False, ir_ch=None, t_ch=None):
        super().__init__()
        self.patch_size = patch_size
        self.use_physics = use_physics
        self.ir_ch = ir_ch
        self.t_ch = t_ch
        proj_in = patch_size * input_dim if patch_size > 1 else input_dim
        self.proj = nn.Linear(proj_in, d_model)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'gdn': GDN2Block(d_model, head_dim, num_heads, expand_v, conv_size, dropout,
                                 use_physics=use_physics),
                'norm': nn.RMSNorm(d_model),
            }) for _ in range(num_layers)
        ])

    def make_phys_mod(self, x, phys):
        B, L, C = x.shape
        ps = self.patch_size
        phys_mod = None
        if self.use_physics:
            if phys is not None:
                if ps > 1:
                    pad = (ps - L % ps) % ps
                    phys_mod = F.pad(phys, (0, 0, 0, pad)).unfold(1, ps, ps).mean(dim=-1)
                else:
                    phys_mod = phys
            elif self.ir_ch is not None:
                ch_list = []
                ir_raw = x[:, :, self.ir_ch:self.ir_ch+1]  # (B, L, 1)
                if ps > 1:
                    pad = (ps - L % ps) % ps
                    ch_list.append(F.pad(ir_raw, (0, 0, 0, pad)).unfold(1, ps, ps).mean(dim=-1))
                else:
                    ch_list.append(ir_raw)
                if self.t_ch is not None:
                    t_raw = x[:, :, self.t_ch:self.t_ch+1]
                    if ps > 1:
                        pad = (ps - L % ps) % ps
                        ch_list.append(F.pad(t_raw, (0, 0, 0, pad)).unfold(1, ps, ps).mean(dim=-1))
                    else:
                        ch_list.append(t_raw)
                phys_mod = torch.cat(ch_list, dim=-1)  # (B, L/ps, 1) or (B, L/ps, 2)
        return phys_mod

    def init_proj(self, x):
        """Patch + project input -> (B, L/ps, d_model)."""
        B, L, C = x.shape
        ps = self.patch_size
        if ps == 1:
            return self.proj(x.view(B, L, -1))
        pad = (ps - L % ps) % ps
        xp = F.pad(x, (0, 0, 0, pad)) if pad else x
        return self.proj(xp.unfold(1, ps, ps).reshape(B, -1, ps * C))

    def apply_layer(self, h, idx, phys_mod=None, return_state=False):
        layer = self.layers[idx]
        if return_state:
            out, S = layer['gdn'](h, phys_mod, return_state=True)
            return layer['norm'](out + h), S
        return layer['norm'](layer['gdn'](h, phys_mod) + h)

    def restore_len(self, h, L):
        ps = self.patch_size
        if ps > 1 and h.size(1) < L:
            h = h.unsqueeze(2).expand(-1, -1, ps, -1).reshape(h.size(0), -1, h.size(-1))[:, :L, :]
        return h

    def forward(self, x, phys=None):
        B, L, C = x.shape
        phys_mod = self.make_phys_mod(x, phys)
        h = self.init_proj(x)
        for layer in self.layers:
            h = layer['norm'](layer['gdn'](h, phys_mod) + h)
        return self.restore_len(h, L)


# ─── Main model ───────────────────────────────────────────

class PhysicsReadout(nn.Module):
    """Structural physics head (PiDDM-style): GDN state → kinetic params →
    time-dependent decay rate → one-cycle Euler integration.

    r_t = -(k1·n^a1 + k2·n^a2)   (k via softplus → r structurally negative)
    Q_hat_{t+1} = Q_t + r_t      (monotonic fade by construction, no penalty)

    n is the normalized cycle age ∈ (0, 1]; a1/a2 are data-calibrated per
    window from the GDN state, so the rate can decelerate (SEI-like) or
    accelerate (LLI/knee-like) over life — unlike the constant-rate prior
    used in the rejected physics-regularizer experiments.
    """

    def __init__(self, d_in, hidden=128, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 2),  # (a1, a2) raw exponents
        )
        self.k1_log = nn.Parameter(torch.tensor(-6.9))  # softplus ≈ 0.001
        self.k2_log = nn.Parameter(torch.tensor(-6.9))

    def forward(self, h, n):
        """h: (B, d) GDN state; n: (B,) normalized cycle age."""
        a1, a2 = self.mlp(h).chunk(2, dim=-1)  # (B, 1) each
        k1, k2 = F.softplus(self.k1_log), F.softplus(self.k2_log)
        n_ = n.to(h.dtype).unsqueeze(-1).clamp(1e-4, 1.0)
        r = -(k1 * n_.pow(a1) + k2 * n_.pow(a2))  # structurally < 0
        return r


class PhysicsIRReadout(nn.Module):
    """Physics-feature rate head: r = softplus(w·h) + softplus(gamma)·IR_last.

    The per-cycle decay rate decomposes into a learned data-driven part
    and a physics-feature part; gamma via softplus is always positive,
    so IR can only accelerate decay (SEI-growth direction). Structural
    monotonicity: Q_hat = Q_last - r <= Q_last.
    Trained in ABSOLUTE space (no per-window z-score).
    """

    def __init__(self, d_in, dropout=0.1):
        super().__init__()
        self.rate_proj = nn.Linear(d_in, 1)
        self.gamma_log = nn.Parameter(torch.tensor(-4.0))  # softplus ≈ 0.018

    def forward(self, h, ir_last=None):
        r = F.softplus(self.rate_proj(h))
        if ir_last is not None:
            r = r + F.softplus(self.gamma_log) * ir_last
        return r  # (B, 1), >= 0 by construction


class GDNBatteryModel(nn.Module):
    """Gated DeltaFormer: multi-scale seq2vec for battery RUL prediction.

    Single-path: [C, V, I, T, IR] concat → multi-scale GDN-2 → seq2vec output.
    No dual-path. No cross-attention. No autoregression.
    """

    def __init__(self, patch_size=2, multiscale=False, cross_exchange=False,
                 stage_query=False, learnable_ps=False, d_model=64,
                 num_layers=2, num_heads=4, head_dim=16, expand_v=2.0,
                 conv_size=4, dropout=0.1, input_dim=1, window_size=64,
                 output_len=64, num_quantiles=1, use_physics=False,
                 ir_ch=None, t_ch=None, readout="mean"):
        super().__init__()
        self.multiscale = multiscale
        self.cross_exchange = cross_exchange and multiscale
        self.stage_query = stage_query and multiscale and not self.cross_exchange
        self.num_quantiles = num_quantiles
        self.output_len = output_len

        self.readout = readout

        if multiscale:
            patch_sizes = (2, 4, 8)
            if learnable_ps:
                self.ps_alphas = nn.Parameter(torch.ones(3) * 0.5)
                self._ps_mins = (1, 3, 6)
                self._ps_maxs = (3, 6, 16)
            else:
                self.ps_alphas = None
            self.branches = nn.ModuleList([
                SingleBranch(ps, input_dim, d_model, num_layers, num_heads,
                             head_dim, expand_v, conv_size, dropout,
                             use_physics=(use_physics and i == 2),
                             ir_ch=ir_ch, t_ch=t_ch)
                for i, ps in enumerate(patch_sizes)
            ])
        else:
            self.branches = nn.ModuleList([
                SingleBranch(patch_size, input_dim, d_model, num_layers, num_heads,
                             head_dim, expand_v, conv_size, dropout,
                             use_physics=use_physics,
                             ir_ch=ir_ch, t_ch=t_ch)
            ])

        if self.cross_exchange:
            self.cross = nn.ModuleList([
                CrossScaleExchange(d_model) for _ in range(num_layers)
            ])
        if self.stage_query:
            state_dim = num_heads * head_dim * int(head_dim * expand_v)
            self.cross_stage = nn.ModuleList([
                StageQueryCrossExchange(d_model, state_dim=state_dim)
                for _ in range(num_layers)
            ])

        n_branches = 3 if multiscale else 1
        head_in = n_branches * d_model
        self.head_phys = None
        if readout == "phys":
            self.head_phys = PhysicsReadout(head_in, dropout=dropout)
        self.head_phys_ir = None
        if readout == "phys_ir":
            self.head_phys_ir = PhysicsIRReadout(head_in, dropout=dropout)
        self.head_cap = nn.Sequential(
            nn.Linear(head_in, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, num_quantiles * output_len),
        )
        self.head_rul = nn.Sequential(
            nn.Linear(head_in, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x, return_rul=False, phys=None, n=None):
        """x: (B, L, C) — capacity + optional features concatenated.
        phys: (B, L, n_phys) — dedicated physics channel (IR/T), only feeds the gate.
        n: (B,) normalized cycle age — required when readout == "phys".

        Returns:
          return_rul=False: (B, Q, K) or (B, K) capacity prediction
          return_rul=True:  ((B, K), (B, 1)) capacity + RUL
        """
        B, L, _ = x.shape
        if self.cross_exchange and len(self.branches) == 3 and self.branches[0].patch_size > 1:
            # Per-layer cross-scale exchange: branches interleave GDN layers
            # with gated cross-branch mixing (vs post-hoc mixing in fallback).
            phys_mods = [b.make_phys_mod(x, phys if b.use_physics else None) for b in self.branches]
            outs = [b.init_proj(x) for b in self.branches]
            L_patch = max(h.size(1) for h in outs)  # align to longest (patch 2)
            for layer_idx in range(len(self.branches[0].layers)):
                outs = [b.apply_layer(h, layer_idx, pm) for b, h, pm in zip(self.branches, outs, phys_mods)]
                if layer_idx < len(self.cross):
                    # align all branches to L_patch before mixing (ceil+repeat+trim)
                    import math as _math
                    aligned = []
                    for b, h in zip(self.branches, outs):
                        if h.size(1) < L_patch:
                            reps = _math.ceil(L_patch / h.size(1))
                            h = h.repeat(1, reps, 1)[:, :L_patch, :]
                        aligned.append(h)
                    cross_mod = self.cross[layer_idx]
                    fm = cross_mod(aligned[0], aligned[1])
                    mf = cross_mod(aligned[1], aligned[0])
                    mc = cross_mod(aligned[1], aligned[2])
                    cm = cross_mod(aligned[2], aligned[1])
                    outs = [fm, (mf + mc) / 2, cm]
            # restore to original length L for fused readout
            outs = [b.restore_len(h, L) for b, h in zip(self.branches, outs)]
        elif self.stage_query and len(self.branches) == 3 and self.branches[0].patch_size > 1:
            # StageQuery V3: coarse-branch GDN state acts as the query;
            # fine/mid reps are read via single-query linear attention.
            phys_mods = [b.make_phys_mod(x, phys if b.use_physics else None) for b in self.branches]
            outs = [b.init_proj(x) for b in self.branches]
            L_patch = max(h.size(1) for h in outs)
            import math as _math
            for layer_idx in range(len(self.branches[0].layers)):
                S_c = None
                new_outs = []
                for i, (b, h, pm) in enumerate(zip(self.branches, outs, phys_mods)):
                    if i == 2:  # coarse branch: also return the GDN state
                        h_new, S_c = b.apply_layer(h, layer_idx, pm, return_state=True)
                    else:
                        h_new = b.apply_layer(h, layer_idx, pm)
                    new_outs.append(h_new)
                outs = new_outs
                if layer_idx < len(self.cross_stage):
                    aligned = []
                    for h in outs:
                        if h.size(1) < L_patch:
                            reps = _math.ceil(L_patch / h.size(1))
                            h = h.repeat(1, reps, 1)[:, :L_patch, :]
                        aligned.append(h)
                    S_flat = S_c.reshape(S_c.size(0), -1)
                    h_f, h_m = self.cross_stage[layer_idx](S_flat, aligned[0], aligned[1])
                    outs = [h_f, h_m, aligned[2]]
            outs = [b.restore_len(h, L) for b, h in zip(self.branches, outs)]
        else:
            outs = []
            for branch in self.branches:
                h = branch(x, phys=phys if branch.use_physics else None)
                outs.append(h)

            if self.cross_exchange and len(outs) == 3:
                for cross in self.cross:
                    fm = cross(outs[0], outs[1])
                    mf = cross(outs[1], outs[0])
                    mc = cross(outs[1], outs[2])
                    cm = cross(outs[2], outs[1])
                    outs = [fm, (mf + mc) / 2, cm]

        if len(outs) == 3:
            h_c = outs[2].mean(dim=1, keepdim=True).expand(-1, L, -1)
            fused = torch.cat([outs[0][:, :L, :], outs[1][:, :L, :], h_c], dim=-1)
        else:
            fused = outs[0][:, :L, :]

        h_pooled = fused.mean(dim=1) if self.readout != "last" else fused[:, -1, :]
        h_rul = fused[:, -1, :]  # last time step for RUL
        if self.readout == "phys":
            # structural physics: Q_next = Q_last + r(n), r < 0 by construction
            assert n is not None, "readout='phys' requires normalized cycle age n"
            r = self.head_phys(h_pooled, n)          # (B, 1)
            out_cap = x[:, -1, 0:1] + r              # (B, 1) in normalized space
            if return_rul:
                return out_cap, self.head_rul(h_rul)
            return out_cap
        if self.readout == "phys_ir":
            # physics-feature rate head: Q_next = Q_last - (learned + gamma*IR)
            # IR term optional: datasets without a physics channel use the
            # learned rate component only (same monotonic structure).
            ir_last = x[:, -1, 1:2] if x.size(-1) >= 2 else None
            r = self.head_phys_ir(h_pooled, ir_last)        # (B, 1) >= 0
            out_cap = x[:, -1, 0:1] - r                     # monotonic
            if return_rul:
                return out_cap, self.head_rul(h_rul)
            return out_cap
        out_cap = self.head_cap(h_pooled)
        out_cap = out_cap.view(B, self.num_quantiles, self.output_len)
        if self.num_quantiles == 1:
            out_cap = out_cap.squeeze(1)
        if return_rul:
            out_rul = self.head_rul(h_rul)
            return out_cap, out_rul
        return out_cap


# ─── Losses ───────────────────────────────────────────────

class PhysicsRegularizer(nn.Module):
    """Penalize mismatch between model-implied decay rate and Arrhenius rate.

    r_pred = -log(c_{t+1}/c_t)          (implied per-step decay from prediction)
    r_phys = base + gamma_ir*IR + gamma_t*arrhenius(T)
    L = lambda * |r_pred - r_phys|      (physical consistency as soft constraint)

    IR/T stay regular input features; physics constrains the OUTPUT behavior,
    so no information is removed and no dead-parameter shortcut exists.
    gamma_ir/gamma_t via softplus -> always positive (IR/T only accelerate decay).
    """

    def __init__(self, lambda_=1.0, base_init=0.004):
        super().__init__()
        self.lambda_ = lambda_
        self.base = nn.Parameter(torch.tensor(float(base_init)))
        self.gamma_ir = nn.Parameter(torch.full((1,), -4.0))   # softplus -> ~0.018
        self.gamma_t = nn.Parameter(torch.full((1,), -4.0))
        self.Ea_log = nn.Parameter(torch.tensor(4.0))          # ~55 kJ/mol

    def forward(self, pred, x_win, phys_last):
        """pred: (B,1); x_win: (B, W, 1) capacity window; phys_last: (B, n_phys) IR/T at t.

        r_pred = LONG-TERM average decay rate from window start to prediction point:
            r_pred = -log(pred / c_first) / (W + 1)
        Capacity regeneration (short 1-2 step recovery) is diluted by the window
        length, so the constraint targets the degradation trend, not instantaneous
        observations. Compatible with regeneration.
        """
        c_first = x_win[:, 0, 0:1]  # (B, 1)
        L = x_win.size(1)
        r_pred = -torch.log((pred / (c_first + 1e-6)).clamp(1e-6, 1e6)) / (L + 1)
        gamma_ir = F.softplus(self.gamma_ir)
        r_phys = self.base + gamma_ir * phys_last[:, 0:1]
        if phys_last.shape[-1] >= 2:
            gamma_t = F.softplus(self.gamma_t)
            t = phys_last[:, 1:2]
            ea = torch.exp(self.Ea_log) * 1000.0  # kJ/mol -> J/mol
            arrhenius = torch.exp(-ea / 8.314 * (1.0 / (t + 273.15) - 1.0 / 298.0))
            r_phys = r_phys + gamma_t * arrhenius
        return self.lambda_ * (r_pred - r_phys).abs().mean()


class PinballLoss(nn.Module):
    """Pinball (quantile) loss."""
    def __init__(self, quantiles=(0.1, 0.5, 0.9)):
        super().__init__()
        self.register_buffer('quantiles', torch.tensor(quantiles, dtype=torch.float32))

    def forward(self, pred, target, mask=None):
        err = target.unsqueeze(1) - pred
        q = self.quantiles.view(1, -1, 1)
        loss = torch.max(q * err, (q - 1) * err)
        if mask is not None:
            loss = loss * mask.unsqueeze(1)
            denom = mask.sum() * len(self.quantiles) + 1e-8
        else:
            denom = loss.numel()
        return loss.sum() / denom


def masked_mae(pred, target, mask=None):
    """MAE with optional mask."""
    err = (pred - target).abs()
    if mask is not None:
        err = err * mask
        return err.sum() / (mask.sum() + 1e-8)
    return err.mean()


# ─── Builder ──────────────────────────────────────────────

def build_gdn_model(patch_size=2, multiscale=False, cross_exchange=False,
                    stage_query=False, learnable_ps=False, window_size=64,
                    output_len=64, num_quantiles=1, use_physics=False,
                    ir_ch=None, t_ch=None, readout="mean", **kw):
    return GDNBatteryModel(
        use_physics=use_physics, ir_ch=ir_ch, t_ch=t_ch,
        patch_size=patch_size, multiscale=multiscale,
        cross_exchange=cross_exchange, stage_query=stage_query,
        learnable_ps=learnable_ps,
        d_model=kw.get('d_model', 64), num_layers=kw.get('num_layers', 2),
        num_heads=kw.get('num_heads', 4), head_dim=kw.get('head_dim', 16),
        expand_v=kw.get('expand_v', 2.0), conv_size=kw.get('conv_size', 4),
        dropout=kw.get('dropout', 0.1),
        input_dim=kw.get('input_dim', 1), window_size=window_size,
        output_len=output_len, num_quantiles=num_quantiles,
        readout=readout,
    )
