/** gdn2_mcu.c — GDN-2 C inference (v3: bit-exact with gdn_v2.py GDN2Block).
 *
 * Replicates the single-branch model exactly:
 *   input (W,1) -> patch embed (ps=2, Linear(2->D)) -> 2 x [GDN2Block + RMSNorm]
 *   -> restore_len -> last token -> GELU head -> scalar
 * GDN2Block per layer:
 *   q = L2norm(causal_conv(q_proj x)),  k = L2norm(causal_conv(k_proj x)),
 *   v = causal_conv(v_proj x)           (depthwise kernel K=4, SiLU act)
 *   g = -exp(A_log[h]) * softplus(f_proj(x) + dt_bias)     per (h, key-dim)
 *   b = sigmoid(b_proj x),  w = sigmoid(w_proj x)
 *   S_t = (I - (k*b)k^T) (decay*S_{t-1}) + k (v*w)^T ,  o_t = S_t^T q
 *   o -> per-head RMSNorm -> * silu(g_proj x) -> out_proj -> residual + RMSNorm(D)
 */
#pragma once
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "gdn2_mcu.h"

static float g_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static float g_silu(float x) { return x * g_sigmoid(x); }
static float g_softplus(float x) { return logf(1.0f + expf(x)); }
static float g_gelu(float x) {
    return 0.5f * x * (1.0f + erff(x / 1.41421356237f));
}

static void g_linear(const float *w, const float *x, float *y, int m, int n,
                     const float *b) {
    for (int i = 0; i < m; i++) {
        float s = b ? b[i] : 0.0f;
        for (int j = 0; j < n; j++) s += w[i * n + j] * x[j];
        y[i] = s;
    }
}

static void g_rmsnorm(float *x, int d, const float *w) {
    float ss = 0;
    for (int i = 0; i < d; i++) ss += x[i] * x[i];
    /* nn.RMSNorm(eps=None) uses float32 machine epsilon */
    float inv = 1.0f / sqrtf(ss / d + 1.1920929e-7f);
    for (int i = 0; i < d; i++) x[i] *= inv * w[i];
}

/* depthwise causal conv: PyTorch Conv1d(pad=K-1)[..., :-(K-1)] + SiLU
 * => y[t] = sum_j cw[j] * x[t-(K-1)+j]  (zero-padded on the left) */
static void g_causal_conv(const float *x, const float *cw, float *y,
                          int T, int C, int K) {
    for (int t = 0; t < T; t++) {
        for (int c = 0; c < C; c++) {
            float s = 0.0f;
            for (int j = 0; j < K; j++) {
                int tt = t - (K - 1) + j;
                if (tt >= 0) s += cw[c * K + j] * x[tt * C + c];
            }
            y[t * C + c] = g_silu(s);
        }
    }
}

/* L2-normalize each contiguous block of DK elements */
static void g_l2norm(float *x, int n, int DK) {
    for (int i = 0; i < n; i += DK) {
        float ss = 0;
        for (int j = 0; j < DK; j++) ss += x[i + j] * x[i + j];
        float inv = 1.0f / sqrtf(ss + 1e-12f);
        for (int j = 0; j < DK; j++) x[i + j] *= inv;
    }
}

/* GDN-2 scan: per-element decay along key dim */
static void gdn2_scan(const float *q, const float *k, const float *v,
                      const float *b, const float *w, const float *decay,
                      float *S, float *o, int L, int H, int DK, int DV) {
    for (int t = 0; t < L; t++) {
        for (int h = 0; h < H; h++) {
            const float *qt = q + t * H * DK + h * DK;
            const float *kt = k + t * H * DK + h * DK;
            const float *vt = v + t * H * DV + h * DV;
            const float *bt = b + t * H * DK + h * DK;
            const float *wt = w + t * H * DV + h * DV;
            const float *dt = decay + t * H * DK + h * DK;
            float *Sh = S + h * DK * DV;
            float Sh_new[DK * DV];
            for (int i = 0; i < DK; i++) {
                float ke = kt[i] * bt[i];
                for (int j = 0; j < DV; j++) {
                    float es = 0;
                    for (int p = 0; p < DK; p++)
                        es += ((p == i ? 1.0f : 0) - ke * kt[p]) *
                              (dt[p] * Sh[p * DV + j]);
                    Sh_new[i * DV + j] = es + kt[i] * (vt[j] * wt[j]);
                }
            }
            memcpy(Sh, Sh_new, sizeof(Sh_new));
            float *oh = o + t * H * DV + h * DV;
            for (int j = 0; j < DV; j++) {
                float s = 0;
                for (int p = 0; p < DK; p++) s += Sh_new[p * DV + j] * qt[p];
                oh[j] = s;
            }
        }
    }
}

/* one full GDN2Block + residual + RMSNorm(D) */
static void gdn2_layer(
    const float *in, int L, int D, int H, int DK, int DV,
    const float *q_w, const float *q_c, const float *k_w, const float *k_c,
    const float *v_w, const float *v_c, const float *out_w,
    const float *f0_w, const float *f1_w, const float *b_w, const float *w_w,
    const float *g0_w, const float *g1_w,
    const float *A_log, const float *dt_b, const float *on_w, const float *n_w,
    float *S, float *out)
{
    int KD = H * DK, VD = H * DV, N = L * KD, NV = L * VD;
    float *q = (float *)calloc(N, sizeof(float));
    float *k = (float *)calloc(N, sizeof(float));
    float *v = (float *)calloc(NV, sizeof(float));
    float *b = (float *)calloc(N, sizeof(float));
    float *w = (float *)calloc(NV, sizeof(float));
    float *decay = (float *)calloc(N, sizeof(float));
    float *scan_o = (float *)calloc(NV, sizeof(float));

    for (int t = 0; t < L; t++) {
        const float *x = in + t * D;
        g_linear(q_w, x, q + t * KD, KD, D, NULL);
        g_linear(k_w, x, k + t * KD, KD, D, NULL);
        g_linear(v_w, x, v + t * VD, VD, D, NULL);
        g_linear(b_w, x, b + t * KD, KD, D, NULL);
        g_linear(w_w, x, w + t * VD, VD, D, NULL);
        /* gate g = -exp(A_log[h]) * softplus(f1 @ f0 @ x + dt_bias) */
        float f0[32], rawg[KD];
        g_linear(f0_w, x, f0, 32, D, NULL);
        g_linear(f1_w, f0, rawg, KD, 32, NULL);
        for (int h = 0; h < H; h++) {
            float a = -expf(A_log[h]);
            for (int i = 0; i < DK; i++) {
                int idx = t * KD + h * DK + i;
                decay[idx] = expf(a * g_softplus(rawg[h * DK + i]
                                                 + dt_b[h * DK + i]));
            }
        }
    }
    for (int t = 0; t < L; t++) {
        for (int i = 0; i < KD; i++) b[t * KD + i] = g_sigmoid(b[t * KD + i]);
        for (int i = 0; i < VD; i++) w[t * VD + i] = g_sigmoid(w[t * VD + i]);
    }
#ifdef DEBUG_MCU
    printf("b0     = %.6f %.6f %.6f %.6f\n", b[0], b[1], b[2], b[3]);
    printf("w0     = %.6f %.6f %.6f %.6f\n", w[0], w[1], w[2], w[3]);
    printf("decay0 = %.6f %.6f %.6f %.6f\n", decay[0], decay[1], decay[2], decay[3]);
#endif
    /* causal convs on the full sequences (in-place temp buffers) */
    float *qt = (float *)calloc(N, sizeof(float));
    float *kt = (float *)calloc(N, sizeof(float));
    float *vt = (float *)calloc(NV, sizeof(float));
    g_causal_conv(q, q_c, qt, L, KD, 4);
    g_causal_conv(k, k_c, kt, L, KD, 4);
    g_causal_conv(v, v_c, vt, L, VD, 4);
    memcpy(q, qt, N * sizeof(float));
    memcpy(k, kt, N * sizeof(float));
    memcpy(v, vt, NV * sizeof(float));
    free(qt); free(kt); free(vt);
#ifdef DEBUG_MCU
    printf("qconv  = %.6f %.6f %.6f %.6f\n", q[0], q[1], q[2], q[3]);
    printf("kconv  = %.6f %.6f %.6f %.6f\n", k[0], k[1], k[2], k[3]);
    printf("vconv  = %.6f %.6f %.6f %.6f\n", v[0], v[1], v[2], v[3]);
#endif
    /* F.normalize(dim=-1) acts on the full key_dim (=H*DK), all heads */
    g_l2norm(q, N, KD);
    g_l2norm(k, N, KD);
#ifdef DEBUG_MCU
    printf("qnorm  = %.6f %.6f %.6f %.6f\n", q[0], q[1], q[2], q[3]);
#endif

    gdn2_scan(q, k, v, b, w, decay, S, scan_o, L, H, DK, DV);
#ifdef DEBUG_MCU
    printf("scan_o = %.6f %.6f %.6f %.6f\n", scan_o[0], scan_o[1], scan_o[2], scan_o[3]);
#endif

    /* per-head RMSNorm over DV, then * silu(g1 @ g0 @ x), then out_proj */
    for (int t = 0; t < L; t++) {
        const float *x = in + t * D;
        float g0[32], go[VD];
        g_linear(g0_w, x, g0, 32, D, NULL);
        g_linear(g1_w, g0, go, VD, 32, NULL);
        for (int h = 0; h < H; h++) {
            float *oh = scan_o + t * VD + h * DV;
            g_rmsnorm(oh, DV, on_w);
            for (int i = 0; i < DV; i++) oh[i] *= g_silu(go[h * DV + i]);
        }
        g_linear(out_w, scan_o + t * VD, out + t * D, D, VD, NULL);
#ifdef DEBUG_MCU
        if (t == 0)
            printf("gdnpre = %.6f %.6f %.6f %.6f\n", out[0], out[1], out[2], out[3]);
#endif
        for (int i = 0; i < D; i++) out[t * D + i] += x[i];
        g_rmsnorm(out + t * D, D, n_w);
    }

    free(q); free(k); free(v); free(b); free(w); free(decay); free(scan_o);
}

GDN2Error GDN2_Infer(const GDN2_Variables *m, const float *input, float *out) {
    int D = m->d_model, W = m->window_size, ps = m->patch_size;
    int H = m->num_heads, DK = m->head_dim, DV = m->head_vdim;
    int LP = W / ps; /* patch count (W divisible by ps here) */

    /* patch embed: (W,1) -> (LP,D) via Linear(ps -> D) */
    float *h0 = (float *)calloc(LP * D, sizeof(float));
    float *h1 = (float *)calloc(LP * D, sizeof(float));
    for (int t = 0; t < LP; t++) {
        float px[4];
        for (int j = 0; j < ps; j++) px[j] = input[t * ps + j];
        g_linear(m->inp_w, px, h0 + t * D, D, ps, m->inp_b);
    }

    float S0[H * DK * DV], S1[H * DK * DV];
    memset(S0, 0, sizeof(S0));
    memset(S1, 0, sizeof(S1));
#ifdef DEBUG_MCU
    printf("emb    = %.6f %.6f %.6f %.6f\n", h0[0], h0[1], h0[2], h0[3]);
#endif
    gdn2_layer(h0, LP, D, H, DK, DV, m->l0_q_w, m->l0_q_c, m->l0_k_w,
               m->l0_k_c, m->l0_v_w, m->l0_v_c, m->l0_out_w, m->l0_f0_w,
               m->l0_f1_w, m->l0_b_w, m->l0_w_w, m->l0_g0_w, m->l0_g1_w,
               m->l0_A_log, m->l0_dt, m->l0_onorm, m->l0_norm, S0, h1);
#ifdef DEBUG_MCU
    printf("l0norm = %.6f %.6f %.6f %.6f\n", h1[0], h1[1], h1[2], h1[3]);
#endif
    gdn2_layer(h1, LP, D, H, DK, DV, m->l1_q_w, m->l1_q_c, m->l1_k_w,
               m->l1_k_c, m->l1_v_w, m->l1_v_c, m->l1_out_w, m->l1_f0_w,
               m->l1_f1_w, m->l1_b_w, m->l1_w_w, m->l1_g0_w, m->l1_g1_w,
               m->l1_A_log, m->l1_dt, m->l1_onorm, m->l1_norm, S1, h0);
#ifdef DEBUG_MCU
    printf("l1norm = %.6f %.6f %.6f %.6f\n", h0[0], h0[1], h0[2], h0[3]);
#endif

    /* restore_len: last token = last patch; GELU head */
    float *hl = h0 + (LP - 1) * D;
    float h128[128];
    g_linear(m->h1_w, hl, h128, 128, D, m->h1_b);
    for (int i = 0; i < 128; i++) h128[i] = g_gelu(h128[i]);
    g_linear(m->h2_w, h128, out, 1, 128, m->h2_b);

    free(h0);
    free(h1);
    return GDN2_OK;
}
