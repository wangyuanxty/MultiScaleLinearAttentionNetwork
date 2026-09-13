/** gdn2_multiscale.c — full DeltaCycle inference in C.
 *
 * Replicates GDNBatteryModel.forward(stage_query=True) exactly: three patch
 * branches (2/4/8) of two GDN-2 layers each, a per-layer cross-scale exchange
 * in which the coarse branch's flattened GDN state becomes a single query
 * reading the fine and mid branches, then a fused last-token readout.
 *
 * Layer order matters: every branch runs its GDN-2 layer on its OWN token
 * count (32/16/8), and only then are the branches tiled up to L_patch for the
 * exchange. From layer 1 on, all branches are L_patch long. Aligning before
 * the first layer instead would scan the wrong sequence length.
 *
 * The per-layer kernel is the one already verified bit-exact against PyTorch
 * in gdn2_mcu.c -- this file only adds the assembly around it, so the
 * single-branch chain stays untouched.
 *
 * Shapes follow the exported checkpoint (d_model=64, H=4, Dk=16, Dv=32,
 * window 64, patch 2/4/8 -> 32/16/8 tokens, L_patch=32).
 */
#pragma once
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "gdn2_mcu.h"
#include "gdn2_mcu.c"

#define MS_NBR 3
#define MS_LAYERS 2
#define MS_H 4
#define MS_DK 16
#define MS_DV 32
#define MS_STATE (MS_H * MS_DK * MS_DV) /* 2048, the flattened coarse state */
#define MS_QD 32                        /* cross-scale query width */
#define MS_L 64                         /* window length */
#define MS_LP 32                        /* token count of the finest branch */
#define MS_D 64                         /* d_model */

typedef struct {
    gdn_mat_t q_w, q_c, k_w, k_c, v_w, v_c, out_w;
    gdn_mat_t f0_w, f1_w, b_w, w_w, g0_w, g1_w;
    const float *A_log, *dt, *onorm, *norm;
} GDN2MS_Layer;

typedef struct {
    gdn_mat_t proj_w;
    const float *proj_b;
    uint16_t patch_size;
    uint16_t tokens; /* window_size / patch_size */
    GDN2MS_Layer layers[MS_LAYERS];
} GDN2MS_Branch;

typedef struct { gdn_mat_t wq, wk, wv; } GDN2MS_Cross;

typedef struct {
    uint16_t d_model, num_heads, head_dim, head_vdim;
    uint16_t window_size, conv_size;
    GDN2MS_Branch br[MS_NBR];
    GDN2MS_Cross cross[MS_LAYERS];
    gdn_mat_t h1_w, h2_w;
    const float *h1_b, *h2_b;
} GDN2MS_Variables;

/* one branch layer -> the verified single-branch kernel */
static void ms_layer(const GDN2MS_Layer *l, const float *in, int toks, int H,
                     int DK, int DV, float *S, float *out) {
    gdn2_layer(in, toks, MS_D, H, DK, DV, &l->q_w, &l->q_c, &l->k_w, &l->k_c,
               &l->v_w, &l->v_c, &l->out_w, &l->f0_w, &l->f1_w, &l->b_w,
               &l->w_w, &l->g0_w, &l->g1_w, l->A_log, l->dt, l->onorm,
               l->norm, S, out);
}

/* align a short branch to L_patch: h.repeat(1, reps, 1)[:, :LP, :] --
 * whole-sequence tiling. (restore_len below is the other operation:
 * per-token repetition. The two are not interchangeable.) */
static void ms_align(const float *h, float *out, int toks, int LP) {
    for (int t = 0; t < LP; t++)
        memcpy(out + t * MS_D, h + (t % toks) * MS_D, MS_D * sizeof(float));
}

/* restore_len: h.unsqueeze(2).expand(ps).reshape(-1)[:, :L, :] */
static void ms_restore_len(const float *h, float *out, int L, int ps) {
    for (int t = 0; t < L; t++)
        memcpy(out + t * MS_D, h + (t / ps) * MS_D, MS_D * sizeof(float));
}

/* Single-query stage attention: q from the coarse GDN state, K/V from the
 * target branch, additive residual, no gate. h_in may alias h_out. */
static void ms_cross(int layer, const GDN2MS_Variables *m, const float *S_flat,
                     const float *h_in, float *h_out, int T) {
    const GDN2MS_Cross *c = &m->cross[layer];
    static float q[MS_QD], k[MS_LP][MS_QD], v[MS_LP][MS_D];
    float sc[MS_LP], mx = -1e30f, sum = 0.0f;

    g_linear(&c->wq, S_flat, q, MS_QD, MS_STATE, NULL);
    for (int t = 0; t < T; t++) {
        g_linear(&c->wk, h_in + t * MS_D, k[t], MS_QD, MS_D, NULL);
        g_linear(&c->wv, h_in + t * MS_D, v[t], MS_D, MS_D, NULL);
    }
    /* softmax(k . q / sqrt(QD)) over positions, then the weighted values */
    for (int t = 0; t < T; t++) {
        float s = 0.0f;
        for (int j = 0; j < MS_QD; j++) s += k[t][j] * q[j];
        sc[t] = s / sqrtf((float)MS_QD);
        if (sc[t] > mx) mx = sc[t];
    }
    for (int t = 0; t < T; t++) { sc[t] = expf(sc[t] - mx); sum += sc[t]; }
    for (int t = 0; t < T; t++)
        for (int j = 0; j < MS_D; j++)
            h_out[t * MS_D + j] = h_in[t * MS_D + j] + (sc[t] / sum) * v[t][j];
}

GDN2Error GDN2MS_Infer(const GDN2MS_Variables *m, const float *input,
                       float *out) {
    const int D = m->d_model, H = m->num_heads, DK = m->head_dim;
    const int DV = m->head_vdim, L = m->window_size, LP = MS_LP;

    float *h[MS_NBR], *tmp, *rf[MS_NBR];
    int ntok[MS_NBR];
    for (int i = 0; i < MS_NBR; i++) {
        h[i] = (float *)calloc(LP * D, sizeof(float));
        rf[i] = (float *)calloc(L * D, sizeof(float));
        ntok[i] = m->br[i].tokens;
    }
    tmp = (float *)calloc(LP * D, sizeof(float));
    float *S = (float *)calloc(MS_NBR * MS_LAYERS * MS_STATE, sizeof(float));
    if (!tmp || !S || !h[0] || !rf[0]) return GDN2_ERR_MEMORY;

    /* patch embed, each branch at its own token count */
    for (int i = 0; i < MS_NBR; i++) {
        const GDN2MS_Branch *b = &m->br[i];
        float px[8];
        for (int t = 0; t < b->tokens; t++) {
            for (int j = 0; j < b->patch_size; j++)
                px[j] = input[t * b->patch_size + j];
            g_linear(&b->proj_w, px, h[i] + t * D, D, b->patch_size, b->proj_b);
        }
    }

    for (int layer = 0; layer < MS_LAYERS; layer++) {
        float *S_c = NULL;
        for (int i = 0; i < MS_NBR; i++) {
            float *S_i = S + (i * MS_LAYERS + layer) * MS_STATE;
            ms_layer(&m->br[i].layers[layer], h[i], ntok[i], H, DK, DV, S_i,
                     tmp);
            memcpy(h[i], tmp, (size_t)ntok[i] * D * sizeof(float));
            if (i == MS_NBR - 1) S_c = S_i; /* coarse state feeds the query */
        }
        /* align AFTER the layer, then exchange; branches stay aligned from
         * here on, which is what the next layer consumes */
        for (int i = 0; i < MS_NBR; i++) {
            if (ntok[i] < LP) {
                ms_align(h[i], tmp, ntok[i], LP);
                memcpy(h[i], tmp, LP * D * sizeof(float));
                ntok[i] = LP;
            }
        }
        ms_cross(layer, m, S_c, h[0], h[0], LP); /* fine <- coarse */
        ms_cross(layer, m, S_c, h[1], h[1], LP); /* mid  <- coarse */
    }

    for (int i = 0; i < MS_NBR; i++)
        ms_restore_len(h[i], rf[i], L, m->br[i].patch_size);

    /* fused readout: [fine_last, mid_last, pooled_coarse] -> 192 */
    float fused[3 * MS_D], c_pool[MS_D];
    memset(c_pool, 0, sizeof(c_pool));
    for (int t = 0; t < L; t++)
        for (int j = 0; j < D; j++) c_pool[j] += rf[2][t * D + j];
    for (int j = 0; j < D; j++) {
        c_pool[j] /= (float)L;
        fused[j] = rf[0][(L - 1) * D + j];
        fused[D + j] = rf[1][(L - 1) * D + j];
        fused[2 * D + j] = c_pool[j];
    }
    float h128[128];
    g_linear(&m->h1_w, fused, h128, 128, 3 * MS_D, m->h1_b);
    for (int i = 0; i < 128; i++) h128[i] = g_gelu(h128[i]);
    g_linear(&m->h2_w, h128, out, 1, 128, m->h2_b);

    for (int i = 0; i < MS_NBR; i++) { free(h[i]); free(rf[i]); }
    free(tmp);
    free(S);
    return GDN2_OK;
}
