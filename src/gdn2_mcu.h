/** gdn2_mcu.h — GDN-2 MCU inference header (v3: matches gdn_v2.py exactly) */
#pragma once
#include <stdint.h>

typedef enum { GDN2_OK, GDN2_ERR_MEMORY } GDN2Error;

/* Weight storage for the 2-D matrices (Linear and depthwise-conv weights).
 *
 * With GDN_Q8 they sit in flash as signed bytes plus one fp32 scale per
 * output channel (symmetric; see write_array in export_gdn_weights.py)
 * and GDN_W applies the scale on read. Without it they are plain fp32
 * arrays. The fp32 variant holds a single pointer, so its layout -- and
 * therefore the verified build -- is byte-for-byte unchanged.
 *
 * 1-D tensors (biases, RMSNorm weights, A_log, dt) stay fp32 in both
 * modes: they amount to a few KB, and PyTorch keeps biases fp32 too. */
#ifdef GDN_Q8
typedef struct { const signed char *w; const float *s; } gdn_mat_t;
#define GDN_MAT(name) ((gdn_mat_t){ (name##_q), (name##_s) })
#define GDN_W(p, i, n, j) ((float)((p)->w[(i) * (n) + (j)]) * (p)->s[i])
#else
typedef struct { const float *w; } gdn_mat_t;
#define GDN_MAT(name) ((gdn_mat_t){ (name) })
#define GDN_W(p, i, n, j) ((p)->w[(i) * (n) + (j)])
#endif

typedef struct {
    uint16_t d_model, num_heads, head_dim, head_vdim, window_size;
    uint16_t patch_size, conv_size;
    /* per-layer weights, 2 layers (single branch) */
    gdn_mat_t l0_q_w, l0_q_c, l0_k_w, l0_k_c, l0_v_w, l0_v_c, l0_out_w;
    gdn_mat_t l0_f0_w, l0_f1_w, l0_b_w, l0_w_w, l0_g0_w, l0_g1_w;
    const float *l0_A_log, *l0_dt, *l0_onorm, *l0_norm;
    gdn_mat_t l1_q_w, l1_q_c, l1_k_w, l1_k_c, l1_v_w, l1_v_c, l1_out_w;
    gdn_mat_t l1_f0_w, l1_f1_w, l1_b_w, l1_w_w, l1_g0_w, l1_g1_w;
    const float *l1_A_log, *l1_dt, *l1_onorm, *l1_norm;
    /* patch embed + head (GELU) */
    gdn_mat_t inp_w, h1_w, h2_w;
    const float *inp_b, *h1_b, *h2_b;
} GDN2_Variables;

GDN2Error GDN2_Infer(const GDN2_Variables *m, const float *input, float *out);
