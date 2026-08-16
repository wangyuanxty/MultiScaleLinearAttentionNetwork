/** gdn2_mcu.h — GDN-2 MCU inference header (v3: matches gdn_v2.py exactly) */
#pragma once
#include <stdint.h>

typedef enum { GDN2_OK, GDN2_ERR_MEMORY } GDN2Error;

typedef struct {
    uint16_t d_model, num_heads, head_dim, head_vdim, window_size;
    uint16_t patch_size, conv_size;
    /* per-layer weights, 2 layers (single branch) */
    const float *l0_q_w, *l0_q_c, *l0_k_w, *l0_k_c, *l0_v_w, *l0_v_c, *l0_out_w;
    const float *l0_f0_w, *l0_f1_w, *l0_b_w, *l0_w_w, *l0_g0_w, *l0_g1_w;
    const float *l0_A_log, *l0_dt, *l0_onorm, *l0_norm;
    const float *l1_q_w, *l1_q_c, *l1_k_w, *l1_k_c, *l1_v_w, *l1_v_c, *l1_out_w;
    const float *l1_f0_w, *l1_f1_w, *l1_b_w, *l1_w_w, *l1_g0_w, *l1_g1_w;
    const float *l1_A_log, *l1_dt, *l1_onorm, *l1_norm;
    /* patch embed + head (GELU) */
    const float *inp_w, *inp_b, *h1_w, *h1_b, *h2_w, *h2_b;
} GDN2_Variables;

GDN2Error GDN2_Infer(const GDN2_Variables *m, const float *input, float *out);
