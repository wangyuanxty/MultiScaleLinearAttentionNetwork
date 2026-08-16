/** verify_deploy.c — PyTorch vs C output comparison (v3).
 * Compile: cd src && gcc -O2 -o verify_deploy verify_deploy.c -lm
 * Run: verify_deploy   (reads test_input.csv, prints C output)
 * (gdn2_mcu.c is included directly, following Mamba pattern) */
#include <stdio.h>
#include <stdlib.h>
#include "gdn2_mcu.h"
#include "gdn_weights.h"
#include "gdn2_mcu.c"

static GDN2_Variables gdn2_init(void) {
    GDN2_Variables m = {0};
    m.d_model = 64; m.num_heads = 4; m.head_dim = 16; m.head_vdim = 32;
    m.window_size = 64; m.patch_size = 2; m.conv_size = 4;
    m.l0_q_w = branches_0_layers_0_gdn_q_proj_weight;
    m.l0_q_c = branches_0_layers_0_gdn_q_conv_conv_weight;
    m.l0_k_w = branches_0_layers_0_gdn_k_proj_weight;
    m.l0_k_c = branches_0_layers_0_gdn_k_conv_conv_weight;
    m.l0_v_w = branches_0_layers_0_gdn_v_proj_weight;
    m.l0_v_c = branches_0_layers_0_gdn_v_conv_conv_weight;
    m.l0_out_w = branches_0_layers_0_gdn_out_proj_weight;
    m.l0_f0_w = branches_0_layers_0_gdn_f_proj_0_weight;
    m.l0_f1_w = branches_0_layers_0_gdn_f_proj_1_weight;
    m.l0_b_w = branches_0_layers_0_gdn_b_proj_weight;
    m.l0_w_w = branches_0_layers_0_gdn_w_proj_weight;
    m.l0_g0_w = branches_0_layers_0_gdn_g_proj_0_weight;
    m.l0_g1_w = branches_0_layers_0_gdn_g_proj_1_weight;
    m.l0_A_log = branches_0_layers_0_gdn_A_log;
    m.l0_dt = branches_0_layers_0_gdn_dt_bias;
    m.l0_onorm = branches_0_layers_0_gdn_out_norm_weight;
    m.l0_norm = branches_0_layers_0_norm_weight;
    m.l1_q_w = branches_0_layers_1_gdn_q_proj_weight;
    m.l1_q_c = branches_0_layers_1_gdn_q_conv_conv_weight;
    m.l1_k_w = branches_0_layers_1_gdn_k_proj_weight;
    m.l1_k_c = branches_0_layers_1_gdn_k_conv_conv_weight;
    m.l1_v_w = branches_0_layers_1_gdn_v_proj_weight;
    m.l1_v_c = branches_0_layers_1_gdn_v_conv_conv_weight;
    m.l1_out_w = branches_0_layers_1_gdn_out_proj_weight;
    m.l1_f0_w = branches_0_layers_1_gdn_f_proj_0_weight;
    m.l1_f1_w = branches_0_layers_1_gdn_f_proj_1_weight;
    m.l1_b_w = branches_0_layers_1_gdn_b_proj_weight;
    m.l1_w_w = branches_0_layers_1_gdn_w_proj_weight;
    m.l1_g0_w = branches_0_layers_1_gdn_g_proj_0_weight;
    m.l1_g1_w = branches_0_layers_1_gdn_g_proj_1_weight;
    m.l1_A_log = branches_0_layers_1_gdn_A_log;
    m.l1_dt = branches_0_layers_1_gdn_dt_bias;
    m.l1_onorm = branches_0_layers_1_gdn_out_norm_weight;
    m.l1_norm = branches_0_layers_1_norm_weight;
    m.inp_w = branches_0_proj_weight;
    m.inp_b = branches_0_proj_bias;
    m.h1_w = head_cap_0_weight;
    m.h1_b = head_cap_0_bias;
    m.h2_w = head_cap_3_weight;
    m.h2_b = head_cap_3_bias;
    return m;
}

int main(void) {
    GDN2_Variables m = gdn2_init();
    float input[64] = {0};
    FILE *fp = fopen("test_input.csv", "r");
    if (!fp) { fprintf(stderr, "missing test_input.csv\n"); return 1; }
    for (int i = 0; i < 64; i++) {
        if (fscanf(fp, "%f", &input[i]) != 1) { fprintf(stderr, "read err\n"); return 1; }
    }
    fclose(fp);
    float output = 0;
    GDN2_Infer(&m, input, &output);
    printf("C output: %.10f\n", output);
    printf("Compare with PyTorch: see test_py_out.txt\n");
    return 0;
}
