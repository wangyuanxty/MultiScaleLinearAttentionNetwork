/* mcu_main.c — full GDN-2 inference on QEMU Cortex-M3 (lm3s6965evb).
 * Bare-metal, no newlib: semihosting SYS_WRITE0 (BKPT 0xAB) output,
 * self-contained vector table + _start, static-pool malloc for the
 * fixed-size inference buffers (no runtime growth). */
#include <stdint.h>
#include <string.h>
#include "gdn2_mcu.h"
#include "gdn_weights.h"
#include "gdn2_mcu.c"
#include "test_input.h"

extern uint32_t _estack;
extern void _start(void);

__attribute__((section(".vectors"), used))
const uint32_t vectors[2] = { (uint32_t)&_estack, (uint32_t)&_start };
extern unsigned char __bss_start__, __bss_end__;

__attribute__((noreturn, section(".text.start")))
void _start(void) {
    /* zero .bss (heap_off etc. must start at 0) */
    for (unsigned char *p = &__bss_start__; p < &__bss_end__; p++) *p = 0;
    main();
    for (;;) {}
}

/* ---- static-pool malloc (fixed-size inference buffers only) ---- */
static unsigned char heap_pool[48 * 1024];
static size_t heap_off = 0;
void *malloc(size_t n) {
    n = (n + 7u) & ~7u;
    void *p = heap_pool + heap_off;
    heap_off += n;
    return p;
}
void *calloc(size_t nmemb, size_t sz) {
    void *p = malloc(nmemb * sz);
    memset(p, 0, nmemb * sz);
    return p;
}
void free(void *p) { (void)p; }

/* ---- libc stubs for -nostdlib + libm linking ---- */
void *memcpy(void *dst, const void *src, size_t n) {
    unsigned char *d = (unsigned char *)dst;
    const unsigned char *s = (const unsigned char *)src;
    while (n--) *d++ = *s++;
    return dst;
}
void *memset(void *dst, int c, size_t n) {
    unsigned char *d = (unsigned char *)dst;
    while (n--) *d++ = (unsigned char)c;
    return dst;
}
int __errno;

static void sh_write0(const char *s) {
    register uint32_t r0 __asm("r0") = 0x04; /* SYS_WRITE0 */
    register const char *r1 __asm("r1") = s;
    __asm volatile("bkpt 0xab" : : "r"(r0), "r"(r1));
}

static void tohex(float x, char *buf) { /* buf >= 9 */
    uint32_t u;
    memcpy(&u, &x, 4);
    for (int i = 0; i < 8; i++)
        buf[i] = "0123456789ABCDEF"[(u >> (28 - 4 * i)) & 0xF];
    buf[8] = 0;
}

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
    float input[64] = { TEST_INPUT };
    float output = 0.0f;
    GDN2_Infer(&m, input, &output);
    char hb[9], out[24];
    tohex(output, hb);
    char *p = out;
    const char *tag = "QEMU_HEX:";
    while (*tag) *p++ = *tag++;
    for (int i = 0; i < 8; i++) *p++ = hb[i];
    *p++ = '\n';
    *p = 0;
    sh_write0(out);
    return 0;
}
