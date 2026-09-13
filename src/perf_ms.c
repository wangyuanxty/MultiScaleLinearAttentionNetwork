/* perf_ms.c — cycle count of the full multi-scale inference, and how much it
 * varies with the input.
 *
 * The model's own control flow is data-independent (fixed unrolled loops, no
 * allocation, no recursion), but the libm routines it calls -- expf, logf,
 * sqrtf, erff -- branch on their arguments. So the honest statement is a
 * measured spread, not "constant WCET".
 *
 * Cycles come from SysTick's current-value register, which QEMU models. It
 * counts DOWN, so a reading before minus a reading after gives the delta.
 *
 * Build: same flags as mcu_main_ms.c.
 */
#include <stdint.h>
#include <string.h>
#include "gdn2_multiscale.c"
#if defined(GDN_Q4)
#include "gdn_ms_weights_q4.h"
#elif defined(GDN_Q8)
#include "gdn_ms_weights_q8.h"
#else
#include "gdn_ms_weights.h"
#endif
#include "ms_init.h"
#include "test_input_ms.h"

extern uint32_t _estack;
extern void _start(void);
extern unsigned char __bss_start__, __bss_end__;

__attribute__((section(".vectors"), used))
const uint32_t vectors[2] = { (uint32_t)&_estack, (uint32_t)&_start };

__attribute__((noreturn, section(".text.start")))
void _start(void) {
    for (unsigned char *p = &__bss_start__; p < &__bss_end__; p++) *p = 0;
    main();
    for (;;) {}
}

#define HEAP_SIZE (2u * 1024u * 1024u)
static unsigned char heap_pool[HEAP_SIZE];
static size_t heap_off = 0;
static int heap_overflow = 0;

static void sh_write0(const char *s) {
    register uint32_t r0 __asm("r0") = 0x04;
    register const char *r1 __asm("r1") = s;
    __asm volatile("bkpt 0xab" : : "r"(r0), "r"(r1));
}
void *malloc(size_t n) {
    n = (n + 7u) & ~7u;
    if (heap_off + n > HEAP_SIZE) { heap_overflow = 1; return 0; }
    void *p = heap_pool + heap_off;
    heap_off += n;
    return p;
}
void *calloc(size_t nmemb, size_t sz) {
    size_t n = nmemb * sz;
    void *p = malloc(n);
    if (p) memset(p, 0, n);
    return p;
}
void free(void *p) { (void)p; }
void *memcpy(void *dst, const void *src, size_t n) {
    unsigned char *d = dst; const unsigned char *s = src;
    while (n--) *d++ = *s++;
    return dst;
}
void *memset(void *dst, int c, size_t n) {
    unsigned char *d = dst;
    while (n--) *d++ = (unsigned char)c;
    return dst;
}
int __errno;

/* SysTick lives in the core's private peripheral space and counts down. */
static uint32_t rd_syst(void) {
    return *(volatile uint32_t *)0xE000E018 & 0x00FFFFFF;
}
static void put_uint(uint32_t x, char *b) {
    char tmp[11]; int i = 0;
    do { tmp[i++] = (char)('0' + x % 10); x /= 10; } while (x);
    while (i) *b++ = tmp[--i];
    *b = 0;
}

static const float base_in[MS_L] = { TEST_INPUT };

/* A fixed amount of work, used to test what the timer is actually counting:
 * if SysTick tracks executed instructions this is stable run to run, and if
 * it tracks the host's wall clock it is not. The store to a volatile keeps
 * the loop from being optimised away. */
static volatile uint32_t sink;
static void busy(uint32_t n) {
    uint32_t a = 1u;
    for (uint32_t i = 0; i < n; i++) a = a * 1103515245u + 12345u;
    sink = a;
}

int main(void) {
    volatile uint32_t *syst_csr = (volatile uint32_t *)0xE000E010;
    volatile uint32_t *syst_rvr = (volatile uint32_t *)0xE000E014;
    *syst_rvr = 0x00FFFFFF;
    *syst_csr = 5; /* core clock source, enable, auto-reload */

    GDN2MS_Variables m;
    float in[MS_L];
    float out = 0.0f;
    ms_init(&m);

    /* three inputs: the reference window, then slightly rescaled copies --
     * different arguments reach different libm branches */
    static const float gain[3] = { 1.0f, 0.99f, 1.01f };
    char msg[96], nb[12];

    for (int k = 0; k < 2; k++) {
        uint32_t b0 = rd_syst();
        busy(500000u);
        uint32_t b1 = rd_syst();
        int p = 0;
        const char *s = "CAL t0=";
        while (*s) msg[p++] = *s++;
        put_uint(b0, nb); for (int i = 0; nb[i]; i++) msg[p++] = nb[i];
        s = " t1="; while (*s) msg[p++] = *s++;
        put_uint(b1, nb); for (int i = 0; nb[i]; i++) msg[p++] = nb[i];
        msg[p++] = '\n'; msg[p] = 0;
        sh_write0(msg);
    }

    for (int run = 0; run < 3; run++) {
        /* One inference per power cycle: the bump allocator never frees, and
         * an inference costs ~820 KB of scratch, so without this reset the
         * third run would walk past the pool. */
        heap_off = 0;
        for (int i = 0; i < MS_L; i++) in[i] = gain[run] * base_in[i];
        uint32_t t0 = rd_syst();
        GDN2MS_Infer(&m, in, &out);
        uint32_t t1 = rd_syst();
        uint32_t cyc = t0 - t1; /* counts down, so t0 > t1 */

        int p = 0;
        const char *s;
        s = "CYC t0="; while (*s) msg[p++] = *s++;
        put_uint(t0, nb); for (int i = 0; nb[i]; i++) msg[p++] = nb[i];
        s = " t1="; while (*s) msg[p++] = *s++;
        put_uint(t1, nb); for (int i = 0; nb[i]; i++) msg[p++] = nb[i];
        s = " d="; while (*s) msg[p++] = *s++;
        put_uint(cyc, nb); for (int i = 0; nb[i]; i++) msg[p++] = nb[i];
        s = " gain="; while (*s) msg[p++] = *s++;
        put_uint((uint32_t)(gain[run] * 1000.0f), nb);
        for (int i = 0; nb[i]; i++) msg[p++] = nb[i];
        s = " out="; while (*s) msg[p++] = *s++;
        uint32_t u; memcpy(&u, &out, 4);
        for (int i = 0; i < 8; i++)
            msg[p++] = "0123456789ABCDEF"[(u >> (28 - 4 * i)) & 0xF];
        msg[p++] = '\n'; msg[p] = 0;
        sh_write0(msg);
    }
    if (heap_overflow) sh_write0("HEAP_OVERFLOW\n");
    return 0;
}
