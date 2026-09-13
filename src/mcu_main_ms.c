/* mcu_main_ms.c — full DeltaCycle inference on QEMU mps2-an385.
 *
 * Bare metal, no libc: semihosting SYS_WRITE0 output, self-contained vector
 * table + _start, and a static-pool allocator for the inference buffers.
 *
 * The pool is sized for the multi-scale model and, unlike the single-branch
 * harness, it REFUSES to over-allocate: gdn2_layer allocates ~115 KB of
 * scratch per call and never frees it (free() is a no-op in a bump pool), so
 * six layer calls need ~690 KB. The old 48 KB pool plus a silent bump
 * allocator would run off the end of the array and corrupt whatever RAM
 * followed -- which on a roomy machine can still produce the right answer by
 * accident.
 *
 * Build:
 *   arm-none-eabi-gcc -mcpu=cortex-m3 -mthumb -ffreestanding -O2 \
 *     -T mps2_linker.ld -o mcu_ms.elf mcu_main_ms.c -nostdlib -nostartfiles \
 *     -lm -lc -lgcc
 * Run:
 *   qemu-system-arm -M mps2-an385 -nographic -semihosting -kernel mcu_ms.elf
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

/* ---- static-pool allocator, with an over-allocation guard ---- */
#define HEAP_SIZE (2u * 1024u * 1024u)
static unsigned char heap_pool[HEAP_SIZE];
static size_t heap_off = 0;
static int heap_overflow = 0;

static void sh_write0(const char *s) {
    register uint32_t r0 __asm("r0") = 0x04; /* SYS_WRITE0 */
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

int main(void) {
    GDN2MS_Variables m;
    float input[MS_L] = { TEST_INPUT };
    float output = 0.0f;
    char msg[32];

    ms_init(&m);
    GDN2Error e = GDN2MS_Infer(&m, input, &output);
    if (e != GDN2_OK || heap_overflow) {
        sh_write0(heap_overflow ? "HEAP_OVERFLOW\n" : "INFER_FAILED\n");
        for (;;) {}
    }

    /* print the raw float32 bits, same convention as the single-branch run */
    uint32_t u;
    memcpy(&u, &output, 4);
    char *p = msg;
    const char *tag = "QEMU_HEX:";
    while (*tag) *p++ = *tag++;
    for (int i = 0; i < 8; i++)
        *p++ = "0123456789ABCDEF"[(u >> (28 - 4 * i)) & 0xF];
    *p++ = '\n';
    *p = 0;
    sh_write0(msg);
    return 0;
}
