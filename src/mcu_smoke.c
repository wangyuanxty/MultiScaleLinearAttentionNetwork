/* mcu_smoke.c — minimal libm test on QEMU (bisect the fault source) */
#include <stdint.h>
#include <math.h>
#include <string.h>

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

int __errno;

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

static void sh_write0(const char *s) {
    register uint32_t r0 __asm("r0") = 0x04;
    register const char *r1 __asm("r1") = s;
    __asm volatile("bkpt 0xab" : : "r"(r0), "r"(r1));
}
static void tohex(float x, char *buf) {
    uint32_t u; memcpy(&u, &x, 4);
    for (int i = 0; i < 8; i++)
        buf[i] = "0123456789ABCDEF"[(u >> (28 - 4 * i)) & 0xF];
    buf[8] = 0;
}

int main(void) {
    float a = expf(1.0f);
    float b = logf(2.0f);
    float c = sqrtf(2.0f);
    float d = erff(0.5f);
    float s = a + b + c + d;
    char hb[9], out[24];
    tohex(s, hb);
    char *p = out;
    const char *tag = "SMOKE_HEX:";
    while (*tag) *p++ = *tag++;
    for (int i = 0; i < 8; i++) *p++ = hb[i];
    *p++ = '\n'; *p = 0;
    sh_write0(out);
    return 0;
}
