/* mcu_min.c — absolute minimal: vectors + _start + SYS_WRITE0 constant */
#include <stdint.h>

extern uint32_t _estack;
extern void _start(void);

__attribute__((section(".vectors"), used))
const uint32_t vectors[2] = { (uint32_t)&_estack, (uint32_t)&_start };

static void sh_write0(const char *s) {
    register uint32_t r0 __asm("r0") = 0x04;
    register const char *r1 __asm("r1") = s;
    __asm volatile("bkpt 0xab" : : "r"(r0), "r"(r1));
}

__attribute__((noreturn, section(".text.start")))
void _start(void) {
    extern unsigned char __bss_start__, __bss_end__;
    for (unsigned char *p = &__bss_start__; p < &__bss_end__; p++) *p = 0;
    main();
    for (;;) {}
}

static char big_bss[1024]; /* force a non-trivial .bss */

int main(void) {
    sh_write0("MAIN_OK\n");
    return 0;
}
