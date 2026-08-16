/* mcu_min2.c — vectors + _start + bl main (no bss loop) */
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

int main(void) {
    sh_write0("MAIN2_OK\n");
    return 0;
}

__attribute__((noreturn, section(".text.start")))
void _start(void) {
    main();
    for (;;) {}
}
