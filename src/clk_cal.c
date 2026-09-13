/* clk_cal.c — what is QEMU's SysTick actually counting?
 *
 * The problem this solves. Under `-icount shift=0` QEMU advances its virtual
 * clock by exactly 1 ns per executed instruction. Peripheral clocks derived
 * from that virtual clock therefore tick at (instructions x 1ns x f_clk), and
 * a SysTick readout is a *time* measured against whatever reference clock the
 * machine model wires up. To turn that into anything physical we need two
 * numbers we did not have: the reference clock frequency, and how many
 * instructions the measured region actually executes.
 *
 * Both are obtained here. The calibration loop is written in inline asm so its
 * instruction count is a constant we chose rather than whatever the compiler
 * felt like emitting: MLA (1) + SUBS (1) + BNE (1) = exactly 3 per iteration.
 * Running it at two different trip counts gives a slope in ticks/instruction,
 * and from
 *
 *     ticks = instructions x 2^shift x f_ref / 1000      (shift = 0)
 *
 * the reference frequency follows:  f_ref[MHz] = 1000 x ticks/instruction.
 *
 * Two trip counts rather than one because a single point cannot distinguish
 * the loop's cost from a constant offset in the timer setup.
 *
 * Build and run:
 *   arm-none-eabi-gcc -mcpu=cortex-m3 -mthumb -nostdlib -nostartfiles \
 *     -ffreestanding -O2 -T mps2_linker.ld -o clk_cal.elf clk_cal.c
 *   qemu-system-arm -M mps2-an385 -nographic -semihosting \
 *     -icount shift=0,sleep=off -kernel clk_cal.elf
 */
#include <stdint.h>

int main(void);

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

static void sh_write0(const char *s) {
    register uint32_t r0 __asm("r0") = 0x04;
    register const char *r1 __asm("r1") = s;
    __asm volatile("bkpt 0xab" : : "r"(r0), "r"(r1));
}

/* Without this the firmware's trailing loop keeps QEMU alive forever, so a
 * run can never be timed end to end -- only killed. Halting the emulator from
 * inside the guest is what makes wall-clock measurement possible. */
static void __attribute__((noreturn)) sh_exit(void) {
    register uint32_t r0 __asm("r0") = 0x18;      /* SYS_EXIT */
    register uint32_t r1 __asm("r1") = 0x20026;   /* ADP_Stopped_ApplicationExit */
    __asm volatile("bkpt 0xab" : : "r"(r0), "r"(r1));
    for (;;) {}
}

#define SYST_CSR (*(volatile uint32_t *)0xE000E010)
#define SYST_RVR (*(volatile uint32_t *)0xE000E014)
#define SYST_CVR (*(volatile uint32_t *)0xE000E018)
#define SYST_MASK 0x00FFFFFFu

static uint32_t rd_syst(void) { return SYST_CVR & SYST_MASK; }

static void put_uint(uint32_t x, char *b) {
    char tmp[11]; int i = 0;
    do { tmp[i++] = (char)('0' + x % 10); x /= 10; } while (x);
    while (i) *b++ = tmp[--i];
    *b = 0;
}

/* Exactly 3 instructions per iteration: mla, subs, bne. `volatile` and the
 * "+r" constraints keep the loop intact; no memory operand appears in it, so
 * the instruction stream is the same at any -O level. */
static uint32_t cal_loop(uint32_t n) {
    uint32_t a = 1u;
    __asm volatile(
        "1:\n\t"
        "mla  %0, %0, %2, %3\n\t"
        "subs %1, %1, #1\n\t"
        "bne  1b"
        : "+r"(a), "+r"(n)
        : "r"(1103515245u), "r"(12345u)
        : "cc");
    return a;
}

/* Not inlined, so the call overhead is identical for both trip counts and
 * cancels in the slope. */
static uint32_t __attribute__((noinline)) run_cal(uint32_t n) { return cal_loop(n); }

static volatile uint32_t sink;

/* 3 instructions per iteration; the two counts are chosen so the difference is
 * a clean multiple of 3. Overridable so a tiny workload can be built to check
 * that the emulator setup works before committing minutes to a full run. */
#ifndef CAL_N0
#define CAL_N0 200000u
#endif
#ifndef CAL_N1
#define CAL_N1 800000u
#endif

int main(void) {
    SYST_RVR = SYST_MASK;
    SYST_CSR = 5; /* ENABLE | CLKSOURCE(core) | no interrupt */

    static const uint32_t iters[2] = { CAL_N0, CAL_N1 };
    uint32_t t[2], v[2];

    char msg[64], nb[12];
    for (int k = 0; k < 2; k++) {
        uint32_t t0 = rd_syst();
        sink = run_cal(iters[k]);
        uint32_t t1 = rd_syst();
        t[k] = (t0 - t1) & SYST_MASK; /* counts down */
        v[k] = iters[k] * 3u;

        int p = 0;
        const char *s = "CAL n=";
        while (*s) msg[p++] = *s++;
        put_uint(iters[k], nb); for (int i = 0; nb[i]; i++) msg[p++] = nb[i];
        s = " insn="; while (*s) msg[p++] = *s++;
        put_uint(v[k], nb); for (int i = 0; nb[i]; i++) msg[p++] = nb[i];
        s = " ticks="; while (*s) msg[p++] = *s++;
        put_uint(t[k], nb); for (int i = 0; nb[i]; i++) msg[p++] = nb[i];
        msg[p++] = '\n'; msg[p] = 0;
        sh_write0(msg);
    }

    /* Slope in milli-ticks per instruction, so the answer survives integer
     * division at the third decimal. f_ref in MHz is then
     * 1000 x ticks_per_insn == <milli> milli-MHz, i.e. divide by 1000. */
    uint32_t dinsn = v[1] - v[0];
    uint32_t dtick = t[1] - t[0];
    uint32_t milli = (uint32_t)(((uint64_t)dtick * 1000u) / dinsn);

    int p = 0;
    const char *s = "SLOPE milli_ticks_per_insn=";
    while (*s) msg[p++] = *s++;
    put_uint(milli, nb); for (int i = 0; nb[i]; i++) msg[p++] = nb[i];
    s = " f_ref_mMHz="; while (*s) msg[p++] = *s++;
    put_uint(milli, nb);
    for (int i = 0; nb[i]; i++) msg[p++] = nb[i];
    msg[p++] = '\n'; msg[p] = 0;
    sh_write0(msg);
    sh_exit();
}
