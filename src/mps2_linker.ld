/* mps2_linker.ld — QEMU mps2-an385 (classic MPS2, Cortex-M3).
 *
 * Replaces bare_linker.ld for the large-model firmware. That one targeted
 * mps2-an505, which is an MPS2-TZ part: in QEMU its -kernel image does not
 * end up where the CPU fetches the reset vector, so a minimal program
 * lockups there regardless of the link address, the semihosting flags, or
 * the toolchain (checked with 13.2.rel1 and 15.2.rel1). The same ELF runs
 * unmodified on mps2-an385.
 *
 * The AN385 gives 4 MB of code SRAM and 4 MB of data SRAM, against the
 * 256 KB / 64 KB of lm3s6965evb, so both the fp32 (1.94 MB) and the int8
 * (484 KB) multi-scale weights fit. It is also a Cortex-M3, which is what
 * the paper reports.
 *
 * Run:
 *   arm-none-eabi-gcc -mcpu=cortex-m3 -mthumb -nostdlib -nostartfiles \
 *     -ffreestanding -O1 -T mps2_linker.ld -o mcu.elf mcu_main.c
 *   qemu-system-arm -M mps2-an385 -nographic -semihosting -kernel mcu.elf
 */
MEMORY
{
  FLASH (rx)  : ORIGIN = 0x00000000, LENGTH = 4M
  RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = 4M
}

_estack = ORIGIN(RAM) + LENGTH(RAM);

SECTIONS
{
  .vectors : { KEEP(*(.vectors)) } > FLASH
  .text    : { *(.text*) }        > FLASH
  .rodata  : { *(.rodata*) }      > FLASH
  .data    : { *(.data*) }        > RAM AT> FLASH
  .bss     : { *(.bss*) }         > RAM
}

/* consumed by the _start in mcu_main.c to zero .bss before main() */
__bss_start__ = ADDR(.bss);
__bss_end__   = ADDR(.bss) + SIZEOF(.bss);
