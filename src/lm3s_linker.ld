MEMORY
{
  FLASH (rx)  : ORIGIN = 0x00000000, LENGTH = 256K
  RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = 64K
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

__bss_start__ = ADDR(.bss);
__bss_end__   = ADDR(.bss) + SIZEOF(.bss);
