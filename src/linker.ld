MEMORY
{
  FLASH (rx) : ORIGIN = 0x00000000, LENGTH = 256K
  RAM   (rw) : ORIGIN = 0x20000000, LENGTH = 64K
}

_estack = ORIGIN(RAM) + LENGTH(RAM);

SECTIONS
{
  .vector_table : { KEEP(*(.vector_table)) } > FLASH
  .text : { *(.text*) } > FLASH
  .rodata : { *(.rodata*) } > FLASH
  _etext = .;              /* FLASH 镜像终点 = .data 拷贝源 */
  .data : {
    _sdata = .;            /* RAM 中 .data 起点 */
    *(.data*)
    _edata = .;            /* RAM 中 .data 终点 */
  } > RAM AT> FLASH
  .bss : {
    _sbss = .;
    __bss_start__ = .;    /* rdimon crt0 (未使用但需链接) 引用 */
    *(.bss*)
    _ebss = .;
    __bss_end__ = .;
  } > RAM
  .heap : {
    __end__ = .;
    end = .;
  } > RAM
}
