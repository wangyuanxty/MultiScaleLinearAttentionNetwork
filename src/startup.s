.syntax unified
.cpu cortex-m3
.thumb
.word _estack
.word _start
.section .text
.global _start
_start:
    bl main
    b .
