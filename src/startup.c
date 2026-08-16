/* startup.c — minimal Cortex-M3 for QEMU lm3s6965evb with rdimon
 * 向量表直接指向 rdimon crt0 的 _mainCRTStartup:
 * 它负责 .data 拷贝/.bss 清零 + initialise_monitor_handles + stdio 初始化,
 * main 返回后 exit() 会 flush 缓冲区并 SYS_EXIT 退出 QEMU。
 * 不要自己写 _reset 跳 main——那会绕过 newlib 的流初始化,printf 静默失败。 */
#include <stdint.h>
extern uint32_t _estack;
extern void _mainCRTStartup(void);

__attribute__((section(".vector_table")))
const void *vectors[] = { (void*)&_estack, (void*)_mainCRTStartup };
