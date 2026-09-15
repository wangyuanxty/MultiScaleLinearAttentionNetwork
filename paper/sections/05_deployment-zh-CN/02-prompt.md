# 翻译上下文（共享）

## 任务
将 `paper/sections/05_deployment.tex` 第 184–262 行（§5.5 *On latency and energy*）由英文译为简体中文。

## 目标读者与风格
- 读者：电池 / 能源方向研究者（学术）
- 风格：学术。正式语域，术语精确，允许复杂从句，但不欧化
- 原则：重写而非直译；事实、数据、逻辑与原文完全一致；数字零误差

## 内容背景
本节是一篇投稿 *Journal of Power Sources* 的论文中"片上部署"章节的第五小节。论文提出 DeltaCycle——一个多尺度线性注意力电池 SOH/RUL 预测模型，本节报告其在 Cortex-M3（QEMU 模拟）上的推理延迟与能耗实测。

关键背景：论文核心主张是**内存开销与上下文长度解耦**，而非"更快"。因此本节里所有关于延迟的陈述都带有主动限定语——这是作者应对审稿意见（"延迟是推算的，不是实测的"）的刻意策略。翻译必须保留这种"主动认账"的语气强度。

## 术语表（权威口径）

| English | 中文 |
|---|---|
| SysTick / tick | SysTick 节拍 / 节拍 |
| Cortex-M3, M7, M4F, M33 | 保持原文 |
| MPS2 board | MPS2 开发板 |
| SYSCLK | SYSCLK（保持缩写） |
| floating-point unit (FPU) | 浮点单元（FPU） |
| soft-float | 软浮点 |
| instruction-count timing | 指令计数计时 |
| guest / host | 客户机 / 宿主机 |
| down-counter | 递减计数器 |
| reload period | 重载周期 |
| exception handler | 异常处理程序 |
| seqlock | 顺序锁（seqlock） |
| cycles per instruction (CPI) | 每指令周期数（CPI） |
| Thumb-2 | Thumb-2（保持原文） |
| kernel（此处） | 内核实现（指矩阵乘法实现，非操作系统） |
| triple-loop matrix–vector product | 三重循环矩阵–向量乘法 |
| unrolling | 展开 |
| fused multiply–add (FMA) | 融合乘加 |
| wall-clock gain | 实际耗时收益 |
| duty cycle | 占空比 |
| quiescent load | 静态负载 |
| clock-invariant | 与主频无关 |
| in relative terms | 在相对意义上 |

## 翻译难点
1. `kernel` 此处指矩阵乘法实现，**不是操作系统内核**——须在译文中避免误读
2. 全部数字与单位零误差（见 `01-analysis.md` 第五节清单）
3. `\texttt{}` 内的命令行参数（`-icount shift=0,sleep=off`、`-O0`、`-O2`）原样保留
4. 公式 $E = cVN\times10^{-12}\ \mathrm{J}$ 保留数学结构
5. 末句 `Read 0.9 s as an estimate and 7.1 s as the figure the emulator itself supports.` 是祈使句，中文须保留祈使语气，不得弱化为"可以认为"
6. `assembled from three conservative choices` 是比喻（拼装），保留"人为构造、非模型固有"的言下之意

## 输出
简体中文，保留原文五段结构。LaTeX 标记移除，数学符号与数值以可读形式呈现。
