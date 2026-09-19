# 翻译分析：§5.5 On latency and energy

## 一、文本性质

- **来源**：`paper/sections/05_deployment.tex` 第 184–262 行（LaTeX 源码）
- **篇幅**：724 词（5 段）
- **语域**：学术论文正文（Journal of Power Sources 投稿）
- **功能**：报告嵌入式部署的延迟与能耗实测结果，并对测量方法与其局限做自我限定

## 二、领域与读者

- **领域**：嵌入式系统测量 + 锂离子电池 BMS 部署
- **目标读者**：电池/能源方向研究者，**多数不熟悉 MCU 计时计量学**
- **推论**：`seqlock`、`icount`、`Thumb-2`、`CPI` 等嵌入式术语需要首次出现时给中文 + 原文标注

## 三、语体判断

原文有一个显著特征：**限定语密度极高**。作者反复主动声明测量局限（模拟非硅片、CPI 是代表值、0.9 s 是估计值）。这不是啰嗦，是应对审稿意见 M4 的策略性写法。

**翻译必须保留这种"主动认账"的语气强度**——若译成圆滑的中文，会削弱作者刻意留下的诚实姿态。

原文还有一处刻意的强指令句：`Read 0.9 s as an estimate and 7.1 s as the figure the emulator itself supports.` 祈使句，写给审稿人看的。中文须保留祈使语气，不能弱化为"可以认为"。

## 四、术语表（本文档为本次翻译的权威口径）

| English | 中文 | 说明 |
|---|---|---|
| SysTick | SysTick 节拍 | Arm 内核系统定时器；tick 统一译"节拍" |
| tick | 节拍 | 与 SysTick 搭配 |
| Cortex-M3 / M7 / M4F / M33 | 保持原文 | 芯片型号，不译 |
| MPS2 board | MPS2 开发板 | |
| SYSCLK | SYSCLK | 系统时钟，保持缩写 |
| floating-point unit (FPU) | 浮点单元（FPU） | |
| soft-float | 软浮点 | 用软件例程模拟浮点 |
| instruction-count timing | 指令计数计时 | QEMU `-icount` 模式 |
| guest / host | 客户机 / 宿主机 | QEMU 术语 |
| down-counter | 递减计数器 | |
| reload period | 重载周期 | |
| exception handler | 异常处理程序 | |
| seqlock | 顺序锁（seqlock） | |
| cycles per instruction (CPI) | 每指令周期数（CPI） | |
| Thumb-2 | Thumb-2 | Arm 指令集，不译 |
| kernel | 内核实现 | 此处指矩阵乘法的实现，**不是操作系统内核**（关键歧义点） |
| triple-loop matrix–vector product | 三重循环矩阵–向量乘法 | |
| unrolling | 展开 | 循环展开 |
| fused multiply–add (FMA) | 融合乘加 | |
| wall-clock gain | 实际耗时收益 | 非"墙上时钟" |
| duty cycle | 占空比 | |
| quiescent load | 静态负载 | BMS 自身基础功耗 |
| clock-invariant | 与主频无关 | |
| relative terms | 相对意义上 | "in relative terms" |

## 五、翻译难点

1. **`kernel` 一词两义**。原文 `The third is the kernel, a plain triple-loop matrix–vector product with no unrolling` —— 这里指矩阵乘法的**实现内核**，与操作系统无关。译作"内核实现"并在括注中澄清，避免读者误读。
2. **数字与单位一个不能错**。4.88×10⁷、48,755,486 / 48,756,706 / 48,753,240、0.007%、3.39×10⁶、14.4×、1.95×10⁹、1.36×10⁸、CPI≈1.3、2.54×10⁹、1.76×10⁸、101 s、7.1 s、0.9 s、225、48.76×10⁶、0.02%、6×10⁻⁷、3×10⁻⁶、300 μA/MHz、3.3 V、2.5 J、0.17 J。
3. **`assembled from three conservative choices`** 是比喻（拼装），非字面"组装"。译"拼装出来"保留其"人为构造、非固有"的言下之意。
4. **LaTeX 标记**：`\texttt{}` 内容是命令行参数（`-icount shift=0,sleep=off`、`-O0`、`-O2`），须原样保留不译。公式 $E = cVN\times10^{-12}\ \mathrm{J}$ 保留数学结构。

## 六、语气基调

学术、克制、主动认账。不用"我们"做主语堆砌（原文多为无主语被动式），中文相应采用无主语或"这/它"承接，避免欧化。

---

# 追加：§5.1 Constraints and design goals

- **来源**：同行文件第 5–35 行（原 286 词，4 段）
- **语体**：与 §5.5 同为学术正文，但**限定语密度低得多**——§5.1 是立论（提出约束与主张），§5.5 是自辩（交代测量局限）。翻译策略相应不同：§5.1 要**立得住**，语气可以更肯定；§5.5 要**认得清**，语气必须保留主动示弱。

## §5.1 补充术语表

| English | 中文 | 说明 |
|---|---|---|
| microcontroller | 微控制器 | |
| battery management system (BMS) | 电池管理系统（BMS） | |
| protection and balancing logic | 保护逻辑、均衡逻辑 | BMS 的两项基本功能 |
| allocate dynamically | 动态分配内存 | |
| working set | 工作集 | 操作系统中借用来的概念，此处指预测器运行时占用的内存 |
| a liability rather than a feature | 是负担而非特性 | 保留原文"负资产 vs 卖点"的对立意 |
| activation scratch | 激活值暂存区 | `scratch` 在此指临时缓冲，非"草稿" |
| static pool allocator | 静态池分配器 | |
| refuses to over-allocate | 超量分配时直接报错 | 拟人化表述，中文以"拒绝"或"直接报错"皆可 |
| flash | flash | 嵌入式语境常态，不译"闪存"更自然 |
| const arrays | `const` 数组 | C 关键字，保留 |
| quadratic attention buffer | 二次注意力缓冲 | |
| decoupled from | 与……解耦 | |

## §5.1 翻译难点

1. **`$d_k\times d_v$`** 是符号不是数值，保留原符号形式，不做数字转换。
2. **`Section~\ref{sec:exp}`** 是跨节引用，译作"第 4 节"——注意**不要**写成"上文"或"实验部分"，因为节号是读者定位用的锚点。
3. **`We deliberately do \emph{not} claim`** 是本节的核心修辞：先主动放弃一个过大主张，再给出一个更窄但更硬的主张。中文必须保留"特意不声称 → 我们声称的更窄但更有用"这个让步-转折结构，不能合并成一句平铺直叙。
4. **`there is no load step and no start-up model transfer`** 两个"没有"是并列的具体事实（无加载步骤 / 无启动传输），不是同义反复，须分别译出。

---

# 追加：§5.2 Numerical fidelity across architectures

- **来源**：同行文件第 23–50 行（192 词，3 段 + 一个两项列表）
- **语体**：本节是全章**最硬**的一段——审稿模拟意见（`reviewer_report.md:184`）把它列为 "survives review well" 并要求 "keep as the centerpiece of the deployment section"。翻译的首要任务是**不弱化数字的确定性**。

## §5.2 标题的译法

`Numerical fidelity` 译作 **数值保真度**。

- 候选：`数值一致性`（强调两边相符）、`数值保真度`（强调部署实现忠实复现参考实现）
- 选定理由：本节比较的两组对象性质不同——(a) 宿主机 C vs PyTorch 是"复现保真"，(b) ARM vs x86 是"跨平台一致"。**"保真度"能同时覆盖两者，"一致性"只覆盖 (b)。**

## §5.2 补充术语表

| English | 中文 | 说明 |
|---|---|---|
| numerical fidelity | 数值保真度 | 见上 |
| patch branch | patch 分支 | `patch` 保留原文（PatchFormer 沿用的时序分块术语） |
| stage-query exchange | 阶段查询交换 | 对应 §3.3 的 stage-query mechanism |
| fused last-token readout | 融合末位 token 读出 | `last-token` 与 token 一起保留原文 |
| bit-exactness | 逐位一致性 | |
| kernel（此处） | 内核实现 | 指 GDN-2 层实现，非操作系统内核 |
| host / Cortex-M3 binary | 宿主机 / Cortex-M3 二进制文件 | |
| relative error | 相对误差 | |
| float32 ULP | float32 的 ULP | ULP = unit in the last place，末位单位，保留缩写 |
| instruction set | 指令集 | |
| the recurrence | 递推 | **不是"递归"**——GDN-2 是逐时刻的状态递推 |
| soft-float | 软浮点 | |
| agree to the last bit but one | 仅差最低一位 | 即除最低有效位外逐位一致 |

## §5.2 翻译难点

1. **`fidelity` 不译"精度"**。精度（precision）是测量概念，本节说的是"两种实现算出的结果相差多少"，是保真/一致，不是精度。译错会与 §5.3 的量化精度混淆。
2. **`the recurrence depends on`** 中的 `recurrence` 指 GDN-2 的**状态递推**（eq. 1），不是"递归"。译作"递推所依赖的 expf/logf/sqrtf/erff 调用"。
3. **`agree to the last bit but one`** 是英文习语（`last but one` = 倒数第二）。此处含义是"只在最低位上不同"，不能直译成"除最后一位之外都一致"——后者会被读成"有两位不同"。
4. **列表两项的对照关系必须保留**：第一项是"宿主机 C ↔ PyTorch"，第二项是"Cortex-M3 ↔ 宿主机"。**两个"对照"的基准不同**，译文中"对照"一词的两侧不能互换。
5. **`We can do better`** 是本节唯一的修辞句，语气自信而不失谦。中译"我们可以做得更好"保留其分量，不弱化为"可以进一步"。


