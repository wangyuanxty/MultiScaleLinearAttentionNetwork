# Paper Plan: DeltaCycle — 数字台账 + 决策历史

**Target Journal**: *Journal of Power Sources*
**说明**: 本文件只存两样东西——**决策历史**和**当前有效数字**。论文正文在 paper/;流程文档在 paper/ars_*.md;跨会话上下文在 Claude memory。过时的数字一律删除(旧管道数字已全部作废,由全量重训数字替代)。

---

## 0.1 2026-08-15 消融定案 & 主模型决定(当前)

### 主模型(全量训练配置)
**多分支(3 branches, patch 2/4/8)+ StageQuery V3 交互 + 直接头 + PatchFormer 相同归一化(z-score),不带物理。**
- 物理是消融实验的维度(已完成),不进主模型
- 归一化协议:输入全局 min-max(train lo/hi)+ 目标 per-window z-score,与 PatchFormer/RUL-Mamba 逐行同构(代码核实过)
- RoPE 已全删(类 + 前向应用 + MCU 头文件 + 全部引用);模型无位置编码

### 消融全谱(seed 42, CALCE,标准协议)
| 配置 | MAE | R² | AE | regen | 归属 |
|---|---|---|---|---|---|
| phys_ir 单分支(绝对空间) | **0.00427±0.00002** | 0.99706 | 1 | 0.164 | 物理扩展(10-seed) |
| phys_ir + 多尺度 + StageQuery | 0.0042 | 0.9971 | 1 | 0.163 | 组合测试:多尺度被物理头完全吸收 |
| **multi + StageQuery + direct(z-score)** | **0.0065±0.0003** | 0.9955 | 3.6 | 0.243 | **主模型**(10-seed) |
| single direct(z-score) | **0.0065±0.0003** | 0.9956 | 1.9 | 0.250 | 基线/最稳(10-seed) |
| multi 无交互(z-score) | 0.0068±0.0003 | 0.9954 | 4.2 | 0.255 | 消融(两数据集均最差, CALCE p=0.037) |
| direct + 绝对空间 | 0.0090±0.0058 | 0.9972 | 1.6 | 0.168 | 归因行(10-seed, 范围0.0052-0.0244) |

### 物理扩展(phys_ir:r = softplus(w·h) + softplus(γ)·IR,γ 恒正;Q̂ = Q_last − r;绝对空间)
- 物理消融:phys_ir vs direct_z = **0.00427 vs 0.0090**(−53% MAE, 10-seed)
- 外推(90/10):**0.7745**;对照 free head 0.374(2026-08-16 复现确认,旧值 0.4471 为 RoPE 移除前状态)
- 鲁棒性(同架构对照,4 模式全胜):
  | 模式 | phys_ir | direct_z |
  |---|---|---|
  | clean | 0.0043 / AE 1.0 | 0.0065 / AE 1.9 |
  | drop30 | **0.0090 / AE 17** | 0.0122 / AE 23 |
  | gauss | 0.0097 | 0.0111 |
  | impulse | 0.0061 | 0.0098 |
- 注入路线全谱(9+ 项):目标正则 ❌(常数先验+参数退化)/ 输入特征 [C,IR]、NASA T/EIS ❌(信息冗余)/ 结构头 z-score ❌(符号冲突)→ 绝对空间解锁。**机理核心洞察:per-window z-score 目标包含单调预测者无法产生的正值;绝对空间是物理头的必要条件但不是精度来源(direct-abs 0.0097 归因证明)**

### 创新点结构
- **I1**:GDN-2 + bit-exact MCU 部署(不变)
- **I2**:多尺度 StageQuery(主模型组件);10-seed 定稿:multi 无交互两数据集最差(CALCE p=0.037),交换恢复 CALCE 持平并 NASA 最优均值 0.0084±0.0004(p=0.32 不显著);单分支=稳定基线(AE 最稳)
- **物理速率头(phys_ir)**:扩展/消融小节,含 tested-and-rejected 叙事

---

## 0. 2026-08-13 决策(归一化 & 创新点调整)

1. **归一化切 per-window**(与 PatchFormer/RUL-Mamba 完全同协议):target 用窗口 mean/std z-score;K=32 在 per-window 下崩坏(AE 15.9→25.3)→ **K=32 创新点移除**
2. **AE 定义统一**:跨越式(`seg[i] ≥ th > seg[i+1]`),全脚本一致
3. **测试集 SP 截断**:与 PatchFormer 一致(SP−W 起)
4. **MIT 切分**:8 train / 2 test
5. 消融不含 physics 配置(物理归入独立消融维度)

---

## 1. 数字台账(当前有效)

### 1.1 主模型全量训练(多分支+StageQuery+direct+z-score)
`test_full_train.py`,results/full_train.json

| 数据集 | seed | MAE | R² | AE | 状态 |
|---|---|---|---|---|---|
| CALCE(W=64) | 42 | 0.0066 | 0.9955 | 0 | ✅ |
| NASA(W=30) | 42 | 0.0096 | 0.9920 | 6 | ✅ |
| MIT(W=64) | 42 | 0.0013 | 0.9999 | 0 | ✅ |
| PANASONIC(W=30) | 42 | 0.0037 | 0.9988 | 1 | ✅ |
| TJU(W=64) | 42 | 0.0013 | 0.9999 | 0 | ✅ |
| 全部 | 43 / 44 | — | — | — | ⏳(待补) |

### 1.1b 逐 SP 10-seed 协议(PatchFormer 对齐,2026-08-26 汇总)

协议(2026-08-17 拍板,严格 PatchFormer 一致性):每个 (SP, seed) 独立重训 —— 训练集 = 非测试电池全部序列 + **测试电池 SP 之前的循环**(NASADataPreProcess 约定);评估自 SP 段起;10 seeds(1..10)。产物:`checkpoints/per_sp/{ds}/SP{sp}_seed{seed}.pt` + `src/results/per_sp_train.json`;聚合(每 seed 先对测试电池平均,再对 10 seeds mean±std,ddof=1;AE 报 seed 均值)见 `src/agg_per_sp.py` → `src/results/per_sp_summary.json`。

| 数据集 | SP | TRUL | MAE | RMSE | R² | AE |
|---|---|---|---|---|---|---|
| CALCE | 300 | 339 | 0.0066±0.0007 | 0.0146±0.0013 | 0.9951±0.0008 | 3.2 |
|  | 400 | 239 | 0.0078±0.0010 | 0.0170±0.0019 | 0.9933±0.0015 | 2.5 |
|  | 500 | 139 | 0.0092±0.0016 | 0.0196±0.0028 | 0.9898±0.0030 | 2.4 |
| NASA | 50 | 73 | 0.0098±0.0018 | 0.0145±0.0013 | 0.9908±0.0018 | 0.9 |
|  | 70 | 53 | 0.0089±0.0009 | 0.0148±0.0005 | 0.9822±0.0011 | 0.4 |
|  | 90 | 33 | 0.0080±0.0009 | 0.0115±0.0011 | 0.9804±0.0036 | 0.5 |
| PANASONIC | 300 | 287 | 0.0037±0.0003 | 0.0096±0.0001 | 0.9976±0.0001 | 0.8 |
|  | 400 | 187 | 0.0036±0.0002 | 0.0101±0.0001 | 0.9965±0.0001 | 0.8 |
|  | 500 | 87 | 0.0040±0.0003 | 0.0111±0.0003 | 0.9936±0.0003 | 0.9 |
| TJU | 200 | 577 | 0.0013±0.0002 | 0.0020±0.0001 | 0.9999±0.0000 | 0.2 |
|  | 300 | 477 | 0.0014±0.0002 | 0.0021±0.0002 | 0.9998±0.0000 | 0.9 |
|  | 400 | 377 | 0.0018±0.0004 | 0.0026±0.0005 | 0.9996±0.0001 | 1.6 |
| MIT | 200/300/400 | — | ✅ 10-seed 完成(2026-08-29):SP200 MAE 0.0019/R² 0.9998/AE 0.9;SP300 0.0025/0.9997/0.9;SP400 0.0032/0.9995/1.1;per-cell AE 0.2/1.6;已入 Table A(全 30 配置收官) |  |  |  |

**与旧 tab:tableA(全量 3-seed 42/43/44)差异**:CALCE 持平(SP500 MAE 0.0092 vs 0.0083);NASA SP50 略降(0.0098 vs 0.0078)但 AE 更小(0.9 vs 2.0,旧表 AE 为数据集聚合值);PANASONIC 几乎零差别(旧表无 SP400 行,SP400 另补 0.0038/0.0103/0.9963);TJU 微降(SP400 MAE 0.0018 vs 0.0012、AE 1.6 vs 0.3)。**结论:逐 SP 训练(与 PatchFormer 同协议)下四数据集表现与全量训练基本等效,SOTA 叙事不因换协议受损(尤其 PANASONIC 0.0037 vs PatchFormer 0.0105)。**

### 1.2 基线参照(旧管道数字,已作废,待新数字替换)
PatchFormer / RUL-Mamba 论文数字见 paper 对比表;我方旧数字全部作废,不保留。

### 1.3 归一化协议(已核实,勿再改)
输入 = (x−lo)/(hi−lo),lo/hi 来自训练电池(≡ PatchFormer NASADataPreProcess.py:100-103 / RUL-Mamba NASA_Data_Process.py:138,额定容量除法在 min-max 中消掉);目标 = (y−wmean)/wstd(窗口统计);评估 = pred×wstd+wmean 反归一化。**phys_ir 用绝对空间目标(物理头在 z 空间无法训练——符号冲突,实证)。**

### 1.4 UQ / CQR(CALCE seed 42,风向标 C)
| 指标 | raw 分位数 | CQR 校准后 | 名义 |
|---|---|---|---|
| 覆盖率(总) | 0.856 | **0.933** | 0.95 |
| SP300/400/500 | 0.881/0.881/0.885 | **0.940/0.931/0.927** | 0.95 |
| 区间宽度 | 0.0286 | 0.0390 | — |
| P50 MAE | 0.0073(点模型 0.0066) | — | — |
- q_adj=0.0052(CS2_36 校准,n=869);残差缺口=有限样本+跨电池漂移,论文如实写
- ckpt: checkpoints/quantile_calce_seed42.pt;数据: results/quantile_uq.json

### 1.5 MCU 部署验证(2026-08-15 重写)
- C 推理(gdn2_mcu.c v3)与当前 GDN2Block 逐行对齐:patch embed(带 bias)、causal conv+SiLU、L2 norm(全 64 维!)、f_proj 门控、per-key-dim decay(dt[p] 源行)、per-head out_norm(RMSNorm eps=1.19e-7 机器精度)、g_proj silu、GELU 头
- x86 验证:C −1.169650 vs PyTorch −1.169648(相对误差 2.5×10⁻⁶,float32 舍入级)
- 调试坑清单:conv 核序反、proj bias 漏、L2 norm 块大小(64 非 16)、decay 下标 i→p、RMSNorm eps(1e-5→机器精度)、head 层 index 2→3(Sequential 里 GELU/Dropout 占位)
- ARM 验证:lm3s6965evb 上 libm 冒烟(expf/logf/sqrtf/erff)与 x86 **位级一致**(hex 40AB1399 = 9913AB40);纯算术位级一致为旧会话已证;全模型在 ARM 板实跑暂缓(全模型 ELF 482KB 超 lm3s 256K flash,an505 启动问题,非必需)——论文声明按 x86 全模型 <3e-6 + ARM 组件位级一致写

---

## 1.6 叙事定稿:四支柱(管理科学口径,2026-08-15)

目标口径:管理科学(可靠性运营/运营管理方向)——从"模型多强"转向"支撑什么运营决策、落地成本多少"。

| 支柱 | 证据 | 决策价值 |
|---|---|---|
| ① 精度第一梯队 | PANASONIC 全指标 SOTA(MAE 0.0037 vs PatchFormer 0.0105)、TJU 全指标 SOTA、CALCE R² 打平/MAE/AE 落后 PatchFormer(如实写)、MIT R² ≥0.9995 双测试(vs Zhao 2025 的 0.9902,协议差异注明)、再生显式跟踪(fig_regen) | 日常监测可信 |
| ② 物理增强(一模块两重价值) | 前瞻:90/10 extR² 0.7745±0.0001 vs 0.374(10-seed);抗损:4 模式 MAE 全胜(-13%~-34%),worst-case AE 自由头 574 vs rate 9(drop30) | 长期规划 + 现场数据韧性 |
| ③ 可信 | CQR 0.856→0.933 | 风险知情排程 |
| ④ 可部署 | MCU 8KB/340KB/2.5e-6 | 规模化近零成本 |

一句话:"第一梯队的观测精度(含再生)+ 物理增强的前瞻与抗损 + 决策级置信区间 + 近零成本的边缘部署。"

补充要点:
- SOTA 判定(对照 OmniTIEFormer 表 5/6 中其他模型的数字,定稿阶段自跑 PatchFormer PANASONIC/TJU 替换来源)
- MIT 全集训练定稿时补(8 train/2 test 双测试口径 R² ≥0.9995,2026-08-16 完整性检查修正:原管线只评了 1 个测试电池)
- 高端落地领域(2026-08-16 用户定稿,只留与验证口径契合的):EV 车队、无人机/机器人车队、电动工具/消费电子电池包(片上 MCU 电量计)、便携医疗设备(安全关键区间)、共享两轮车换电(排程)、梯次利用分选(再生跟踪);大电芯场景(储能/UPS/轨交)因 GOTION 排除+未验证,主动移除
- 配图更新:新增系统总览图(gpt-image-2,四支柱价值链)+ UQ 区间带图(matplotlib,quantile ckpt 数据)

## 2. 待办(2026-08-16 更新)

- ✅ 主模型 3-seed 全量训练(42/43/44)+ Table A mean±std + AE 均值(用户定稿:2026-08-16 去掉括号/中位数)
- ✅ 对比表拆分:4 张 per-dataset 小表入各 case-study;PANASONIC SP400 补算(0.0038/0.0103/0.9963);CALCE 加 RULMamba 行(OmniTIEFormer 基线 AE 12.5/12.5/13.0);RUL-Mamba 论文有 AE(Table5/7)——NASA 0.8/0.9/2.5、TJU 2.6/2.6/2.6 已填;并发现 TJU 行原抄错方法(Autoformer 值误作 RUL-Mamba*),已改 0.0014/0.0022/0.9998;PANASONIC iTransformer SP500 由 OmniTIEFormer 平均行反推(0.0203/0.0327/0.9278/1.4)
- ✅ CALCE AE seed 分布解释入文(0/1/7,均值 2.7;平尾斜率 0.0022/cycle ≈1.9 mAh/cycle,~10 mAh 局部偏置 → 5–7 cycle 平移;逐 epoch 诊断确认非过拟合)
- ✅ K=32 相关删除、Method StageQuery 重写;Sec4 重构(2026-08-16):tableA 移 case studies 前、sec:compare 改名 Cross-dataset comparison discussion 删 fig_metrics_sp、sec:phys 拆 3 subsubsection、图 12 张、per-dataset 口径矛盾句修正、non-recursive 全文降调、fig_compare 数字更新重绘
- ✅ 消融重构(2026-08-17 用户指令):tab:ablation 纯架构 3 行(single/multi 无交互/multi+stage-query),物理行(0.0097 归因、0.0042 速率头)与注入路线研究移入 sec:phys 开头;AE SP 不变性证明写入协议节;8KB 软化 fixed-size;CALCE 第二定位、NASA R² 第二;架构消融扩展 PANASONIC 列入定稿清单
- ✅ 物理章节补图:fig_extrap(0.374/0.775/−5.28)+ fig_robust(drop30 23→17)已入 sec:phys 三个 subsubsection;ckpt 存 checkpoints/phys_figs_models.pt、轨迹存 results/phys_figs.npz(重画零训练)
- ✅ 物理机制泄露审查(2026-08-16,用户要求):6 项全过(窗口/目标对齐、IR 只用最后一步因果、lo/hi 仅训练集、容量序列与主表逐点一致、前向填充无未来、跨 cell 隔离);三个诚实性注记入文:①IR 是同期健康信号(相关−0.98)非预测未来,4.6.1 机制表述改为 Q̂=Q_last−r 锚定+IR 同期感知;②IR 信号 cell 间不均(CS2_38 相关仅−0.12)已作 limitation;③drop30 明确定义为容量通道单独故障场景(电流传感器漂移/容量估计失效),全遥测掉线不在场景内
- ✅ CALCE seed 定案:44 重训复现 AE=7(确定性,非 bug);45=6;Table A 维持 42/43/44,AE 报均值 2.7;逐 epoch 诊断:AE 自 ep5 起 1↔7 振荡,非过拟合(ep10 即 7 且 train loss 持续下降)
- ✅【2026-09-03 完成】自跑 PatchFormer+RUL-Mamba(NASA B0005/TJU CY25-1,3 repeats,官方管线=mat/csv 官方读取+官方 TSD/Trainer;PF NASA batch16);PS PANASONIC 未跑(无官方运行器,omitie 数值保留);phys_ir 3 seeds、MIT 全集、MCU phys_ir 头验证仍待办
- ${table-注} tab:lit_nasa/tab:lit_tju 的 PF/RM 行已换自跑值并重核加粗;RM TJU 自跑复现文献(说明其 TJU 管线实为单变量,原 17-dim 脚注已删)

