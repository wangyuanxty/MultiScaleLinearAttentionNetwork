# 交接：MAE 从「归一化容量」换算成 Ah

## 为什么做

审稿意见："对能源期刊读者，归一化 MAE 是无意义的。电化学背景的审稿人看到
MAE = 0.0054 无法判断这到底好不好。"

**是对的。** 而且两条基线（PatchFormer、RUL-Mamba）本来就是 Ah：
`RUL_Prediction_PatchFormer.py:312-324` 在算 MAE 前把预测和真值都
`* args.Rated_Capacity` 乘回额定容量，连 EOL 阈值都是 `Rated_Capacity*0.7`。
OmniTIEFormer 那篇的表头也明写 `MAE (Ah)`。

论文原来的 Table 3 是**混单位**：我们的行是归一化，基线的行是 Ah。
`2f283c5` 那次"修复"把基线**除以量程**变成归一化 —— 方向反了，
整张表改成了一种基线和整个领域都不用的单位。

## 三条铁律

**① 换算因子是训练集量程 `hi − lo`，不是额定容量。**

```python
from make_figures import load_series, norm_setup
caps, train_cells, test_cell, W, sps, eol = load_series(ds)
lo, hi = norm_setup(caps, train_cells)      # ← 就是它
```

实测值（已算好，直接用）：

| 数据集 | lo | hi | 因子 (=hi−lo) | 额定 EOL |
|---|---|---|---|---|
| PANASONIC | 1.7591 | 2.7784 | **1.0194** | 2.12 Ah |
| TJU | 1.6535 | 2.4208 | **0.7673** | 1.75 Ah |
| GOTION | 21.7020 | 28.0923 | **6.3904** | 21.6 Ah |
| MIT | 0.8251 | 1.0861 | **0.2610** | 0.86 Ah |
| NASA | 1.1538 | 2.0353 | **0.8815** | 1.40 Ah |
| CALCE | 0.1422 | 1.1345 | **0.9923** | 0.77 Ah |

MIT 的 0.261 特别要注意：它的训练电芯只跨 0.83–1.09 Ah，所以归一化 MAE
0.0025 换算过去只有 0.66 mAh。**用额定容量 1.07 当因子是错的。**

**② 基线不要"乘回去"，要从 `git show afd3ed8` 恢复原值。**

`2f283c5` 把基线的 Ah 值除以量程后**四舍五入到 4 位**，再乘回去有舍入漂移。
`afd3ed8` 是那次改动之前的版本，基线的值就是两份文献的原印值
（已逐格核对，见下）。**只有我们自己的行需要真换算。**

**③ RMSE 和 MAE 一起改，R² 和 AE 不动。**

- RMSE 与 MAE 同为容量量纲，同一个因子 → **Table 2 的 RMSE 列也要改**
- R² 无量纲 → 不动
- AE 单位是"循环"，与容量尺度无关 → 不动
- Table 3 **没有** RMSE 列，只有 MAE / R² / AE

## 跨库汇总那一行

`04_experiments.tex:137`

```
Average across all SPs and seeds: AE = 3.4 cycles, R2 = 0.9947, MAE = 0.0054
```

**删掉 MAE，保留 AE 和 R²。** 六个库的电池完全不同（GOTION 27 Ah vs MIT
1.07 Ah），把它们的绝对误差平均在一起没有意义。AE 是循环数、R² 无量纲，
两者跨库可比。

## 核对现状：Table 3 的基线已全部验过

`src/verify_table3_sources.py`（只读，打印核对结果）：

```
100 格  脚本指纹核对通过（MAE + AE 双列同时相等）
  5 格  手工核对通过 —— 源 md 某几行多一个空列，解析器读不到，数字本身正确
 18 格  自跑的 PatchFormer / RUL-Mamba / Ours，不属于本核对范围
──────
105 / 105 可核格  全部有据可查
```

那 5 格是：`PANASONIC iTransformer SP500`、`GOTION iTransformer SP750`、
`GOTION PathFormer SP450/600/750`。例：源 md 的 GOTION PathFormer SP450 是
`0.1663 / AE 12.8`，我们表里也是 `0.1663 / AE 12.8`，逐位相同 —— 只是那行
的 markdown 格式畸形。

**没试过加容错，试过一次，把 CALCE 带崩了，已回退。** 别在收尾阶段再动
解析器。

来源分布（脚本自动判定，不靠约定）：
- PANASONIC / GOTION → OmniTIEFormer md
- NASA / TJU → PatchFormer md
- CALCE → 两边都有，逐格不同；**iTransformer SP300/400 来自 OM，SP500 来自 PF**
- MIT → 只有自跑的 PatchFormer 和 RUL-Mamba

## 排序不变，已验证

```
dataset      归一化 ours/best      Ah ours/best       排名
PANASONIC    0.00377/0.00773=0.49  0.00384/0.00788=0.49   1st
TJU          0.00150/0.00197=0.76  0.00115/0.00151=0.76   1st
GOTION       0.00803/0.00873=0.92  0.05134/0.05581=0.92   1st
MIT          0.00253/0.00320=0.79  0.00066/0.00084=0.79   1st
NASA         0.00890/0.00777=1.15  0.00785/0.00685=1.15   2nd
CALCE        0.00787/0.00693=1.13  0.00781/0.00688=1.13   2nd
```

**四个第一 + 两个第二，不变。** 一个库内所有人乘同一个正数，
比值和排序都不可能变。

## 全文改动清单（2026-09-21 扫描，务必逐条核对）

**要改的 9 处**（行号为扫描时状态，改前重新定位）：

| 位置 | 内容 | 数据集 | 因子 |
|---|---|---|---|
| `04_experiments.tex:137` | 汇总行 `MAE = 0.0054` | 跨库 | **删掉** |
| `04_experiments.tex:233` | `mean per-SP MAE: 0.0038, against 0.0077` | PANASONIC | ×1.0194 |
| `04_experiments.tex:236` | `SP300 (MAE 0.0032 ...)` | PANASONIC | ×1.0194 |
| `04_experiments.tex:239` | `(MAE 0.0040 vs 0.0164)` | PANASONIC | ×1.0194 |
| `04_experiments.tex:332` | `SP90 it leads on MAE (0.0080 vs 0.0082)` | NASA | ×0.8815 |
| `04_experiments.tex:451` | `CALCE MAE (0.00427±0.00002 vs ...)` | CALCE | ×0.9923 |
| `04_experiments.tex:597` | `P50 ... (MAE 0.0076±0.0007 vs ...)` | CALCE | ×0.9923 |
| `04_experiments.tex:641` | 表内 `P50 MAE & 0.0076 ... (0.0075)` | CALCE | ×0.9923 |
| `05_deployment.tex:45` | `(R2=0.9955, MAE=0.00581)` | **先确认是哪个库** | 待定 |

**不动的 6 处**（都是无量纲或单位无关）：

| 位置 | 内容 | 为什么不动 |
|---|---|---|
| `03_method.tex:160` | `5.7% on average, 22 of 30, p=0.014` | 百分比 / 计数 |
| `03_method.tex:210` | `MAE 0.0090±0.0058` | **先确认数据集**；若要改，用其对应因子 |
| `04_experiments.tex:367` | `≤0.001 in the full-segment MAE` | 阈值陈述，需人工判断是否随单位走 |
| `04_experiments.tex:399` | `5.7% mean reduction (p=0.014)` | 百分比 |
| `05_conclusion.tex:29` | `p=0.014` | 无量纲 |
| `01_intro.tex` / `highlights.txt` | 各 1 处 | 扫描未见数字，仍需人工过一遍 |

**风险点**：改表不改正文 = 论文自相矛盾，这是本次改动最大的失败模式。
**改完必须做的自查**：正文里每一个 MAE 数字，都要能在新表里按键（数据集, SP）
找到同一个值；找不到就是漏改。

## 还没做的

```
✗ Table 2 的 RMSE 列  —— src/convert_mae_to_ah.py 里 cells[3] 改成 cells[3:5]
✗ 正文 9 处 —— 清单见上，尚未动笔
✗ 图表 —— fig_compare 等若画的是 MAE 需重生成；GA 的六库比值柱无量纲，不受影响
```


## 文件

```
src/convert_mae_to_ah.py         生成对照清单（读 load_series + 解析 .tex，只打印）
src/mae_ah_preview.txt       已生成的对照：Table 3 全 + Table 2 的 MAE 列
src/verify_table3_sources.py     来源核对（100 自动 + 5 手工）
paper/sections/04_experiments.tex  要改的文件；Table 2 与 Table 3 都在这里
git show afd3ed8:paper/sections/04_experiments.tex   基线原值的来源
```

## 工作方式

**先出清单、逐格核对、再动文件。** 上一轮单位改动涉及 477 格，这次不能
在没有对照的情况下改 .tex。改完把 `verify_table3_sources.py` 的 ref 换成
新提交再跑一次，应当仍是 100/5。
