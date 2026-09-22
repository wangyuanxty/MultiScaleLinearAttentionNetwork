# 给下一个模型的提示词（可直接复制）

---

你要帮我收尾一张学术论文的**图片摘要（graphical abstract）**。图已经做出来 90%，
问题是版面拥挤、有文字重叠、不够美观。**不要从零重画**，在现有脚本上改。

## 文件

```
src/make_graphical_abstract_v5.py     ← 要改的脚本，452 行，能跑，15 秒出图
src/GA_HANDOFF.md                     ← 交接说明，先读这个
submission/ga_v5.png                  ← 当前效果
submission/_ga_v5_preview500.png      ← 实际显示尺寸（500×200），判断可读性看这张
submission/ga_ref_B.png               ← 设计参考（AI 生成，只参考构图，数字是编的，别抄）
```

## 硬约束（不要试图绕过）

1. **画布必须 2.5:1**。Elsevier 原文：*"If you are submitting a larger image,
   please use the same ratio (500 wide x 200 high)"*，且会被缩放塞进
   **500×200 像素**的窗口。改高会被压扁。当前是 13.28 × 5.31 cm。
2. **物理尺寸无意义**。不管声明多少厘米都显示成 500×200，决定可读性的是
   **字高 ÷ 画布高**。5pt 在 5.31cm 画布上是 3.32%，屏幕上约 6.6px。
3. **不用守 5pt**。那是 nature-figure skill 的规矩，不是 Elsevier 的。
   Elsevier 只要求"在 500×200 下可读"。当前刻度/主张句 4.5pt、标题 5.5pt。
4. 字体只能用 Times / Arial / Courier / Symbol。当前用 Arial。
   注意 **Arial 没有 U+207B**，`×10⁻³` 会渲染成豆腐块。
5. 图内不能出现期刊 logo，不能有 "Graphical Abstract" 这种标题。
6. 文字用**近黑**（#1A1A1A / #3A3A3A）。之前用 #767676 灰字，缩略图里根本看不清。

## 结构（跟 Elsevier 官方例图一致）

他们的模板图中间那栏明写 "Here is where you showcase your methodology"，
三张已发表例图也都带方法栏。所以：

```
标题栏
METHOD | EVIDENCE | OUTCOME     三栏，每栏一个填色栏头
CONCLUSION 框（全宽）            他们 AJKD 那张例图本身就是表格
```

## 数据

**每一个数字都是论文里的真值，已经逐格核对过，不要改动数字。**
出处见 `GA_HANDOFF.md` 的表格。其中交换那组（22/30 配对改善、
均值 −5.7%、p=0.014）逐位复现了论文 `tab:ablation`。

## 已知的坑（文档里有详细版，这里列最致命的）

- `fig.text()` 收的是**图坐标比例**，不是数据坐标。给 log 轴传
  `fig.text(8800, 48, ...)` 会把文字画到画布外 10⁴ 倍远，matplotlib 3.11
  上直接崩在字体变换溢出。
- matplotlib 把**刻度数字和轴标题画在 axes 矩形之外**。只按 axes 高度算，
  必然压到下一行。每个图要 `图高 + 刻度带`，左边要留刻度数字的槽。
- `make_figures.py` 在 import 时设了 `savefig.bbox='tight'` 和
  `axes.grid=True`，而 matplotlib 的 rcParams 是**合并**不是替换。
  import 顺序错了：画布比例会变成 3.6:1，或者网格线穿过每一个标签。
- 这个尺寸下，5pt 文字的高度**正好等于六行柱状图里一行的行高**，
  所以参考线无处可藏 —— 数值要放进 y 轴刻度标签，不能放柱子上。
- `fig.patches` 的 zorder 和 axes 打平，背景矩形会盖住整张图。用负 zorder。

## 现在具体哪里不对

1. **图的刻度行压到上一张图**：`64 / 400 / 800` 那行贴在区间图下沿，
   `792 / 840 / 880` 贴在物理图上。`TICK_BAND = 0.036` 对 4.5pt 行来说太小。
2. **`memory (KB)` 竖排标签越出 OUTCOME 栏**，压到 EVIDENCE 栏的主张条上。
3. **`runs in C on a Cortex-M3`** 是裸文字，不是像其他四处那样的浅色主张条。
4. **六库柱状图的数值标签丢了**（删轴标题时一起没了）。

## 工作方式（重要）

- **砍，不要塞。** 500×200 下读者只看得进**形状和大约五个数字**。
  我这两小时每一次重叠，都是因为往画布里多塞了一样东西。
  建议优先级：三行 SP 表压成一行 → 三个 patch 芯片压成一行 →
  GDN-2 行并进去 → 把腾出的高度全给三张证据图。
- **不要盯着碰撞审计的数字优化。** `audit_figure_collisions.py` 会把
  一排连续刻度文字**合并成一个 bbox**（单个 `"0"` 被报成 54pt 宽），
  然后把这一排报成和邻居重叠 —— 23 条 FAIL 里一大半是这种假象。
  每条都要对着渲染图核，是假象就记下来放过。
- **改完必须跑这两个审计**（在仓库根目录）：
  ```
  python .claude/skills/nature-figure/scripts/audit_pdf_text.py submission/ga_v5.pdf
  python .claude/skills/nature-figure/scripts/audit_figure_collisions.py submission/ga_v5.pdf
  ```
- 脚本里已有**游标式布局**（每个元素从上一个的下沿往下排）和
  `assert y >= BODY_B` 断言。超支会当场报出来 —— 信它，不要靠肉眼估。
- 跑之前先 `export PYTHONIOENCODING=utf-8`，否则 `R²` 这类字符会让
  print 崩在 GBK 编码上。

## 交付

改好的 `src/make_graphical_abstract_v5.py`，跑出 `submission/ga_v5.pdf` +
`.png`，并告诉我：改了什么、两个审计的结论、以及**你认为还可以再砍掉什么**。
