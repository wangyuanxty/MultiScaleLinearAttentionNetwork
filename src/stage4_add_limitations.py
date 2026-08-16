"""Stage 4 edit: add Limitations subsection + Data/code availability."""
P1 = "../paper/sections/04_experiments.tex"
P2 = "../paper/main.tex"

s = open(P1, encoding="utf-8").read()
anchor = "\\label{tab:deploy}\n\\begin{tabular}{lccc}\n\\toprule\nConfig & fp32 AE & INT8 AE & Memory \\\\\n\\midrule\nSingle-branch & 2.2 & 2.3 & 337 KB \\\\\nMulti-scale & 1 & 1 & 340 KB \\\\\n\\bottomrule\n\\end{tabular}\n\\end{table}"
lim = anchor + r"""

\subsection{Limitations}
\label{sec:limitations}

All main results are aggregated over three training seeds
(42/43/44); the physics extension and the uncertainty study use a
single seed and a single calibration cell (CS2\_36), respectively,
and their associated variance is not quantified. The evaluation
protocol is single-step; multi-step rollout errors and the remaining
planned evaluations (full 124-cell MIT set, additional seeds, a
second calibration cell) are left to the final version. MCU latency
is projected from cycle-accurate emulation rather than measured on
silicon, and no on-device power figure is reported. All validation
is on public laboratory cycling data; field conditions such as
partial cycling, pack-level effects, and non-stationary loads remain
open."""
assert anchor in s, "deploy table anchor not found"
s = s.replace(anchor, lim)
open(P1, "w", encoding="utf-8").write(s)
print("limitations subsection added")

m = open(P2, encoding="utf-8").read()
old = "\\input{sections/05_conclusion}"
new = r"""\input{sections/05_conclusion}

\section*{Data and code availability}

The datasets used in this study are public: CALCE (Center for
Advanced Life Cycle Engineering, request access), NASA (Prognostics
Center of Excellence Data Repository), MIT/Stanford
(data.matr.io/1), PANASONIC and TJU (as distributed with the
PatchFormer and RUL-Mamba repositories). Code for training,
evaluation, figure generation, and the MCU deployment is available
at \url{https://github.com/wangyuanxty/MultiScaleLinearAttentionNetwork}. The
repository also contains the manuscript source (LaTeX)."""
assert old in m, "conclusion anchor not found"
m = m.replace(old, new)
open(P2, "w", encoding="utf-8").write(m)
print("data availability added")
