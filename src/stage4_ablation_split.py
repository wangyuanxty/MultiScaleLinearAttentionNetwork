"""Restructure sec:ablation to architecture-only; move physics routes
to sec:phys. Tolerates mixed line endings via regex DOTALL."""
import re

P = "../paper/sections/04_experiments.tex"
s = open(P, encoding="utf-8").read()

# 1) opening block: justification + architecture + injection routes -> architecture-only
pat1 = re.compile(
    r"All ablations are conducted on CALCE.*?"
    r"independently measured feature\.\s*\n",
    re.S,
)
new1 = (
    "This section ablates the architecture only; the physics extension is\n"
    "validated separately in Section~\\ref{sec:phys}. Ablations are\n"
    "conducted on CALCE, the designated primary dataset, whose monotonic\n"
    "profile isolates architectural effects from regeneration confounds;\n"
    "extending the architecture ablations to regeneration-rich datasets\n"
    "is scheduled as future work (Section~\\ref{sec:limitations}).\n\n"
    "The stage-query exchange is essential (seed 42, standard protocol;\n"
    "Table~\\ref{tab:ablation}): three branches without interaction\n"
    "degrade by 15\\% (MAE $0.0076$ vs $0.0066$), while the exchange\n"
    "outperforms a single-branch backbone ($0.0066$ vs $0.0070$).\n"
)
assert pat1.search(s), "block 1 not found"
s = pat1.sub(lambda m: new1, s, count=1)

# 2) table: 5 rows -> 3 rows
pat2 = re.compile(
    r"\\caption\{Architecture and objective ablation.*?\\end\{table\}",
    re.S,
)
new2 = (
    "\\caption{Architecture ablation on CALCE (seed 42, standard protocol,\n"
    "full-sequence MAE / R$^2$ / AE). Stage-query is the main model.}\n"
    "\\label{tab:ablation}\n"
    "\\footnotesize\n"
    "\\begin{tabular}{lcccc}\n"
    "\\toprule\n"
    "Configuration & Objective & MAE & R$^2$ & AE \\\\\n"
    "\\midrule\n"
    "single branch & z-score & 0.0070 & 0.9945 & 1 \\\\\n"
    "multi, no interaction & z-score & 0.0076 & 0.9938 & 0 \\\\\n"
    "\\textbf{multi + stage-query (main)} & z-score & \\textbf{0.0066} & \\textbf{0.9955} & \\textbf{0} \\\\\n"
    "\\bottomrule\n"
    "\\end{tabular}\n"
    "\\end{table}"
)
assert pat2.search(s), "block 2 not found"
s = pat2.sub(lambda m: new2, s, count=1)

# 3) remove duplicated bridge paragraph at end of sec:ablation
pat3 = re.compile(
    r"The rate head further transfers to the two regimes where physics is\n"
    r"expected to matter\..*?AE \$17\$ vs \$23\$\)\.\s*\n\s*\n",
    re.S,
)
assert pat3.search(s), "block 3 not found"
s = pat3.sub(lambda m: "", s, count=1)

# 4) sec:phys opening: carry injection-route study + attribution
pat4 = re.compile(
    r"The rate head of Section~\\ref\{sec:physics\} is the surviving mechanism"
    r".*?quantify both, and we\s*\nclose with the interpretability of the learned physics term\.",
    re.S,
)
new4 = (
    "We compared every physics injection route in the literature against\n"
    "the main backbone: (i) an objective regularizer with an\n"
    "Arrhenius-plus-IR rate prior; (ii) physics features concatenated to\n"
    "the input ($[C,\\mathrm{IR}]$ on CALCE; temperature and EIS\n"
    "resistance $[C,T,\\mathrm{Re},\\mathrm{Rct}]$ on NASA); (iii) a\n"
    "structural monotone head trained under the per-window z-score\n"
    "protocol. All three fail: the regularizer's constant-rate prior\n"
    "conflicts with the EOL acceleration tail ($R^2$ $0.61\\to0.34$ on\n"
    "extrapolation), the features are redundant with capacity history (no\n"
    "gain in any setting), and the structural head cannot represent the\n"
    "positive z-score targets of plateau segments (training loss stalls).\n"
    "The surviving mechanism---the rate head of\n"
    "Section~\\ref{sec:physics}---moves the constraint to absolute\n"
    "capacity space and to an independently measured feature; a control\n"
    "free head trained in the same absolute space ($0.0097$) isolates the\n"
    "mechanism from the objective, and the rate head itself improves\n"
    "CALCE MAE by 40\\% ($0.0042$ vs $0.0070$). Its value shows in the\n"
    "two regimes where physics is expected to matter;\n"
    "Figures~\\ref{fig:extrap} and \\ref{fig:robust} together with\n"
    "Tables~\\ref{tab:extrap} and \\ref{tab:robust} quantify both, and we\n"
    "close with the interpretability of the learned physics term."
)
assert pat4.search(s), "block 4 not found"
s = pat4.sub(lambda m: new4, s, count=1)

# 5) cross-ref in sec:compare
pat5 = re.compile(
    r"physics-consistent rate head \(unseen-tail\s*\nextrapolation and corruption robustness,\s*\nSections~\\ref\{sec:ablation\}--\\ref\{sec:phys\}\)",
    re.S,
)
new5 = (
    "physics-consistent rate head (unseen-tail\n"
    "extrapolation and corruption robustness,\n"
    "Section~\\ref{sec:phys})"
)
assert pat5.search(s), "block 5 not found"
s = pat5.sub(lambda m: new5, s, count=1)

open(P, "w", encoding="utf-8").write(s)
print("ablation restructured: architecture-only table; physics routes moved to sec:phys")
