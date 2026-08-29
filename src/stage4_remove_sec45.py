"""Remove §4.5 Cross-dataset comparison discussion; redistribute its
content: positioning+summary+fig_compare -> §4.3 intro; CALCE AE
spread + self-run PatchFormer -> CALCE case study."""
import re

P = "../paper/sections/04_experiments.tex"
s = open(P, encoding="utf-8").read()

# ---- 1) build the §4.3 intro block (moved content) ----
intro_block = (
    "Battery-specific foundation and pretrained models published outside\n"
    "the per-SP RUL protocol family are not included in the comparison\n"
    "tables, because their reported metrics use different evaluation\n"
    "conventions (early-cycle prediction, or trajectory regression\n"
    "without per-SP RUL errors); head-to-head comparisons under our\n"
    "protocol are left to future work. Each case study below carries\n"
    "the per-dataset comparison table against PatchFormer\n"
    "\\cite{ref_patchformer}, RUL-Mamba \\cite{ref_rulmamba}, and the\n"
    "general time-series transformers, using the numbers reported in\n"
    "those papers on the shared test cells, EOL thresholds, and starting\n"
    "points (NASA B0005 and TJU CY25-1 are used identically by all\n"
    "works). AE is filled wherever the source paper reports it:\n"
    "RUL-Mamba's NASA and TJU rows carry its reported AE (an average\n"
    "over 10 runs), and on CALCE and PANASONIC---where RUL-Mamba's own\n"
    "paper has no results---we use OmniTIEFormer's baseline runs of\n"
    "RUL-Mamba \\cite{ref_omnitie}, which also report AE.\n"
    "Figure~\\ref{fig:compare} visualizes the per-SP trajectory MAE of\n"
    "the compared methods on the two datasets where all three\n"
    "battery-specific baselines overlap.\n"
)

# ---- 2) CALCE case study: fix forward reference + add the two moved paragraphs after the lit_calce table ----
calce_fix = ("amplifies a small trajectory bias into several cycles of RUL error,\n"
             "discussed in Section~\\ref{sec:compare}.")
calce_fix_new = ("amplifies a small trajectory bias into several cycles of RUL error,\n"
                 "as quantified below.")
assert calce_fix in s, "calce fix anchor not found"
s = s.replace(calce_fix, calce_fix_new)

calce_anchor = "\\end{table}\n\n\\subsubsection{NASA: short-history rapid assessment}"
calce_insert = (
    "\\end{table}\n\n"
    "\\emph{On the CALCE AE spread.} The AE column deserves an\n"
    "explicit note: the three seeds give 0/1/7 cycles (mean 2.7), and the\n"
    "maximum is not a training failure but a geometric property of the\n"
    "test cell. Near the EOL threshold the CALCE trajectory is nearly\n"
    "flat (local slope $\\approx 0.0022$\\,/cycle in normalized capacity,\n"
    "$\\approx 1.9$\\,mAh/cycle), so a local prediction bias of\n"
    "$\\sim$10\\,mAh near the tail---diluted to $\\le 0.001$ in the\n"
    "full-segment MAE---shifts the threshold crossing by 5--7 cycles.\n"
    "The same mechanism keeps PatchFormer's CALCE AE in a narrow band\n"
    "(0.8--1.1): on flat tails AE is a coarse-grained statistic; we report the\n"
    "mean and flag the seed-to-seed spread explicitly. On CALCE, where we\n"
    "additionally run PatchFormer's official code under our identical\n"
    "protocol, the self-run baseline gives AE $=$ 1 against our AE $=$ 0\n"
    "(R$^2$ $0.9955$ vs.\\ $0.9951$).\n\n"
    "\\subsubsection{NASA: short-history rapid assessment}"
)
assert calce_anchor in s, "calce table anchor not found"
s = s.replace(calce_anchor, calce_insert)

# ---- 3) insert the intro block before the CALCE subsubsection ----
anchor_intro = "\\subsubsection{CALCE: long-term monotonic monitoring}"
assert anchor_intro in s, "intro anchor not found"
s = s.replace(anchor_intro, intro_block + "\n" + anchor_intro)

# ---- 4) delete §4.5 wholesale ----
pat = re.compile(
    r"\\subsection\{Cross-dataset comparison discussion\}.*?(?=\\subsection\{Physics extension and interpretability\})",
    re.S,
)
assert pat.search(s), "section 4.5 not found"
s = pat.sub("", s, count=1)

open(P, "w", encoding="utf-8").write(s)
print("sec 4.5 removed; content redistributed to sec 4.3 intro and CALCE case study")
