"""Preview: what every MAE in the paper becomes if we report Ah, not normalized.

READ-ONLY.  Parses Table 2 and Table 3 out of paper/DeltaCycle_Multi-Scale_Linear_Attention.tex,
multiplies each dataset's MAE column by that dataset's real train-cell range
(hi - lo from make_figures.load_series -- NOT the rated capacity), and writes a
before/after listing to src/mae_ah_preview.txt for the author to check.

Nothing in this script writes to the paper.

Usage: python src/convert_mae_to_ah.py
"""
import io
import os
import re
import sys

_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
sys.path.insert(0, _SRC)

from make_figures import load_series, norm_setup   # noqa: E402

PAPER = os.path.join(_ROOT, "paper", "DeltaCycle_Multi-Scale_Linear_Attention.tex")
OUT = os.path.join(_SRC, "mae_ah_preview.txt")   # lives with the paper it describes

ORDER = ["PANASONIC", "TJU", "GOTION", "MIT", "NASA", "CALCE"]
KEY = {"PANASONIC": "panasonic", "TJU": "tju", "GOTION": "gotion",
       "MIT": "mit", "NASA": "nasa", "CALCE": "calce"}

# ---------------------------------------------------------------- ranges
RANGE = {}
print("train-cell ranges (the divisor each dataset's MAE is normalised by)")
for ds in ORDER:
    caps, train_cells, test_cell, W, sps, eol = load_series(KEY[ds])
    lo, hi = norm_setup(caps, train_cells)
    RANGE[ds] = float(hi - lo)
    print(f"  {ds:10s} lo={lo:9.4f}  hi={hi:9.4f}  range={hi - lo:9.4f}  "
          f"(rated-cell EOL at {eol} Ah)")

# ------------------------------------------------- table 3 (comparison)
src = io.open(PAPER, encoding="utf-8").read()
t3 = src.split("\\label{tab:lit_all}")[1].split("\\end{tabular}")[0]
t3 = re.sub(
    r"\\multirow\{(\d+)\}\{\*\}\{\\shortstack\[l\]\{([A-Za-z\-]+)"
    r"(?:\$\^\{[^}]*\}\$)?\\+(?:\\emph\{[^}]*\})?\}\}",
    lambda m: "\\multirow{%s}{*}[DS:%s]" % (m.group(1), m.group(2)), t3)

rows3, cur = [], None
for line in (r.strip() for r in re.split(r"\\\\", t3)):
    m = re.search(r"\\multirow\{\d+\}\{\*\}\[DS:([A-Za-z\-]+)\]", line)
    if m:
        cur = m.group(1)
        line = line[m.end():]
    cells = [c.strip() for c in line.split("&")]
    if len(cells) < 11 or cur is None:
        continue
    meth = cells[1].strip()
    if not meth or "\\" in meth.replace("\\textbf", ""):
        continue
    vals = []
    for c in cells[2:11]:
        mm = re.search(r"([0-9]*\.?[0-9]+)", c.replace("\\textbf{", ""))
        vals.append(float(mm.group(1)) if mm else None)
    rows3.append((cur, meth, [vals[0], vals[3], vals[6]]))

# ------------------------------------------------- table 2 (ours, per-SP)
t2 = src.split("\\label{tab:tableA}")[1].split("\\end{tabular}")[0]
rows2, cur = [], None
for line in (r.strip() for r in re.split(r"\\\\", t2)):
    cells = [c.strip() for c in line.split("&")]
    if len(cells) < 6:
        continue
    if cells[0] and not cells[0].startswith(("\\", "$")):
        cur = cells[0].replace("\\", "").split("$")[0].strip().upper()
    if cur not in RANGE or not re.match(r"^\d+$", cells[1].strip()):
        continue
    mae = re.search(r"([0-9]*\.?[0-9]+)", cells[3])
    rmse = re.search(r"([0-9]*\.?[0-9]+)", cells[4]) if len(cells) > 4 else None
    if mae:
        rows2.append((cur, cells[1].strip(), float(mae.group(1)),
                      float(rmse.group(1)) if rmse else None))

# tab:ablation quotes the SAME PANASONIC numbers as Table 2/3 for the exchange
# arm, so it moves with them or the three tables disagree in public
t_ab = src.split("\label{tab:ablation}")[1].split("\end{tabular}")[0]
rows_ab = []
for line in (r.strip() for r in re.split(r"\\\\", t_ab)):
    cells = [c.strip() for c in line.split("&")]
    if len(cells) < 4 or not re.match(r"^\d+$", cells[0]):
        continue
    a = re.search(r"([0-9]*\.?[0-9]+)", cells[1])
    b = re.search(r"([0-9]*\.?[0-9]+)", cells[2])
    if a and b:
        rows_ab.append((cells[0], float(a.group(1)), float(b.group(1))))

# ---------------------------------------------------------------- report
L = []
L.append("MAE in normalized capacity  ->  MAE in Ah   (multiply by train range)")
L.append("")
L.append("ranges used:")
for ds in ORDER:
    L.append(f"  {ds:10s} {RANGE[ds]:9.4f}")
L.append("")
L.append("=" * 78)
L.append("TABLE 3  (comparison with baselines)")
L.append("=" * 78)
for ds in ORDER:
    L.append(f"-- {ds}  (range {RANGE[ds]:.4f})")
    for dsx, meth, v in rows3:
        if dsx != ds:
            continue
        k = RANGE[ds]
        cells = "  ".join(f"{x:7.5f} -> {x * k:8.5f}" for x in v)
        L.append(f"   {meth:12s} {cells}")
L.append("")
L.append("=" * 78)
L.append("TABLE 2  (ours, per starting point)")
L.append("=" * 78)
L.append(f"{'dataset':10s} {'SP':>4s} {'MAE norm':>10s} {'MAE Ah':>10s} "
         f"{'RMSE norm':>10s} {'RMSE Ah':>10s}")
for ds, sp, v, r in rows2:
    rs = "None" if r is None else f"{r:10.5f}"
    ra = "None" if r is None else f"{r * RANGE[ds]:10.5f}"
    L.append(f"{ds:10s} {sp:>4s} {v:10.5f} {v * RANGE[ds]:10.5f} {rs:>10s} {ra:>10s}")

L.append("")
L.append("=" * 78)
L.append("tab:ablation  (PANASONIC, factor 1.0194)")
L.append("=" * 78)
for sp, noex, ex in rows_ab:
    k = RANGE["PANASONIC"]
    L.append(f"SP{sp}  no-exchange {noex:.4f} -> {noex * k:.4f}   "
             f"+exchange {ex:.4f} -> {ex * k:.4f}")

# rank check: the unit is a per-dataset positive scalar, so order cannot move
L.append("")
L.append("=" * 78)
L.append("RANK CHECK")
L.append("=" * 78)
nfirst = 0
for ds in ORDER:
    d = {meth: sum(v) / 3 for dsx, meth, v in rows3 if dsx == ds}
    if "Ours" not in d:
        continue
    cands = {k: v for k, v in d.items() if k != "Ours"}
    bk = min(cands, key=cands.get)
    win = d["Ours"] < cands[bk]
    nfirst += int(win)
    L.append(f"{ds:10s} ours {d['Ours']:.5f} vs {bk:12s} {cands[bk]:.5f}"
             f"  ratio {d['Ours'] / cands[bk]:.2f}   |  Ah: "
             f"{d['Ours'] * RANGE[ds]:.5f} vs {cands[bk] * RANGE[ds]:.5f}"
             f"  ratio {d['Ours'] / cands[bk]:.2f}   {'1st' if win else '2nd'}")
L.append("")
L.append(f"first on {nfirst} of 6")

io.open(OUT, "w", encoding="utf-8").write("\n".join(L))
print(f"\nwrote {OUT}  ({len(L)} lines)")
print("\n".join(L[-10:]))
