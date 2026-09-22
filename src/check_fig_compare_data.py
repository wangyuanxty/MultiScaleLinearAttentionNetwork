"""Check fig_compare's hardcoded AMAE values against the current Table 3.

READ-ONLY.  fig_compare plots the three-SP mean MAE of every method from
tab:lit_all.  Its data is hardcoded, so it does not follow the table when the
table changes -- and it was written in the same mixed units Table 3 had
(baselines in Ah, ours normalized), which on a log axis overstated our margin
on GOTION by roughly six times.

Prints a per-method verdict.  Writes nothing.

Usage: python src/check_fig_compare_data.py
"""
import io
import os
import re

_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
TEX = os.path.join(_ROOT, "paper", "DeltaCycle_Multi-Scale_Linear_Attention.tex")
FIGPY = os.path.join(_SRC, "make_figures_extra.py")
MARK_RE = (r"\\multirow\{(\d+)\}\{\*\}\{\\shortstack\[l\]\{([A-Za-z\-]+)"
           r"(?:\$\^\{[^}]*\}\$)?\\+(?:\\emph\{[^}]*\})?\}\}")

# ------------------------------------------------- what the table says now
body = io.open(TEX, encoding="utf-8").read()
body = body.split("\\label{tab:lit_all}")[1].split("\\end{tabular}")[0]
body = re.sub(MARK_RE, lambda m: "[DS:%s]" % m.group(2), body)
table, cur = {}, None
for raw in re.split(r"\\\\", body):
    m = re.search(r"\[DS:([A-Za-z\-]+)\]", raw)
    if m:
        cur = m.group(1)
        raw = raw[m.end():]
    cells = [c.strip() for c in raw.split("&")]
    if len(cells) < 11 or cur is None:
        continue
    meth = cells[1].replace("\\textbf{", "").replace("\\", "").strip()
    if not meth or meth == "Method":
        continue
    vals = []
    for c in cells[2:11]:
        mm = re.search(r"[0-9]*\.?[0-9]+", c.replace("\\textbf{", ""))
        vals.append(float(mm.group(0)) if mm else None)
    mae = [vals[0], vals[3], vals[6]]
    if all(v is not None for v in mae):
        table[(cur, meth)] = sum(mae) / 3

# ------------------------------------------- what fig_compare has hardcoded
src = io.open(FIGPY, encoding="utf-8").read()
blk = src.split("def fig_compare()")[1].split("fig, axes")[0]
hard, ds = {}, None
for line in blk.splitlines():
    m = re.match(r'\s*"([A-Z]+)":\s*\{', line)
    if m:
        ds = m.group(1)
        continue
    if ds and line.strip().startswith('"'):
        for k, v in re.findall(r'"([A-Za-z\-]+)":\s*([0-9.]+)', line):
            hard[(ds, k)] = float(v)

print(f"table entries {len(table)}   fig_compare entries {len(hard)}\n")
print(f"{'dataset':10s} {'method':12s} {'table':>9s} {'figure':>9s}  verdict")
bad = 0
for key in sorted(set(table) | set(hard)):
    t, h = table.get(key), hard.get(key)
    if t is None or h is None:
        v = "MISSING from " + ("figure" if t else "table")
        bad += 1
    elif abs(t - h) <= 6e-5:
        v = "ok"
    else:
        v = f"DIFFERS  {t / h:.2f}x"
        bad += 1
    if v != "ok":
        ts = f"{t:.4f}" if t is not None else "-"
        hs = f"{h:.4f}" if h is not None else "-"
        print(f"{key[0]:10s} {key[1]:12s} {ts:>9s} {hs:>9s}  {v}")
print(f"\n{bad} entries disagree")

# ------------------------------------- the ranking claim, from the same parse
print(f"\n{'dataset':10s} {'ours':>9s} {'closest':>12s} {'its MAE':>9s} "
      f"{'ratio':>6s}  rank")
nfirst = 0
for ds in ("PANASONIC", "TJU", "GOTION", "MIT", "NASA", "CALCE"):
    ours = table.get((ds, "Ours"))
    if ours is None:
        continue
    cands = {m: v for (d, m), v in table.items() if d == ds and m != "Ours"}
    bk = min(cands, key=cands.get)
    win = ours < cands[bk]
    nfirst += int(win)
    print(f"{ds:10s} {ours:9.4f} {bk:>12s} {cands[bk]:9.4f} "
          f"{ours / cands[bk]:6.2f}  {'1st' if win else '2nd'}")
print(f"\nfirst on {nfirst} of 6; second on {6 - nfirst}")
