"""Diff Table 3 between commit afd3ed8 and the working tree.

READ-ONLY.  Purpose: before converting MAE back to Ah, establish exactly which
cells changed between the revision that still held the published Ah values
(afd3ed8) and the current one.  Anything that differs for a reason OTHER than
the unit conversion must be known before editing, or the edit will silently
revert it.

Prints an added / removed / changed summary.  Writes nothing.

Usage: python src/check_table3_revs.py
"""
import io
import os
import re
import subprocess

_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
TEX = "paper/sections/04_experiments.tex"


def grab(src):
    """-> {(dataset, method): [9 raw MAE/R2/AE strings]} from Table 3."""
    t = src.split("\\label{tab:lit_all}")[1].split("\\end{tabular}")[0]
    t = re.sub(r"\\multirow\{(\d+)\}\{\*\}\{\\shortstack\[l\]\{([A-Za-z\-]+)"
               r"(?:\$\^\{[^}]*\}\$)?\\+(?:\\emph\{[^}]*\})?\}\}",
               lambda m: "[DS:%s]" % m.group(2), t)
    rows, cur = {}, None
    for line in (r.strip() for r in re.split(r"\\\\", t)):
        m = re.search(r"\[DS:([A-Za-z\-]+)\]", line)
        if m:
            cur = m.group(1)
            line = line[m.end():]
        cells = [c.strip() for c in line.split("&")]
        if len(cells) < 11 or cur is None:
            continue
        meth = cells[1].replace("\\textbf{", "").replace("\\", "").strip()
        if not meth or meth == "Method":
            continue
        rows[(cur, meth)] = [re.sub(r"[\\${}]|textbf", "", c) for c in cells[2:11]]
    return rows


old_src = subprocess.run(
    ["git", "show", "afd3ed8:" + TEX], cwd=_ROOT, capture_output=True,
    text=True, encoding="utf-8").stdout
old = grab(old_src)
new = grab(io.open(os.path.join(_ROOT, TEX), encoding="utf-8").read())

print(f"afd3ed8 Table 3: {len(old)} rows")
print(f"working tree   : {len(new)} rows\n")

print("only in working tree (added since afd3ed8):")
for k in new:
    if k not in old:
        print("   +", k)
print("only in afd3ed8 (removed since):")
for k in old:
    if k not in new:
        print("   -", k)

print("\ncells that differ:")
n_rows = 0
for k in new:
    if k not in old:
        continue
    d = [(i, a, b) for i, (a, b) in enumerate(zip(old[k], new[k])) if a != b]
    if d:
        n_rows += 1
        print(f"   {k[0]:10s} {k[1]:12s} {len(d)}/9 cells   "
              + "  ".join(f"col{i}: {a}->{b}" for i, a, b in d[:3]))
print(f"\n{n_rows} rows differ")
