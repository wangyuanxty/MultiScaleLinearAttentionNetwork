"""Every MAE number in the prose must exist in a table.

READ-ONLY.  After converting the reported MAE from normalized capacity to Ah,
the danger is not a wrong value -- it is a value nobody updated, so the prose
and the tables disagree.  This collects every number that appears in a table,
then every number that appears within one line of "MAE"/"RMSE" in the prose,
and reports the prose numbers with no counterpart.

Percentages, p-values, R^2, AE and coverage are excluded: they are
dimensionless or measured in cycles and do not move with the unit.

Prints a verdict; writes nothing.

Usage: python src/check_mae_consistency.py
"""
import io
import os
import re

_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
SECTIONS = os.path.join(_ROOT, "paper", "sections")

# a line that is only about a dimensionless or cycle-scale quantity
NOT_MAE = re.compile(r"(p\s*=|coverage|nominal|width|R\\text|R\^?2|\bAE\b|%|"
                     r"\\times|seeds|cycles|epochs)")


def numbers(text):
    return set(re.findall(r"0\.\d{3,5}", text))


table_vals, prose_lines = set(), []
for fn in sorted(os.listdir(SECTIONS)):
    if not fn.endswith(".tex"):
        continue
    raw = io.open(os.path.join(SECTIONS, fn), encoding="utf-8").read()
    for body in re.findall(r"\\begin\{tabular\}(.*?)\\end\{tabular\}", raw,
                           flags=re.S):
        table_vals |= numbers(body)
    prose = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", "", raw, flags=re.S)
    for i, line in enumerate(prose.splitlines(), 1):
        if re.search(r"\bMAE\b|\bRMSE\b", line):
            prose_lines.append((fn, i, line))

orphan = []
for fn, i, line in prose_lines:
    if NOT_MAE.search(line):
        continue
    for v in numbers(line):
        if v not in table_vals:
            orphan.append((fn, i, v, line.strip()[:96]))

print(f"numbers appearing in tables : {len(table_vals)}")
print(f"prose lines mentioning MAE  : {len(prose_lines)}")
print(f"prose MAE numbers with no table counterpart: {len(orphan)}")
for fn, i, v, txt in orphan:
    print(f"   {fn}:{i}  {v}   {txt}")

# The set test above has a blind spot: a STALE value that happens to equal some
# legitimate table value elsewhere passes it.  Print every line so they can be
# read, which is the only check that does not depend on coincidence.
print("\n--- every prose line carrying an MAE/RMSE number ---")
for fn, i, line in prose_lines:
    nums = sorted(numbers(line))
    if nums:
        print(f"{fn}:{i}  {nums}\n      {line.strip()[:100]}")

# A sentence wraps: "...on MAE at every starting point\n(Table 3: 0.0013, ...)".
# Scanning one line at a time misses precisely the cases that were wrong here,
# so scan the following line too.
print("\n--- MAE mentioned but the numbers are on the NEXT line ---")
by_file = {}
for fn, i, line in prose_lines:
    by_file.setdefault(fn, {})[i] = line
for fn in sorted(by_file):
    raw = io.open(os.path.join(SECTIONS, fn), encoding="utf-8").read()
    prose = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", "", raw, flags=re.S)
    lines = prose.splitlines()
    for i, line in by_file[fn].items():
        if numbers(line):
            continue
        nxt = lines[i] if i < len(lines) else ""
        nn = sorted(numbers(nxt))
        if nn:
            print(f"{fn}:{i}->{i + 1}  {nn}\n      {line.strip()[:70]}\n      {nxt.strip()[:90]}")
