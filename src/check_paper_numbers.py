"""Audit number use across the manuscript, beyond the table cross-checks.

src/check_mae_consistency.py asks whether a prose MAE appears in some table.  It
therefore cannot catch a claim that is internally consistent but wrong, and it
says nothing about percentages, correlation coefficients, figure captions or
footnote markers.  Those are the blind spots this script covers:

  1. footnote markers vs notes -- a table carrying "10*" with no note, or a note
     opening with an asterisk that no cell carries;
  2. figure-caption numbers that appear nowhere else, which usually means the
     caption quotes a single run while the table quotes a ten-seed mean;
  3. the same claim stated with different numbers in different places.

Read-only: parses paper/DeltaCycle_Multi-Scale_Linear_Attention.tex, writes
nothing.

Usage: python src/check_paper_numbers.py
"""

import io
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "paper", "DeltaCycle_Multi-Scale_Linear_Attention.tex")
BS = chr(92)
src = io.open(TEX, encoding="utf-8").read()

ENV = re.compile(BS * 2 + r"begin\{(table\*?|figure\*?)\}(.*?)"
                 + BS * 2 + r"end\{\1\}", re.S)
envs = [(m.group(1), m.group(2)) for m in ENV.finditer(src)]

NUM = re.compile(r"-?\d+\.?\d*")
MARK = BS * 2 + r"(?:ast|dagger|ddagger)"


def numbers(text):
    return [float(n) for n in NUM.findall(text) if n.strip("-")]


def env_label(body, fallback):
    m = re.search(BS * 2 + r"label\{(tab:[a-z_]+|fig:[a-z_]+)\}", body)
    return m.group(1) if m else fallback


print("=" * 72)
print("1) footnote markers vs notes")
print("=" * 72)
# Inside a table a marker must occur at least twice: once on the cell it
# qualifies and once in the note that explains it.  Cells write ^{\dagger}
# while notes write $^{\dagger}$, so match the superscript, not the dollars.
problems = 0
for i, (kind, body) in enumerate(envs, 1):
    if not kind.startswith("table"):
        continue
    counts = {}
    for sym in re.findall(r"\^\{(" + MARK + r")\}", body):
        counts[sym] = counts.get(sym, 0) + 1
    if not counts:
        continue
    unpaired = sorted(s for s, c in counts.items() if c < 2)
    problems += len(unpaired)
    print(f"  {env_label(body, f'env#{i}'):16s} " +
          "  ".join(f"{s}={counts[s]}" for s in sorted(counts)) +
          (f"   UNPAIRED {unpaired}" if unpaired else "   ok"))
print(f"  -> {problems} unpaired" if problems else "  -> all paired")

print()
print("=" * 72)
print("2) numbers in figure captions that appear nowhere else")
print("=" * 72)
tables = " ".join(b for k, b in envs if k.startswith("table"))
table_nums = set(numbers(tables))
body_nums = set(numbers(ENV.sub("", src)))
orphans = 0
for kind, env in envs:
    if not kind.startswith("figure"):
        continue
    cap = re.search(BS * 2 + r"caption\{(.*?)\n\}", env, re.S)
    if not cap:
        continue
    name = env_label(env, "figure")
    for n in sorted(set(numbers(cap.group(1)))):
        if n in table_nums or n in body_nums:
            continue
        orphans += 1
        ctx = " ".join(cap.group(1).split())
        j = ctx.find(str(n))
        print(f"  {name:14s} {n:<10g} ...{ctx[max(0, j - 45):j + 35]}...")
print(f"  -> {orphans} caption numbers with no counterpart elsewhere"
      if orphans else "  -> every caption number also appears elsewhere")

print()
print("=" * 72)
print("3) same claim, different numbers")
print("=" * 72)
CLAIMS = [
    ("rate head improves MAE by",
     r"improves?[^.]{0,30}MAE by\s*\$?(\d+(?:\.\d+)?)"),
    ("GOTION relative error",
     r"relative error stays at[^.]*"),
    ("IR-capacity correlation",
     r"correlation with capacity[^.]*|correlated with capacity at[^,.]*"),
    ("exchange lowers MAE by", r"lowers? MAE at every starting point, by[^.]*"),
    # Narrow on purpose: a loose "drop30 ... AE" pattern straddles the figure
    # caption (one run) and the impulse footnote and reports a false positive.
    ("worst-case AE vs", r"worst-case AE\s*\$?(\d+)\$?\s*(?:vs\.?|vs)\s*\$?(\d+)"),
]
for name, pat in CLAIMS:
    hits = [(m.start(), m.group(0)) for m in re.finditer(pat, src, re.I)]
    if len(hits) < 2:
        continue
    vals = [(src[:pos].count("\n") + 1, tuple(NUM.findall(txt)))
            for pos, txt in hits]
    flag = "   <-- DIFFERS" if len({v for _, v in vals}) > 1 else ""
    print(f"  {name}{flag}")
    for line, v in vals:
        print(f"      L{line:<5d} {v}")
