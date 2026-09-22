"""Verify every Table-3 baseline cell against the literature it was copied from.

READ-ONLY.  Sources: the OmniTIEFormer and PatchFormer markdown conversions in
literature/.  Self-run PatchFormer / RUL-Mamba rows are excluded -- they come
from reference_repos/, not from either paper.

Table 3 is read from commit afd3ed8, i.e. BEFORE the unit conversion, so the
baselines are still in whatever unit their source printed.

Matching is by FINGERPRINT, not by method name: the first data row of every
table in both markdowns has an EMPTY method cell (the name is given in the
prose list above the table), so name matching silently drops exactly those
rows -- TimeMixer on TJU/NASA/CALCE and ModernTCN on GOTION were lost that way.
Two independent numbers (MAE and AE) agreeing at once is a far stronger
fingerprint than a name that may be blank.

Usage: python src/verify_table3_sources.py
"""
import io
import os
import re
import subprocess

_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)

OM = os.path.join(_ROOT, "literature",
                  "OmniTIEFormer - A tri-branch transformer with cross-scale "
                  "transfer learning for multi-scale battery.md")
PF = os.path.join(_ROOT, "literature",
                  "PatchFormer_A_novel_patch-based_transformer_for_accurate_"
                  "RUL_prediction.md")

SELF_RUN = {"PatchFormer", "RUL-Mamba", "Ours"}
ALIAS = {"CALCE": ["CALCE I", "CALCE"], "TJU": ["TJU"], "NASA": ["NASA"],
         "PANASONIC": ["PANASONIC"], "GOTION": ["GOTION"], "MIT": ["MIT"]}
SPMAP = {"PANASONIC": (300, 400, 500), "TJU": (200, 300, 400),
         "GOTION": (450, 600, 750), "MIT": (200, 300, 400),
         "NASA": (50, 70, 90), "CALCE": (300, 400, 500)}


def parse_md(path):
    """-> [(dataset, method_or_None, sp, MAE, AE, line)] for every table row."""
    out, ds, meth = [], None, None
    for ln, raw in enumerate(io.open(path, encoding="utf-8"), 1):
        if not raw.startswith("|"):
            ds = meth = None
            continue
        cells = [c.strip() for c in raw.strip().strip("|").split("|")]
        if len(cells) < 9 or cells[0] == "Dataset":
            continue
        if cells[0] and set(cells[0]) <= set(":- "):
            continue                      # separator row; "" is a continuation
        if cells[0]:
            ds = re.sub(r"\s+", " ", cells[0]).replace("\\", "")
        if len(cells) > 1 and cells[1]:
            meth = cells[1].replace("\\", "")
        if ds is None or not re.match(r"^\d+$", cells[2]):
            continue
        try:
            out.append((ds, meth, int(cells[2]), float(cells[5]),
                        float(cells[8]), ln))
        except ValueError:
            continue
    return out


def parse_table3(ref):
    """-> [(dataset, method, [MAE x3], [AE x3])] from the given commit."""
    txt = subprocess.run(
        ["git", "show", f"{ref}:paper/sections/04_experiments.tex"],
        cwd=_ROOT, capture_output=True, text=True, encoding="utf-8").stdout
    t3 = txt.split("\\label{tab:lit_all}")[1].split("\\end{tabular}")[0]
    t3 = re.sub(r"\\multirow\{(\d+)\}\{\*\}\{\\shortstack\[l\]\{([A-Za-z\-]+)"
                r"(?:\$\^\{[^}]*\}\$)?\\+(?:\\emph\{[^}]*\})?\}\}",
                lambda m: "\\multirow{%s}{*}[DS:%s]" % (m.group(1), m.group(2)), t3)
    rows, cur = [], None
    for line in (r.strip() for r in re.split(r"\\\\", t3)):
        m = re.search(r"\\multirow\{\d+\}\{\*\}\[DS:([A-Za-z\-]+)\]", line)
        if m:
            cur = m.group(1)
            line = line[m.end():]
        cells = [c.strip() for c in line.split("&")]
        if len(cells) < 11 or cur is None:
            continue
        meth = cells[1].replace("\\textbf", "").replace("\\", "").strip()
        if not meth or meth == "Method":
            continue
        vals = []
        for c in cells[2:11]:
            mm = re.search(r"([0-9]*\.?[0-9]+)", c.replace("\\textbf{", ""))
            vals.append(float(mm.group(1)) if mm else None)
        rows.append((cur, meth, [vals[0], vals[3], vals[6]],
                     [vals[2], vals[5], vals[8]]))
    return rows


om, pf = parse_md(OM), parse_md(PF)
print(f"OmniTIEFormer md: {len(om):4d} rows")
print(f"PatchFormer md  : {len(pf):4d} rows")
rows = parse_table3("afd3ed8")
print(f"afd3ed8 Table 3 : {len(rows)} method-blocks\n")

ok = miss = skip = 0
unmatched = []
for ds, meth, v, ae in rows:
    if meth in SELF_RUN:
        skip += 1
        continue
    verdict = []
    for k, sp in enumerate(SPMAP[ds]):
        hit = None
        for nm in ALIAS[ds]:
            for tag, raw in (("OM", om), ("PF", pf)):
                for r in raw:
                    if (r[0] != nm or r[2] != sp or r[3] is None
                            or ae[k] is None):
                        continue
                    if abs(r[3] - v[k]) <= 5e-5 and abs(r[4] - ae[k]) <= 0.05:
                        hit = (tag, r[1] or "(blank cell)", r[5])
        if hit is None:
            verdict.append(f"SP{sp}:NOT-FOUND")
            miss += 1
            unmatched.append((ds, meth, sp, v[k], ae[k]))
        else:
            verdict.append(f"SP{sp}:{hit[0]}OK")
            ok += 1
    flag = "  " if all("OK" in t for t in verdict) else "!!"
    print(f" {flag} {ds:10s} {meth:14s} " + "  ".join(verdict))

print(f"\nmatched {ok}   unmatched {miss}   self-run skipped {skip}")
if unmatched:
    print(f"\n{'dataset':10s} {'method':14s} {'SP':>4s} {'MAE':>9s} {'AE':>6s}")
    for r in unmatched:
        a = "None" if r[4] is None else f"{r[4]:.2f}"
        print(f"{r[0]:10s} {r[1]:14s} {r[2]:>4d} {r[3]:9.5f} {a:>6s}")
