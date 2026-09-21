"""Convert every reported MAE (and RMSE) from normalized capacity to Ah.

DRY RUN BY DEFAULT -- prints a diff and a verification block, writes nothing.
Pass --apply to write.

Rules (all verified against afd3ed8 -- see src/check_table3_revs.py):
  * Table 3 baselines are RESTORED from afd3ed8, not multiplied back.  The
    earlier revision divided them by the range and rounded to 4 dp; reversing
    that drifts (GOTION ModernTCN: 0.0087*6.3904 = 0.0556, published 0.0553).
  * Our own rows are multiplied by the dataset's train-cell range.
  * RMSE shares the MAE dimension and moves with it; R^2 (dimensionless) and
    AE (cycles) do not move.
  * BOLD IS RECOMPUTED, inherited from neither revision.  afd3ed8 bolded on
    MIXED units -- baselines in Ah, ours normalized -- so at TJU SP400 it marks
    ModernTCN 0.0016 over our 0.0018, a comparison across two different scales.
    Restoring its cell text would import that error back.

Traps this file exists to avoid, each of which produced silently wrong output
while it was being written:

  1. The \\multirow text ITSELF contains \\\\ (between the dataset name and the
     \\emph list).  Splitting a table on \\\\ therefore tears every multirow in
     half and no row can be attributed to a dataset.  Mark the multirows in the
     whole body FIRST, then split.
  2. Only the first starting point of each block carries the dataset name, the
     later rows start empty, and a block's first row is preceded by \\midrule.
     Match a leading name alone and one row in three gets converted, with the
     wrong dataset's factor scattered over the rest.
  3. Re-attaching the marker to every row makes unmark() paste a full
     \\multirow block at the head of each of them.  Re-attach only where one
     was, and keep the newline that preceded it.

Usage:
    python src/apply_mae_ah.py                 # dry run
    python src/apply_mae_ah.py --diff          # dry run + full diff
    python src/apply_mae_ah.py --apply         # write
"""
import difflib
import io
import os
import re
import subprocess
import sys

_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
sys.path.insert(0, _SRC)

from make_figures import load_series, norm_setup   # noqa: E402

TEX = os.path.join(_ROOT, "paper", "sections", "04_experiments.tex")
REF = "afd3ed8"
KEY = {"PANASONIC": "panasonic", "TJU": "tju", "GOTION": "gotion",
       "MIT": "mit", "NASA": "nasa", "CALCE": "calce"}
MARK_RE = (r"\\multirow\{(\d+)\}\{\*\}\{\\shortstack\[l\]\{([A-Za-z\-]+)"
           r"(?:\$\^\{[^}]*\}\$)?\\+(?:\\emph\{[^}]*\})?\}\}")

RANGE = {}
for _ds, _k in KEY.items():
    _c, _tr, _tc, _W, _sp, _e = load_series(_k)
    _lo, _hi = norm_setup(_c, _tr)
    RANGE[_ds] = float(_hi - _lo)
print("train-cell ranges: " + "  ".join(f"{d}={RANGE[d]:.4f}" for d in KEY))


def scale_cell(c, k, dp=4):
    """c with every number multiplied by k, keeping \\textbf and $\\pm$."""
    if not re.search(r"[0-9]", c):
        return c
    return re.sub(r"-?[0-9]*\.?[0-9]+",
                  lambda m: f"{float(m.group(0)) * k:.{dp}f}", c)


def set_bold(cell, on):
    """Turn \\textbf on or off around the number in a table cell."""
    bare = re.sub(r"\\textbf\{([^}]*)\}", r"\1", cell)
    if not on:
        return bare
    m = re.search(r"-?[0-9]*\.?[0-9]+", bare)
    if not m:
        return cell
    return (bare[:m.start()] + "\\textbf{" + bare[m.start():m.end()] + "}"
            + bare[m.end():])


def value_of(cell):
    m = re.search(r"-?[0-9]*\.?[0-9]+", cell.replace("\\textbf{", ""))
    return float(m.group(0)) if m else None


def table_body(src, label):
    return src.split("\\label{%s}" % label)[1].split("\\end{tabular}")[0]


_ORIG = {}


def mark(body):
    """\\multirow{...}{...} -> [DS:NAME], remembering the original text."""
    def rep(m):
        _ORIG[m.group(2)] = m.group(0)
        return "[DS:%s]" % m.group(2)
    return re.sub(MARK_RE, rep, body)


def unmark(text):
    return re.sub(r"\[DS:([A-Za-z\-]+)\]", lambda m: _ORIG[m.group(1)], text)


def method_of(cell):
    return cell.replace("\\textbf{", "").replace("\\", "").strip()


def split_table(src, label):
    """-> list of records {prefix, marker, cells}, one per LaTeX row."""
    recs = []
    for raw in re.split(r"\\\\", mark(table_body(src, label))):
        m = re.search(r"\[DS:([A-Za-z\-]+)\]", raw)
        if m:
            recs.append({"prefix": raw[:m.start()], "marker": m.group(1),
                         "cells": raw[m.end():].split("&")})
        else:
            recs.append({"prefix": "", "marker": None, "cells": raw.split("&")})
    return recs


def join_table(recs):
    out = []
    for r in recs:
        head = r["prefix"] + ("[DS:%s]" % r["marker"] if r["marker"] else "")
        out.append(head + "&".join(r["cells"]))
    return "\\\\".join(out)


src_cur = io.open(TEX, encoding="utf-8").read()
src_ref = subprocess.run(
    ["git", "show", f"{REF}:paper/sections/04_experiments.tex"],
    cwd=_ROOT, capture_output=True, text=True, encoding="utf-8").stdout

# ============================================================ Table 3
body_new = table_body(src_cur, "tab:lit_all")
recs = split_table(src_cur, "tab:lit_all")
recs_ref = split_table(src_ref, "tab:lit_all")
assert len(recs) == len(recs_ref), "row counts differ between revisions"

# which rows are data rows, and for which dataset
data_rows = []           # (index, dataset, method)
cur_ds = None
for i, r in enumerate(recs):
    if r["marker"]:
        cur_ds = r["marker"]
    if cur_ds is None or len(r["cells"]) < 11:
        continue
    meth = method_of(r["cells"][1])
    if not meth or meth == "Method":
        continue
    data_rows.append((i, cur_ds, meth))
print(f"Table 3: {len(data_rows)} data rows")

# 1. set the MAE values
n_val = 0
for i, ds, meth in data_rows:
    k = RANGE[ds]
    for col in (2, 5, 8):
        before = recs[i]["cells"][col]
        recs[i]["cells"][col] = (scale_cell(before, k) if meth == "Ours"
                                 else recs_ref[i]["cells"][col])
        if recs[i]["cells"][col] != before:
            n_val += 1
print(f"Table 3: {n_val} MAE values set")

# 2. recompute the bold marks from the NEW values
n_bold = 0
for ds in {d for _, d, _ in data_rows}:
    for col in (2, 5, 8):
        rows_here = [(i, m) for i, d, m in data_rows if d == ds]
        vals = [(value_of(recs[i]["cells"][col]), i) for i, _ in rows_here]
        vals = [(v, i) for v, i in vals if v is not None]
        lo = min(v for v, _ in vals)
        for v, i in vals:
            before = recs[i]["cells"][col]
            recs[i]["cells"][col] = set_bold(before, abs(v - lo) < 1e-12)
            if recs[i]["cells"][col] != before:
                n_bold += 1
print(f"Table 3: {n_bold} bold marks changed")

new_body = unmark(join_table(recs))
assert new_body.count("\\multirow") == body_new.count("\\multirow"), \
    "multirow count changed -- the round trip mangled the table"
assert new_body.count("\n") == body_new.count("\n"), \
    "newline count changed -- rows were glued together"
out = src_cur.replace(body_new, new_body, 1)
assert out != src_cur, "Table 3 replacement did not apply"


# ============================================================ other tables
def convert_block(label, cols, factor_of, name, default_ds=None):
    """Convert `cols` in every data row, tracking the dataset across rows."""
    global out
    body = table_body(src_cur, label)
    recs_b = split_table(src_cur, label)
    n, cur, seen = 0, default_ds, set()
    for i, r in enumerate(recs_b):
        # column 0 carries the dataset name on the first row of each block; on
        # that row it is preceded by \midrule, so clean it before matching
        first = re.sub(r"^\\midrule\s*", "", r["cells"][0].strip())
        m = re.fullmatch(r"([A-Za-z][A-Za-z\-]*)", first)
        if m and m.group(1).upper() in RANGE:
            cur = m.group(1).upper()
        if cur is None or len(r["cells"]) <= max(cols):
            continue
        probe = [re.sub(r"^\\midrule\s*", "", c.strip()) for c in r["cells"][:2]]
        if not any(re.match(r"^\d+$", p) for p in probe):
            continue
        k = factor_of(cur)
        seen.add(cur)
        for col in cols:
            before = r["cells"][col]
            r["cells"][col] = scale_cell(before, k)
            if r["cells"][col] != before:
                n += 1
    new = join_table(recs_b)
    assert new != body, f"{name}: no change produced"
    out = out.replace(body, new, 1)
    print(f"{name}: {n} cells changed across datasets {sorted(seen)}")
    return n


convert_block("tab:tableA", (3, 4), lambda d: RANGE.get(d, 1.0), "Table 2 ")
convert_block("tab:ablation", (1, 2), lambda d: RANGE["PANASONIC"], "ablation",
              default_ds="PANASONIC")

# ================================================== independent re-parse
new_recs = split_table(out, "tab:lit_all")
ref_recs = split_table(src_ref, "tab:lit_all")
cur_recs = split_table(src_cur, "tab:lit_all")
bad = []
cur_ds = None
for i, r in enumerate(new_recs):
    if r["marker"]:
        cur_ds = r["marker"]
    if cur_ds is None or len(r["cells"]) < 11:
        continue
    meth = method_of(r["cells"][1])
    if not meth or meth == "Method":
        continue
    for col in (2, 5, 8):
        want = (round(value_of(cur_recs[i]["cells"][col]) * RANGE[cur_ds], 4)
                if meth == "Ours" else value_of(ref_recs[i]["cells"][col]))
        got = value_of(r["cells"][col])
        if got is None or abs(got - want) > 5e-5:
            bad.append(f"{cur_ds} {meth} col{col}: {got} != {want}")
    for col in (3, 4, 6, 7, 9, 10):           # R^2 and AE must be untouched
        if r["cells"][col] != cur_recs[i]["cells"][col]:
            bad.append(f"{cur_ds} {meth} col{col}: R2/AE changed")
print(f"re-parse: mismatches={len(bad)}")
for b in bad[:8]:
    print("   !", b)
assert not bad, f"{len(bad)} mismatches after conversion"

# bold must now sit exactly on the minimum of each (dataset, column)
bold_bad, bold_checked = [], 0
for ds in sorted({d for _, d, _ in data_rows}):
    idxs = [i for i, d, _ in data_rows if d == ds]
    for col in (2, 5, 8):
        vals = {i: value_of(new_recs[i]["cells"][col]) for i in idxs}
        vals = {i: v for i, v in vals.items() if v is not None}
        lo = min(vals.values())
        for i, v in vals.items():
            bold_checked += 1
            if ("\\textbf" in new_recs[i]["cells"][col]) != (abs(v - lo) < 1e-12):
                bold_bad.append(
                    f"{ds} {method_of(new_recs[i]['cells'][1])} col{col} "
                    f"v={v} min={lo}")
print(f"bold check: {bold_checked} cells checked, {len(bold_bad)} wrong")
for b in bold_bad[:8]:
    print("   !", b)

# ============================================================ report
d = list(difflib.unified_diff(src_cur.splitlines(), out.splitlines(),
                              "before", "after", lineterm="", n=0))
print(f"\n--- diff ({sum(1 for x in d if x[:1] in '+-' and x[1:2] not in '+-')} changed lines) ---")
if "--diff" in sys.argv:
    print("\n".join(d))
if "--apply" in sys.argv:
    # Refuse to write unless the file is exactly at HEAD.  Running this twice
    # would multiply every MAE by the range a second time, and the result would
    # still look plausible.
    dirty = subprocess.run(["git", "diff", "--quiet", "--", TEX],
                           cwd=_ROOT).returncode != 0
    if dirty:
        sys.exit(f"REFUSING TO APPLY: {TEX} already differs from HEAD.\n"
                 "  Revert it first (git checkout -- the file) or inspect the "
                 "existing change -- running this twice corrupts every value.")
    io.open(TEX, "w", encoding="utf-8", newline="").write(out)
    print(f"\nWROTE {TEX}")
    print(f"revert with:  git checkout -- paper/sections/04_experiments.tex")
else:
    print("[dry run -- pass --apply to write, --diff to print the diff]")
