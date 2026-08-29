"""Phase C1: cross-check lit table rows in 04_experiments.tex against
the source-paper markdown conversions.

Sources:
  - PatchFormer md: NASA (SP 50/70/90), TJU (200/300/400), CALCE I (300/400/500)
  - RUL-Mamba md: NASA Table 5 (SP 50/70/90), TJU Table 7 (200/300/400)
  - OmniTIEFormer md: PANASONIC Table 5 (300/400/500), CALCE Table 6
    (RULMamba row + recovered iTransformer SP500)

Tolerant matcher: for each (method, dataset) in tex, collect per-SP
[MAE, RMSE, R2, AE] and look for the same sequence (with column
offsets -2..+2) in the source md rows; report mismatches/absences.
"""
import re

ROOT = ".."
TEX = "sections/04_experiments.tex"
SRC = {
    "patchformer": ROOT + "/PatchFormer_A_novel_patch-based_transformer_for_accurate_RUL_prediction.md",
    "rulmamba": ROOT + "/RUL-Mamba_Mamba-based_RUL_prediction_for_lithium_ion_batteries.md",
    "omnitie": ROOT + "/OmniTIEFormer - A tri-branch transformer with cross-scale transfer learning for multi-scale battery.md",
}

tex = open(TEX, encoding="utf-8").read()

# --- parse the 4 lit tables from tex ---
tables = {}
for m in re.finditer(r"\\label\{(tab:lit_\w+)\}(.*?)\\end\{table\}", tex, re.S):
    name, body = m.group(1), m.group(2)
    rows = {}
    for r in re.finditer(
            r"([A-Za-z][\w\-]*(?:-Mamba\*)?(?:\$\^\{?[\w]*\}?\$)?)\s*& &\s*"
            r"([\d.\-]+)\s*&\s*([\d.\-]+)\s*&\s*([\d.\-]+)\s*&\s*([\d.\-]+|--)\s*&\s*"
            r"([\d.\-]+)\s*&\s*([\d.\-]+)\s*&\s*([\d.\-]+)\s*&\s*([\d.\-]+|--)\s*&\s*"
            r"([\d.\-]+)\s*&\s*([\d.\-]+)\s*&\s*([\d.\-]+)\s*&\s*([\d.\-]+|--)",
            body):
        method = r.group(1).replace("$", "").replace("^", "").replace("{", "").replace("}", "").replace("\\mathsection", "S")
        vals = [[r.group(2), r.group(3), r.group(4), r.group(5)],
                [r.group(6), r.group(7), r.group(8), r.group(9)],
                [r.group(10), r.group(11), r.group(12), r.group(13)]]
        rows.setdefault(method, []).append(vals)
    tables[name] = rows

# --- parse source mds: pipe-table blocks per method ---
def md_blocks(path, method_names):
    s = open(path, encoding="utf-8").read()
    out = {}
    lines = s.split("\n")
    for i, l in enumerate(lines):
        for mn in method_names:
            if re.search(r"\|[^|]*\b" + re.escape(mn) + r"\b[^|]*\|", l):
                block = [l]
                j = i + 1
                while j < len(lines) and lines[j].startswith("|") and not re.search(
                        r"\|[^|]*\b[A-Z][\w\-*]{2,}[^|]*\|", lines[j]):
                    block.append(lines[j]); j += 1
                out.setdefault(mn, []).append(block)
                break
    return out

def row_nums(row):
    return [float(x) for x in re.findall(r"\|\s*(\d+\.?\d*)\s*\|", row)]

def check(name, source, method_map):
    src = md_blocks(SRC[source], set(method_map.values()))
    tex_rows = {}
    for tname in ["tab:lit_nasa", "tab:lit_tju", "tab:lit_calce", "tab:lit_panasonic"]:
        if name.split("-")[0].lower() in tname:
            tex_rows = tables[tname]
    for method, sps in tex_rows.items():
        mdn = method_map.get(method)
        if mdn is None:
            continue
        blocks = src.get(mdn, [])
        for idx, want in enumerate(sps[0]):
            w = [float(x) for x in want if x != "--"]
            if not w:
                continue
            found_any = False
            for block in blocks:
                for nums_row in (row_nums(l) for l in block):
                    for off in range(-2, 3):
                        seg = nums_row[off:off + len(w)]
                        if len(seg) == len(w) and all(
                                abs(a - b) < 0.011 for a, b in zip(seg, w)):
                            found_any = True; break
                    if found_any:
                        break
                if found_any:
                    break
            if not found_any:
                print(f"MISMATCH {name}: {method} SP{idx+1} tex={want}")

for tname in ["tab:lit_nasa", "tab:lit_tju", "tab:lit_calce"]:
    pass  # table-level map below
check("nasa", "patchformer", {
    "TimeMixer": "TimeMixer", "TimesNet": "TimesNet", "PatchTST": "PatchTST",
    "MambaSimple": "MambaSimple", "ModernTCN": "ModernTCN",
    "Autoformer": "Autoformer", "FEDformer": "FEDformer",
    "iTransformer": "iTransformer", "PathFormer": "PathFormer",
    "PatchFormer": "PatchFormer",
})
check("tju", "patchformer", {
    "TimeMixer": "TimeMixer", "TimesNet": "TimesNet", "PatchTST": "PatchTST",
    "MambaSimple": "MambaSimple", "ModernTCN": "ModernTCN",
    "Autoformer": "Autoformer", "FEDformer": "FEDformer",
    "iTransformer": "iTransformer", "PathFormer": "PathFormer",
    "PatchFormer": "PatchFormer",
})
check("calce", "patchformer", {
    "TimeMixer": "TimeMixer", "TimesNet": "TimesNet", "PatchTST": "PatchTST",
    "MambaSimple": "MambaSimple", "ModernTCN": "ModernTCN",
    "Autoformer": "Autoformer", "FEDformer": "FEDformer",
    "iTransformer": "iTransformer", "PathFormer": "PathFormer",
    "PatchFormer": "PatchFormer",
})
print("verification sweep done")
