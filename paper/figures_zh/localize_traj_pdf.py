"""Localize paper/figures/fig_traj.pdf -> paper/figures_zh/fig_traj.pdf.

In-place VECTOR text replacement: the English spans (x/y axis labels, legend
entries, chemistry subtitle) are redacted and re-inserted as Chinese at the
same optical centres.  Everything that is line art -- capacity curves,
gridlines, EOL rules, stage rules, legend swatches, spines -- is preserved:
redactions run with graphics=NONE, images=NONE and fill=None, so only glyphs
disappear and nothing is painted over.

Two fonts are used, so the result stays typeset rather than pasted:
  * SimHei           for CJK runs (same face as figures_zh/fig_traj.png)
  * DejaVuSerif.ttf  for the surrounding ASCII runs
DejaVuSerif.ttf is the very font matplotlib used here, so the Latin parts
("(Ah)", "(K=1)", "1.1 Ah", ...) keep their original shapes and metrics.

Matplotlib emits a multi-line title as a SINGLE PDF text object, so redacting
the chemistry line also deletes the dataset-name line above it.  The script
therefore re-extracts the text after redacting, finds every span that vanished
without being a target ("collateral"), and re-inserts it at its original
baseline, centred on its original advance box -- a matplotlib span rect IS the
advance box (x0 = text start, x1 = text start + advance width), so for the
centred multi-line titles this reproduces the original placement.

Geometry: a PyMuPDF insertion point is the baseline start; for rotate=0 the
text runs towards +x, for rotate=90 (bottom-up) towards -y.

Run from the repo root:  python paper/figures_zh/localize_traj_pdf.py
"""
import os

import numpy as np
import fitz
import matplotlib  # noqa: F401  (only to locate the bundled DejaVu Serif)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "paper", "figures", "fig_traj.pdf")
DST = os.path.join(ROOT, "paper", "figures_zh", "fig_traj.pdf")
ZH_TTF = "C:/Windows/Fonts/simhei.ttf"
LAT_TTF = os.path.join(os.path.dirname(matplotlib.__file__),
                       "mpl-data", "fonts", "ttf", "DejaVuSerif.ttf")

# --- translation table --------------------------------------------------------
# Title line 1 (CALCE / NASA / MIT / PANASONIC / TJU / GOTION) is a proper noun
# and keeps its wording; only the chemistry+capacity subtitle is translated.
TITLE_ZH = {
    "LCO · 1.1 Ah": "钴酸锂 (LCO) · 1.1 Ah",
    "LCO · 2.0 Ah": "钴酸锂 (LCO) · 2.0 Ah",
    "LFP · 1.07 Ah": "磷酸铁锂 (LFP) · 1.07 Ah",
    "NCA · 3.03 Ah": "镍钴铝酸锂 (NCA) · 3.03 Ah",
    "NCM+NCA · 2.5 Ah": "三元 (NCM)+镍钴铝 (NCA) · 2.5 Ah",
    "LFP · 27 Ah": "磷酸铁锂 (LFP) · 27 Ah",
}
LABEL_ZH = {
    "cycle": "循环次数",
    "capacity (Ah)": "容量 (Ah)",
    "true capacity": "真实容量",
    "prediction (K=1)": "预测值 (K=1)",
    "EOL threshold": "EOL 阈值",
}
LEFT_ALIGNED = {"true capacity", "prediction (K=1)", "EOL threshold"}
PAD = 0.6    # pt: redaction padding around an old glyph box
AUDIT_PAD = 2.0  # pt: extra slack when declaring "this area was allowed to change"

ZH, LAT = "zh", "lat"  # PDF resource names, registered with insert_font
FONTS = {ZH: fitz.Font(fontfile=ZH_TTF), LAT: fitz.Font(fontfile=LAT_TTF)}
ASC, DESC = FONTS[ZH].ascender, FONTS[ZH].descender  # 0.859 / -0.141


def zh_for(text):
    """-> (chinese, align) or (None, None) if this span is not translated."""
    if text in TITLE_ZH:
        return TITLE_ZH[text], "center"
    if text in LABEL_ZH:
        return LABEL_ZH[text], ("left" if text in LEFT_ALIGNED else "center")
    return None, None


def is_cjk(ch):
    return ("\u2e80" <= ch <= "\u9fff" or "\u3000" <= ch <= "\u303f"
            or "\uff00" <= ch <= "\uffef")


def runs(text):
    """(substring, font) runs: CJK -> SimHei, everything else -> DejaVu Serif."""
    out = []
    for ch in text:
        f = ZH if is_cjk(ch) else LAT
        if out and out[-1][1] == f:
            out[-1][0] += ch
        else:
            out.append([ch, f])
    return [(t, f) for t, f in out]


def width(rs, size):
    return sum(FONTS[f].text_length(t, fontsize=size) for t, f in rs)


def extents(rs, size):
    """Ascent/descent of the run mix, as multiples of the font size."""
    up = max(FONTS[f].ascender for _, f in rs)
    dn = min(FONTS[f].descender for _, f in rs)
    return up * size, dn * size


def collect(page):
    """Every text span, with the geometry needed to re-place it."""
    out = []
    for bl in page.get_text("dict")["blocks"]:
        if bl["type"]:
            continue
        for line in bl["lines"]:
            for sp in line["spans"]:
                out.append(dict(text=sp["text"], rect=fitz.Rect(sp["bbox"]),
                                origin=tuple(sp["origin"]), size=sp["size"],
                                rot=90 if line["dir"][1] != 0 else 0))
    return out


def emit(page, rs, size, rot, start):
    """Draw font runs in sequence on one baseline; @start is the baseline start."""
    pos = list(start)
    for text, f in rs:
        page.insert_text(tuple(pos), text, fontname=f, fontsize=size,
                         rotate=rot, color=(0, 0, 0))
        # rotate=0 advances towards +x, rotate=90 (bottom-up) towards -y
        pos[0 if rot == 0 else 1] += FONTS[f].text_length(text, fontsize=size) \
            * (1 if rot == 0 else -1)


def place(page, rs, size, rot, cx=None, x0=None, baseline=None, cy=None):
    """Blit runs anchored either by box centre/left + baseline, or centred on cy."""
    w = width(rs, size)
    up, dn = extents(rs, size)
    if rot == 0:
        x = cx - w / 2 if cx is not None else x0
        y = baseline if baseline is not None else cy + (ASC + DESC) / 2 * size
        box = fitz.Rect(x, y - up, x + w, y - dn)
        start = (x, y)
    else:  # rotate=90 reads bottom-up; ascent goes to -x, advance to -y
        x = cx + (ASC + DESC) / 2 * size
        y = cy + w / 2
        box = fitz.Rect(x - up, y - w, x - dn, y)
        start = (x, y)
    emit(page, rs, size, rot, start)
    return box


def build(out=DST, verbose=True):
    doc = fitz.open(SRC)
    page = doc[0]
    before = collect(page)
    targets = [s for s in before if zh_for(s["text"])[0] is not None]

    for t in targets:
        page.add_redact_annot(t["rect"] + (-PAD, -PAD, PAD, PAD), fill=None)
    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE,
                          graphics=fitz.PDF_REDACT_LINE_ART_NONE,
                          text=fitz.PDF_REDACT_TEXT_REMOVE)

    # Matplotlib packs a multi-line title into one text object: redacting the
    # subtitle also destroys the line above it.  Find and repair the fallout.
    def key(s):
        return (s["text"], round(s["origin"][0], 1), round(s["origin"][1], 1))

    survivors = {key(s) for s in collect(page)}
    tgt_keys = {key(t) for t in targets}
    collateral = [s for s in before if key(s) not in survivors and key(s) not in tgt_keys]

    page.insert_font(fontname=ZH, fontfile=ZH_TTF)
    page.insert_font(fontname=LAT, fontfile=LAT_TTF)

    boxes = []
    for s in collateral:  # verbatim repair on the original baseline
        r = s["rect"]
        box = place(page, runs(s["text"]), s["size"], s["rot"],
                    cx=(r.x0 + r.x1) / 2, baseline=s["origin"][1])
        boxes.append(r + (-PAD, -PAD, PAD, PAD) | box)
    for t in targets:
        zh, align = zh_for(t["text"])
        r = t["rect"]
        box = place(page, runs(zh), t["size"], t["rot"],
                    cx=(r.x0 + r.x1) / 2 if align == "center" else None,
                    x0=r.x0, cy=(r.y0 + r.y1) / 2)
        boxes.append(r + (-PAD, -PAD, PAD, PAD) | box)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    # Without subsetting, the full SimHei face (~10 MB) would be embedded.
    doc.subset_fonts()
    doc.save(out, garbage=3, deflate=True)
    doc.close()

    if verbose:
        print(f"{len(targets)} spans translated, {len(collateral)} repaired "
              f"-> {os.path.relpath(out, ROOT)}")
        for t in targets:
            zh, _ = zh_for(t["text"])
            print(f"  {t['text']:<18} -> {zh:<26} {t['size']:>4.1f}pt "
                  f"rot={t['rot']:>2} w={width(runs(zh), t['size']):5.1f}"
                  f"/{t['rect'].width:5.1f} runs={[f for _, f in runs(zh)]}")
        for s in collateral:
            print(f"  repaired: {s['text']!r} @ baseline "
                  f"{tuple(round(v, 1) for v in s['origin'])}")
    return targets, collateral, boxes


def render(path, dpi=300):
    pix = fitz.open(path)[0].get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)


def main():
    targets, collateral, boxes = build()

    # ---- containment audit: nothing outside the declared boxes may change ----
    a, b = render(SRC), render(DST)
    assert a.shape == b.shape, f"canvas changed: {a.shape} -> {b.shape}"
    scale = 300 / 72.0
    d = np.abs(a.astype(int) - b.astype(int))
    changed = d > 12
    mask = np.zeros_like(changed)
    for r in boxes:
        r = r + (-AUDIT_PAD, -AUDIT_PAD, AUDIT_PAD, AUDIT_PAD)
        mask[max(int(r.y0 * scale), 0):int(r.y1 * scale) + 1,
             max(int(r.x0 * scale), 0):int(r.x1 * scale) + 1] = True
    leak = int((changed & ~mask).sum())
    if leak:
        ys, xs = np.where(changed & ~mask)
        for x, y in list(zip(xs, ys))[:10]:
            print(f"  LEAK px=({x},{y}) pt=({x / scale:.1f},{y / scale:.1f}) "
                  f"{a[y, x]} -> {b[y, x]}")
    print(f"changed px: {int(changed.sum())} (declared boxes: "
          f"{int((changed & mask).sum())}), outside: {leak}")

    # how faithful is the verbatim repair? ink-box offset vs the original
    print("repair fidelity (ink bbox of the repaired line vs the original):")
    for s in collateral:
        r = s["rect"]
        win = (int((r.y0 * scale) - 6), int((r.x0 * scale) - 6),
               int((r.y1 * scale) + 6), int((r.x1 * scale) + 6))
        bb = []
        for im in (a, b):
            sub = im[win[0]:win[2], win[1]:win[3]] < 200
            ys, xs = np.where(sub)
            bb.append((win[1] + xs.min(), win[1] + xs.max(),
                       win[0] + ys.min(), win[0] + ys.max()))
        print(f"  {s['text']:<16} dx_left={((bb[1][0] - bb[0][0]) / scale):+.2f}pt "
              f"dx_right={((bb[1][1] - bb[0][1]) / scale):+.2f}pt "
              f"dy_top={((bb[1][2] - bb[0][2]) / scale):+.2f}pt")
    assert leak == 0, "ink changed outside the declared boxes"
    print("OK")


if __name__ == "__main__":
    main()
