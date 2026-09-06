"""Localize fig_traj.png: ylabels "capacity (Ah)" / xlabels "cycle" / legend
"true","pred" -> Chinese (SimHei), everything else (titles, ticks, curves,
spines, swatches, frame) untouched.

All edited areas sit on pure-white background -> erase = white fill.
"""
from PIL import Image, ImageDraw, ImageFont
import numpy as np

SRC = "paper/figures/fig_traj.png"
DST = "paper/figures_zh/fig_traj.png"
FONT_HEI = "C:/Windows/Fonts/simhei.ttf"

im = Image.open(SRC).convert("RGB")
orig = np.asarray(im).astype(int)
draw = ImageDraw.Draw(im)


def erase(box):
    """Fill the box with white (bg is pure white; guard against colored ink)."""
    x0, y0, x1, y1 = box
    sub = orig[y0:y1, x0:x1]
    sat = sub.max(2) - sub.min(2)
    n = int((sat > 40).sum())
    if n:
        print(f"WARN {n} colored px inside erase box {box}")
    draw.rectangle(box, fill=(255, 255, 255))


def draw_text(text, x, y, size, anchor):
    font = ImageFont.truetype(FONT_HEI, size)
    draw.text((x, y), text, font=font, fill=(0, 0, 0, 255), anchor=anchor)


def draw_rot(text, cx, cy, size, rot=90):
    """Rotated text; ink center aligned to (cx, cy). rot=90 -> reads bottom-up."""
    tmp = Image.new("RGBA", (400, 400), (0, 0, 0, 0))
    td = ImageDraw.Draw(tmp)
    font = ImageFont.truetype(FONT_HEI, size)
    td.text((200, 200), text, font=font, fill=(0, 0, 0, 255), anchor="mm")
    tmp = tmp.rotate(rot, expand=True)
    bb = tmp.getbbox()
    im.paste(tmp, (int(cx - (bb[0] + bb[2]) / 2),
                   int(cy - (bb[1] + bb[3]) / 2)), tmp)


# ---- measured geometry (ink bboxes measured on the source) ----
EM_AXIS = 17   # fontsize 8 @150 dpi = 16.7 px
EM_LEG = 12    # fontsize 6  @150 dpi = 12.5 px

# ylabel k0 (CALCE): ink cols 18-35, rows 123-242, center (26.5, 182.5)
eraser_yl_k0 = (12, 117, 42, 248)
# ylabel k3 (PANASONIC): ink cols 18-35, rows 476-595, center (26.5, 535.5)
eraser_yl_k3 = (12, 470, 42, 602)
# xlabel k3 (PANASONIC): ink cols 248-292, rows 714-731, baseline ~727.5
eraser_xl_k3 = (244, 710, 296, 735)
# xlabel k4 (TJU): ink cols 686-730
eraser_xl_k4 = (682, 710, 734, 735)
# legend "true": ink cols 413-439, rows 57-65, baseline ~65.5 (upper leg box
# region has no gridline -> fill (409,53,445,70))
eraser_lt = (409, 53, 445, 70)
# legend "pred": ink cols 413-443, rows 75-88 (rows 72-74 clean except the
# horizontal gridline band at row 74 -> KEEP row 74, fill only rows 75-89)
eraser_lp = (409, 75, 445, 89)

ERASES = [eraser_yl_k0, eraser_yl_k3, eraser_xl_k3, eraser_xl_k4,
          eraser_lt, eraser_lp]
# audit boxes: for "pred" the declared area includes the kept gridline row 74
BOXES = [eraser_yl_k0, eraser_yl_k3, eraser_xl_k3, eraser_xl_k4,
         eraser_lt, (409, 72, 445, 90)]

for b in ERASES:
    erase(b)

draw_rot("容量 (Ah)", 26.5, 182.5, EM_AXIS)
draw_rot("容量 (Ah)", 26.5, 535.5, EM_AXIS)
draw_text("循环", 270, 727.5, EM_AXIS, "ms")
draw_text("循环", 708, 727.5, EM_AXIS, "ms")
draw_text("真实", 413.5, 65.5, EM_LEG, "ls")
draw_text("预测", 413.5, 84.5, EM_LEG, "ls")

# ---- containment audit ----
final = np.asarray(im).astype(int)
d = (np.abs(orig - final).sum(2) > 12)
mask_out = np.ones_like(d)
for x0, y0, x1, y1 in BOXES:
    mask_out[y0:y1, x0:x1] = False
outside = int((d & mask_out).sum())
changed = int(d.sum())
if outside:
    r, c = np.where(d & mask_out)
    for x, y in zip(c, r):
        print("  leak", (x, y), "orig", tuple(orig[y, x]), "new", tuple(final[y, x]))
print(f"changed px: {changed}, outside declared boxes: {outside}")
assert outside == 0, "leak outside declared boxes!"

import os
os.makedirs("paper/figures_zh", exist_ok=True)
im.save(DST)
print("saved", DST)
