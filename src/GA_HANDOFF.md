# Graphical abstract — handoff brief

Script: `src/make_graphical_abstract_v5.py` (runs in ~15 s)
Output: `submission/ga_v5.pdf` + `.png` (3137×1254) + `_ga_v5_preview500.png`
Design references: `submission/ga_ref_B.png` (AI-generated, structure only —
its numbers are invented and must not be copied)

## Hard constraints

1. **Canvas must be 2.5:1.** Elsevier: *"If you are submitting a larger image,
   please use the same ratio (500 wide x 200 high)"*; the image is scaled to fit
   a **500 × 200 px** window on ScienceDirect. The script uses 13.28 × 5.31 cm.
   A taller canvas is not an option — it gets letterboxed.
2. **Physical size is irrelevant.** Whatever cm figure size you declare, it
   displays at 500×200, so what governs legibility is *text height ÷ canvas
   height*. A 5 pt glyph is 3.32 % of a 5.31 cm canvas → ~6.6 px on screen.
   Making the canvas 26 cm changes nothing.
3. **The 5 pt floor is optional.** It comes from the `nature-figure` skill, not
   from Elsevier. Elsevier only asks for readability at 500×200. The current
   script sits at 4.5 pt for ticks/claims and 5.5 pt for titles, which is a
   reasonable compromise; going to 4 pt buys another ~20 % of vertical space.
4. **Fonts**: Elsevier allow Times / Arial / Courier / Symbol. Script uses Arial.
   Note Arial lacks U+207B (superscript minus) — `×10⁻³` renders as tofu.
5. **No journal logos, no "Graphical Abstract" heading inside the image.**

## Structure (follows Elsevier's own examples)

Their template figure gives the middle column to methodology
("Here is where you showcase your methodology"); all three of their published
examples carry a method or cohort column. So:

```
title bar
METHOD | EVIDENCE | OUTCOME      three columns, filled header bars
CONCLUSION box, full width       (as in their AJKD example, which is a TABLE)
```

## Data — every number is the paper's, verified

| panel | source |
|---|---|
| capacity window | `make_figures.load_series("panasonic")`, test cell, 24-cycle window |
| calibrated intervals | `results/ga_uq_band.npz` ← `checkpoints/quantile_calce_seed42.pt` |
| physics tail | `src/results/phys_figs.npz` ← `make_figures_phys.py` |
| cross-scale exchange | `src/results/per_sp_train.json` (10 seeds × SP300/400/500) vs `src/results/per_sp_ablation_panasonic_multi_wide.json` (same protocol, parameter-matched) |
| six-dataset ratios | `tab:lit_all` in `paper/sections/04_experiments.tex`, re-parsed by `D:/Temp/verify_ga_ratios.py` |
| memory vs context | `tab:deploy` formulas: scores `4L²·4 B`, KV `6·2L·64·4 B`, state flat 48 KB |

The exchange numbers reproduce the paper's `tab:ablation` exactly:
SP300 `0.0040±0.0004 → 0.0037±0.0003` (−6.6 %), SP400 `0.0038 → 0.0036`
(−6.3 %), SP500 `0.0042 → 0.0040` (−4.4 %); **22 of 30 paired runs improve**,
mean −5.7 %, p = 0.014. Re-derive rather than trusting this table.

## Traps that cost this session hours — do not re-discover them

1. **`fig.text()` takes FIGURE FRACTIONS, not data coordinates.** Passing
   `fig.text(8800, 48, ...)` for a point on a log axis draws it ~10⁴ canvas
   widths off the page. On matplotlib 3.11 this ends in a font-transform
   overflow crash; it can also fail silently.
2. **matplotlib draws tick labels and axis labels OUTSIDE the axes rect.**
   Sizing a chart by its axes rect alone silently overruns the row beneath it.
   Every chart needs `height + tick_band` of vertical space and a left gutter
   for the tick numbers.
3. **`make_figures.py` sets `savefig.bbox='tight'` and `axes.grid=True` at
   import**, and matplotlib MERGES rcParams dicts. Import it *before* your own
   `plt.rcParams.update`, or your settings get overwritten and (a) the canvas
   comes out at the wrong aspect ratio, (b) gridlines draw a stroked line
   through every label.
4. **At this size a 5 pt text row is as tall as a bar in a 6-row bar chart**,
   so no vertical gap exists for a reference line to hide in. Put values into
   the y tick labels instead of on the bars, or the 1.00 line crosses them.
5. `fig.patches` z-order ties with axes: a background rectangle at `zorder=0`
   paints OVER the plot. Use a negative zorder.

## What is still wrong in v5

- **Chart tick rows overlap the charts above them.** The `64 / 400 / 800` row
  sits on the lower edge of the intervals chart and `792 / 840 / 880` on the
  physics chart. The per-item vertical budget is
  `title + chart + tick_band + strip`; `TICK_BAND = 0.036` is too small for a
  4.5 pt row including its pad.
- **The `memory (KB)` rotated label collides with the EVIDENCE column's strip.**
  The OUTCOME column starts at x=0.752 and the label is drawn left of the tick
  numbers, i.e. outside its own column.
- **`runs in C on a Cortex-M3`** is plain text, not a strip like the other four
  result statements.
- **The six-dataset bars lost their value labels** when the axis titles were
  removed.
- `scripts/audit_figure_collisions.py` reports 23 FAIL. A large share are
  artifacts — it merges a *row* of consecutive tick labels into one bbox (a
  single `"0"` is reported 54 pt wide) and then reports that row as overlapping
  its neighbours. Verify each one against the render before chasing it; do not
  optimise the counter blindly.

## What I would do next

Cut, do not cram. At 500×200 the reader takes in **shapes and about five
numbers**. Candidates, in order of what I would drop first:

1. The three `SP300/400/500` table rows → one line: `−5.7 % mean, 22/30 pairs
   improve, p = 0.014`.
2. The `patch 2 / patch 4 / patch 8` chips → one line `patch 2/4/8`.
3. The `GDN-2 ×2` row → fold into the patch line.
4. Then give the freed height to the three evidence charts, which are the part
   a reader actually looks at.

Writing more text into this canvas is what produced every overlap in this
session.
