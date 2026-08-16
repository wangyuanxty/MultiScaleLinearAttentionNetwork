"""Stage 4 edit: baseline-positioning paragraph + CQR decision-rule paragraph."""
P = "../paper/sections/04_experiments.tex"

s = open(P, encoding="utf-8").read()

anchor7 = "The per-dataset comparison tables are distributed"
if anchor7 not in s:
    print("anchor7 missing")
else:
    insert7 = (
        "Battery-specific foundation and pretrained models published outside\n"
        "the per-SP RUL protocol family are not included in the comparison\n"
        "tables, because their reported metrics use different evaluation\n"
        "conventions (early-cycle prediction, or trajectory regression\n"
        "without per-SP RUL errors); head-to-head comparisons under our\n"
        "protocol are left to future work.\n\n"
    )
    s = s.replace(anchor7, insert7 + anchor7, 1)
    print("insert7 ok")

anchor8 = "Figure~\\ref{fig:uq_band} shows the calibrated band on the test cell."
if anchor8 not in s:
    print("anchor8 missing")
else:
    insert8 = anchor8 + """

\\emph{From intervals to a decision rule.} The calibrated band
translates into an operational policy directly: the simplest
risk-aware rule is ``schedule replacement at the first cycle where
the P2.5 trajectory crosses the EOL threshold.'' With CQR the
$93.3\\%$ coverage bounds the missed-EOL rate, and the $+36\\%$
interval-width cost quantified above is the price paid for that
bound. Quantifying the associated maintenance-cost trade-off under a
concrete cost model is left to future work."""
    s = s.replace(anchor8, insert8, 1)
    print("insert8 ok")

old_lim = "(full 124-cell MIT set, additional seeds, a\nsecond calibration cell) are left to the final version."
new_lim = "(full 124-cell MIT set, additional seeds, a\nsecond calibration cell, and a quantitative replacement-policy cost\nexample) are left to the final version."
if old_lim in s:
    s = s.replace(old_lim, new_lim, 1)
    print("limitations ok")
else:
    print("limitations anchor missing")

open(P, "w", encoding="utf-8").write(s)
print("done")
