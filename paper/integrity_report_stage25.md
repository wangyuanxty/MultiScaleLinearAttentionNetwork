# Academic Integrity Verification Report — Stage 2.5 (Pre-Review)

- **Verification Mode**: Initial Verification (Stage 2.5, pre-review)
- **Paper**: DeltaCycle (paper/main.tex, 35 pages, elsarticle)
- **Date**: 2026-08-16
- **Verifier**: integrity_verification_agent (ARS v3.11.1)

## Verdict: **PASS WITH NOTES** (after corrections applied — all SERIOUS/MEDIUM issues fixed in-round)

## Verification Summary

| Category | Total | Passed | Issues found → fixed |
|----------|-------|--------|----------------------|
| Reference Existence (Phase A0-A1) | 12 fresh-verified / 48 | 12 | 1 SERIOUS (ref_severson mashup) |
| Bibliographic Accuracy (A2) | 12 | 11 | 1 MEDIUM (ref_ekf_yan author name) |
| Ghost Citations (A3) | 48 cited / 48 bib | 48 | 0 dangling / 0 orphan |
| Citation Context (B1, spot-check) | 6 spot-checked | 6 | 0 |
| Statistical Data (C1, lit tables vs sources) | ~130 rows | 129 | 1 MINOR (TJU MambaSimple R² 0.9875→0.9874) |
| Statistical Data (C1, our numbers vs results) | Table A / phys tables | all | 0 (verified against results/table_a_3seed.json, phys_figs.npz) |
| Internal Consistency (C2) | — | — | 3 MEDIUM (MIT cell counts 43/7*/8-2 → 124/10*/8-1, all fixed) |
| Originality (D1, spot-check) | 1 paragraph | 1 ORIGINAL | 0 (one MINOR claim softening applied) |
| Claim Verification (E, spot-check) | ~15 claims | 15 | 0 |

## Issue List (all fixed in-round)

### SERIOUS — fixed
| # | Category | Location | Issue | Correct | Source |
|---|----------|----------|-------|---------|--------|
| 1 | Reference | refs.bib ref_severson | Journal/vol/pages/DOI were a mashup: "Science 360(6385):658–661, 10.1126/science.aat7631" — does not correspond to the cited title | Nature Energy, 4, 383–391, DOI 10.1038/s41560-019-0356-8; author list + Richard D. Braatz, "Meng H. Chen" → "Michael H. Chen" | WebSearch (Scopus/Springer) |

### MEDIUM — fixed
| # | Category | Location | Issue | Correct | Source |
|---|----------|----------|-------|---------|--------|
| 2 | Reference | refs.bib ref_ekf_yan | First author given name wrong: "Yan, Wensheng" | "Yan, Wuzhao" | IEEE Xplore / DBLP |
| 3 | Internal consistency | sec:datasets + tab:datasets + MIT case study | MIT cell counts disagreed: "43 cells" (¶), "7-cell subset" (¶), "7*" (table), "8 train/2 test" (case study) vs pipeline evaluating only MIT_TEST_CELLS[0] | 124 cells (source dataset), 10-cell subset; **both test cells now evaluated** (eval_mit_2test.py, no retraining) and Table A/case-study/abstract updated to 2-test aggregates | load_datasets.py + eval run |

### MINOR — fixed
| # | Category | Location | Issue | Fix |
|---|----------|----------|-------|-----|
| 4 | Statistical data | tab:lit_tju MambaSimple SP400 | R² 0.9875 vs source 0.9874 | corrected to 0.9874 |
| 5 | Claim (E) | 02_related.tex | "Recurrent models were the *first* deep approach" — priority claim not explicitly supported by sources | softened to "among the first" |

## Phase A Coverage Note (documented deviation)

12 of 48 references were freshly re-verified via WebSearch/arXiv this round (the data-transcribing baselines ref_patchformer/ref_rulmamba/ref_omnitie, core methods ref_gdn/ref_gdn2/ref_piddm/ref_mambalithium, and high-risk entries ref_severson/ref_zhou2025/ref_wang_pinn/ref_ekf_yan/ref_grey_hybrid — all VERIFIED except the two fixed above). The remaining 36 rely on the prior Crossref/arXiv audit recorded in CLAUDE.md. The 12/48 fresh rate already surfaced 2 real errors (16% hit rate), so **Stage 4.5 (final check) should run Phase A at 100% fresh** per protocol.

## AI Research Failure Mode Checklist (7 modes)

| Mode | Verdict | Evidence |
|------|---------|----------|
| 1 Citation hallucination | FIXED — no current instances | 2 historical instances found & corrected this round |
| 2 Implementation bugs | CLEARED | Seed-44 determinism check, phys numbers reproduced by independent rerun (0.3743 exact) |
| 3 Hallucinated results | CLEARED | Every table number traced to results/*.json, checkpoints, or source-paper md |
| 4 Shortcut reliance | NOTED | phys section seed-42-only, MIT subset — stated as limitations in paper/ledger |
| 5 Bug-as-insight | CLEARED | z-score/monotone incompatibility has direct-abs control (0.0097); flat-tail AE mechanism verified numerically |
| 6 Methodology fabrication | CLEARED | Protocols line-verified against PatchFormer/RUL-Mamba preprocessing code; MCU claims scoped to actual QEMU runs |
| 7 Pipeline frame-lock | CLEARED | Honest per-dataset claims (CALCE behind PatchFormer admitted); tested-and-rejected physics routes reported |

## Audit Trail (abridged)

- `"PatchFormer" DOI 10.1016/j.jpowsour.2025.236187` → ScienceDirect S0378775325000230 — VERIFIED
- `"RUL-Mamba" DOI 10.1016/j.est.2025.116376` → ScienceDirect S2352152X25010898 — VERIFIED
- `"OmniTIEFormer" DOI 10.1016/j.apenergy.2026.127858` → ScienceDirect S0306261926005106 — VERIFIED
- `arXiv:2412.06464` (Gated Delta Networks) → ICLR 2025 proceedings — VERIFIED
- `arXiv:2605.22791` (Gated DeltaNet-2) → arXiv abstract API — VERIFIED
- `arXiv:2607.29095` (PiDDM) → arXiv abstract API — VERIFIED
- `arXiv:2403.05430` (MambaLithium) → arXiv abstract API — VERIFIED
- `"Severson" 10.1126/science.aat7631` → MISMATCH, corrected to Nature Energy 10.1038/s41560-019-0356-8
- `"Lebesgue-sampling EKF" Yan` → IEEE Xplore 8375148 — VERIFIED, author name corrected
- `"Physics-informed neural network" Wang NatComms 4332` → DOI 10.1038/s41467-024-48779-z — VERIFIED
- `"hybrid grey" ESWA 126905` → DOI 10.1016/j.eswa.2025.126905 — VERIFIED
- `"Early prediction ... cycle-consistency" JES 118147` → DOI 10.1016/j.est.2025.118147 — VERIFIED

## Tool Limitation Disclaimer

> Phase D used WebSearch heuristic comparison, not professional plagiarism software (Turnitin/iThenticate). Coverage limited to publicly searchable literature, sampling rate ~1 paragraph this round. Professional duplicate checking recommended before formal submission.
