# Pipeline Status Dashboard

```
+=========================================+
|   Academic Pipeline Status              |
+=========================================+
| Topic: DeltaCycle — Multi-Scale Linear  |
|        Attention for Battery Prognostics|
+-----------------------------------------+

  Stage 1 RESEARCH    [✅] complete
    deep-research: literature library + 48 verified refs
    outputs: paper_plan.md, refs.bib

  Stage 2 WRITE       [🔄] in progress (major refactor)
    mode: full
    draft: paper/sections/*.tex (5 sections, 32pp PDF)
    refactor-in-flight: per-window normalization, stage-query
    attention exchange, K=32 removal, MIT 8/2 subset,
    physics ablation redesign

  Stage 3 REVIEW      [✅] complete (round 1)
    reviewer_report.md: C1-C4, H1-H7, M1-M7, L1-L5
    C2/C3/C4 resolved; C1 (multi-seed) resolved

  Stage 4 REVISE      [🔄] in progress
    revision_round: 1
    addressed: C1(3-seed stats), C2(comparison table),
      C3(horizon analysis), C4(SP700), H1/H2/H5/H7,
      M1/M3, normalization unification, AE definition
    in-flight: per-window migration, physics ablation redesign

  Stage 3' RE-REVIEW  [⏳] pending
    loop_count: 0

  Stage 5 FINALIZE    [⏳] pending
    format: elsarticle (LaTeX), J. Power Sources

+-----------------------------------------+
| Materials:                              |
|   [✅] RQ Brief  -> ars_rq_brief.md     |
|   [⏳] Methodology Blueprint -> ars_methodology_blueprint.md |
|   [✅] Bibliography -> refs.bib (48)    |
|   [⏳] Synthesis Report -> ars_synthesis_report.md |
|   [✅] Paper Draft -> sections/*.tex    |
|   [✅] Review Reports -> reviewer_report.md |
|   [⏳] Revision Roadmap -> ars_revision_roadmap.md |
|   [🔄] Revised Draft -> sections/*.tex (refactor) |
|   [⏳] Response to Reviewers -> ars_response_reviewers.md |
|   [⏳] Final Paper                     |
+-----------------------------------------+
| Revision History:                       |
|   2026-08-10: v0 draft compiled (19pp)  |
|   2026-08-11/12: 3-seed stats, comparison table, MIT full->subset |
|   2026-08-13: major refactor decision (per-window, stage-query, K=32 out) |
+-----------------------------------------+
| Next Step: complete physics ablation redesign experiments, |
|            then re-run full training under new pipeline,     |
|            update draft, re-review                           |
+=========================================+
```

## 当前进行中的实验(2026-08-13)

| 实验 | 目的 | 状态 |
|---|---|---|
| 外推消融(前 90%/后 10%,物理 on/off) | I3 价值验证 | 🔄 运行中(1 seed) |
| 噪声鲁棒性(3 损坏 × 物理 on/off) | I3 价值验证 | 🔄 运行中(1 seed) |
| per-window 崩坏 K=32 验证 | 归一化决策依据 | ✅ 完成(AE 15.9→25.3) |
| single/multi/V3 对比 | 交互机制验证 | ✅ 完成(V3≈标量门>multi) |
