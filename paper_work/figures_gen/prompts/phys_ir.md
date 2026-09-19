Create a Journal of Power Sources / NeurIPS-style module diagram for a physics-informed extension, landscape 16:9, soft literature-science colors, minimal academic layout. NO title, NO caption, NO watermark.

REALISM (critical): render like a researcher's manually-drawn figure in PowerPoint/vector software — precise thin black lines, flat fills, consistent plain sans-serif typography, generous whitespace. NO generative-AI cartoon aesthetics: no glossy shine, no 3D bevels, no chunky oversized rounded corners, no whimsical decorative shapes, no emoji-like icons, no candy-soft pastel exaggeration. Modest realistic shadows only.

Layout: top-to-bottom method pipeline with thin arrows.

1. Top module: "hidden state h (GDN last token)".
2. Two parallel term modules below, connected by thin arrows and joined by a plus symbol:
   a) "softplus(w · h)" with sub-label "learned rate component";
   b) "softplus(γ) · IR_last" with sub-label "physics feature term" and a small coral annotation chip "γ ≥ 0 : IR only accelerates decay".
3. The two terms merge into a rounded block "degradation rate r ≥ 0".
4. Below: prediction equation card "Q̂ = Q_last − r" with a sage annotation chip "monotonic fade by construction".
5. Bottom inset, dashed-border box labeled "training in absolute capacity space" with small text "L = MAE(Q̂, y)".

Style requirements: soft Nature/Science palette (muted teal, dusty blue, sage green, warm sand, coral accents), white background, precise vector-like arrows, modest shadows only, readable small labels, generous whitespace, no futuristic HUD, no 3D gloss, no watermark.
