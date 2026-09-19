Create a Journal of Power Sources / NeurIPS-style mechanism diagram of ONE GDN-2 linear-attention block, landscape 16:9, soft literature-science colors, minimal academic layout. NO title, NO caption, NO watermark.

REALISM (critical): render like a researcher's manually-drawn figure in PowerPoint/vector software — precise thin black lines, flat fills, consistent plain sans-serif typography, generous whitespace. NO generative-AI cartoon aesthetics: no glossy shine, no 3D bevels, no chunky oversized rounded corners, no whimsical decorative shapes, no emoji-like icons, no candy-soft pastel exaggeration. Modest realistic shadows only.

Layout: top-to-bottom data flow with thin curved residual arrows.

1. Top: input token module "x_t".
2. Four small projection modules in a row labeled "k_t", "w_t", "v_t", "q_t" (keys, write values, values, query), with thin arrows from x_t to each.
3. Central pale-teal block "state S_t" with the recurrence equation written beside it EXACTLY as:
   S_t = (I − k_t(b_t ⊙ k_t)ᵀ) diag(α_t) S_{t−1} + k_t(w_t ⊙ v_t)ᵀ
   CRITICAL: the element-wise multiplication symbol must be a circle WITH A DOT inside (⊙), never a plain circle.
   A curved arrow loops from S_t back to itself labeled "recurrence (fixed size)".
4. A small module "diag(α_t)" labeled "per-channel decay" with an arrow into the state block.
5. Bottom: output equation "o_t = S_tᵀ q_t" with an arrow to an output module "o_t".
6. Side annotation chip: "fixed 8 KB state, H heads".

Style requirements: soft Nature/Science palette (muted teal, dusty blue, sage green, warm sand, coral accents), white background, precise vector-like arrows, modest shadows only, readable small labels, generous whitespace, no futuristic HUD, no 3D gloss, no watermark.
