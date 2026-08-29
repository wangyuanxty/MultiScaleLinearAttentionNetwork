"""Move fig:appl to the first figure position (right after the
application-domains paragraph) and de-name DeltaCycle in its caption."""
P = "../paper/sections/01_intro.tex"
s = open(P, encoding="utf-8").read()

i = s.index("\\caption{Application outlook as an operational pipeline")
b = s.rindex("\\begin{figure}", 0, i)
e = s.index("\\end{figure}", i) + len("\\end{figure}")
block = s[b:e]

block = block.replace(
    "needs stress a specific\nDeltaCycle capability",
    "needs stress a specific\ncapability of the proposed system",
)

s = s[:b] + s[e:]

anchor = "least\nsupervised---monitoring problem of all."
j = s.index(anchor) + len(anchor)
s = s[:j] + "\n\n" + block + s[j:]

open(P, "w", encoding="utf-8").write(s)
print("fig_appl moved to first figure position")
