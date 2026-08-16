"""One-off: move tab:tableA before the case studies (sec 4.3 restructure)."""
P = "../paper/sections/04_experiments.tex"
lines = open(P, encoding="utf-8").read().split("\n")

cap = next(i for i, l in enumerate(lines) if "Per-SP trajectory results" in l)
b = max(i for i in range(cap) if "\\begin{table}" in lines[i])
e = next(i for i in range(cap, len(lines)) if "\\end{table}" in lines[i])
block = "\n".join(lines[b:e + 1])
del lines[b:e + 1]

a = next(i for i, l in enumerate(lines) if "\\label{fig:traj}" in l)
a = next(i for i in range(a, len(lines)) if "\\end{figure}" in lines[i])
lines[a + 1:a + 1] = ["", block]

open(P, "w", encoding="utf-8").write("\n".join(lines))
j = next(i for i, l in enumerate(lines) if "Per-SP trajectory results" in l)
print("moved; caption now at line", j + 1)
