"""Remove the duplicated (older) Limitations subsection."""
P = "../paper/sections/04_experiments.tex"
s = open(P, encoding="utf-8").read()

marker = "\\subsection{Limitations}"
i1 = s.index(marker)
i2 = s.index(marker, i1 + 1)
# find where the second block ends: next \subsection or \section after i2
import re
m = re.search(r"\\sub?section\{", s[i2 + len(marker):])
end = i2 + len(marker) + m.start() if m else len(s)
s = s[:i2] + s[end:]
open(P, "w", encoding="utf-8").write(s)
print("removed second Limitations block; occurrences now:",
      s.count(marker))
