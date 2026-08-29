"""Cross-reference audit: every \\ref must have a matching \\label."""
import re
import glob

refs = set()
labels = set()
for f in ["main.tex"] + glob.glob("sections/*.tex"):
    s = open(f, encoding="utf-8").read()
    refs.update(re.findall(r"\\ref\{([^}]+)\}", s))
    labels.update(re.findall(r"\\label\{([^}]+)\}", s))

dangling = refs - labels
print("labels:", len(labels), "refs:", len(refs))
print("dangling:", sorted(dangling) if dangling else "NONE")
