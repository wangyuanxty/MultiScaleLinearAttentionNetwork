"""Ghost citation check for the paper (Stage 2.5 Phase A3)."""
import re
import glob

cited = set()
for f in ["main.tex"] + glob.glob("sections/*.tex"):
    s = open(f, encoding="utf-8").read()
    for m in re.findall(r"\\cite\{([^}]*)\}", s):
        for k in m.split(","):
            cited.add(k.strip())

bib = set(re.findall(r"@\w+\{([^,]+),", open("refs.bib", encoding="utf-8").read()))
dangling = sorted(cited - bib)
orphan = sorted(bib - cited)
print("cited:", len(cited), "bib entries:", len(bib))
print("DANGLING (cited but not in bib):", dangling if dangling else "NONE")
print("ORPHAN (in bib but never cited):", orphan if orphan else "NONE")
