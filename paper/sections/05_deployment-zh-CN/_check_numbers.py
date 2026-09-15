# -*- coding: utf-8 -*-
"""Compare numeric tokens between the LaTeX source of §5.5 and its Chinese translation.

Both sides are normalised to the canonical form  <mantissa>e<exponent>  before
extraction, so that  4.88\\times10^{7}  (LaTeX)  and  4.88x10^7  (unicode
superscript)  both collapse to  4.88e7 .
"""
import re, io, sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE = r"D:\我的材料\资料\论文\南航-谢乃明\mine\RUL_forecast\MultiScaleLinearAttentionNetwork\paper"

src = '\n'.join(open(BASE + r"\sections\05_deployment.tex", encoding='utf-8')
                .read().split('\n')[183:262])
tr = open(BASE + r"\sections\05_deployment-zh-CN\translation.md", encoding='utf-8').read()

SUP = {'⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4', '⁵': '5',
       '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9', '⁻': '-', '⁺': '+'}


def unsup(t):
    """Turn runs of unicode superscripts into e<exp>, skipping the base '10'."""
    out, i = [], 0
    while i < len(t):
        if t[i] in SUP:
            j, b = i, ''
            while j < len(t) and t[j] in SUP:
                b += SUP[t[j]]
                j += 1
            out.append('e' + b)
            i = j
        else:
            out.append(t[i])
            i += 1
    return ''.join(out)


def norm(t):
    t = unsup(t)
    # LaTeX:  {\times}10^{7} / \times 10^{-12}
    t = re.sub(r'\\times\s*10\^\{?(-?\d+)\}?', r'e\1', t)
    t = t.replace('{,}', '')                      # 48{,}755{,}486
    # unicode:  4.88x10e7  ->  4.88e7   (also  x, X, *)
    t = re.sub(r'([\d.]+)\s*[x×*]\s*10e(-?\d+)', r'\1e\2', t)
    # any leftover bare power-of-ten exponential (no mantissa) -> sentinel, not a number
    t = re.sub(r'(?<![\d.])10e-?\d+', 'POWTEN', t)
    t = t.replace('\\,', '').replace('\\', '')
    t = re.sub(r',(?=\d{3}\b)', '', t)            # thousands separators
    return t


NUM = re.compile(r'(?<![\d.])\d+(?:\.\d+)?(?:e-?\d+)?')


def nums(t):
    return set(m.group().rstrip('.') for m in NUM.finditer(norm(t)))


S, T = nums(src), nums(tr)
print('source numeric tokens     :', len(S))
print('translation numeric tokens:', len(T))
print()
print('=== 源有译无 ===')
for m in sorted(S - T, key=lambda x: (len(x), x)):
    print('   ', m)
print()
print('=== 译有源无 ===')
for e in sorted(T - S, key=lambda x: (len(x), x)):
    print('   ', e)
print()
print('=== 命中的共同数字（前 40）===')
print('   ', ', '.join(sorted(S & T, key=lambda x: (len(x), x))[:40]))
