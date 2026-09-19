# -*- coding: utf-8 -*-
"""Numeric parity check: §5.1 and §5.2 of the LaTeX source vs their Chinese translation."""
import re, io, sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE = r"D:\我的材料\资料\论文\南航-谢乃明\mine\RUL_forecast\MultiScaleLinearAttentionNetwork\paper"
tex = open(BASE + r"\sections\05_deployment.tex", encoding='utf-8').read().split('\n')
zh = open(BASE + r"\sections\05_deployment-zh-CN\translation.md", encoding='utf-8').read()

SUP = {'⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4', '⁵': '5',
       '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9', '⁻': '-', '⁺': '+'}


def unsup(t):
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
    t = re.sub(r'\\times\s*10\^\{?(-?\d+)\}?', r'e\1', t)   # latex power of ten
    t = re.sub(r'([\d.]+)\s*[x×*]\s*10e(-?\d+)', r'\1e\2', t)  # unicode power of ten
    t = t.replace('{,}', '')
    t = t.replace('\\,', '').replace('\\', '')
    t = re.sub(r',(?=\d{3}\b)', '', t)
    return t


NUM = re.compile(r'(?<![\d.])\d+(?:\.\d+)?(?:e-?\d+)?')


def nums(t):
    return set(m.group().rstrip('.') for m in NUM.finditer(norm(t)))


def tex_slice(start_pat, end_pat):
    a = next(k for k, l in enumerate(tex) if start_pat in l)
    b = next(k for k, l in enumerate(tex) if l.strip() == end_pat)
    return '\n'.join(tex[a:b])


def zh_slice(start, end):
    a = zh.index(start)
    b = zh.index(end)
    return zh[a:b]


cases = [
    ("5.1 Constraints", tex_slice('Constraints and design goals', r'\subsection{Numerical fidelity across architectures}'),
     zh_slice('# 5.1 约束与设计目标', '\n---\n\n# 5.2')),
    ("5.2 Fidelity", tex_slice('Numerical fidelity across architectures', r'\subsection{Quantization}'),
     zh_slice('# 5.2 跨架构数值保真度', '\n---\n\n# 5.5')),
]

for name, s, t in cases:
    S, T = nums(s), nums(t)
    missing = sorted(S - T, key=lambda x: (len(x), x))
    extra = sorted(T - S, key=lambda x: (len(x), x))
    print(f'--- {name} ---')
    print(f'  源 {len(S)} 个 | 译 {len(T)} 个')
    print(f'  源有译无: {missing if missing else "无"}')
    print(f'  译有源无: {extra if extra else "无"}')
    print(f'  共同: {", ".join(sorted(S & T, key=lambda x: (len(x), x)))}')
    print()
