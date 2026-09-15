# -*- coding: utf-8 -*-
"""Which numeric results were dropped when §5.5 was condensed to a third?"""
import re, io, sys, subprocess

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

REPO = r"D:\我的材料\资料\论文\南航-谢乃明\mine\RUL_forecast\MultiScaleLinearAttentionNetwork"
HEAD = '\\subsection{Comparison with an attention model at equal context}'

old_all = subprocess.run(['git', 'show', 'HEAD:paper/sections/05_deployment.tex'],
                         cwd=REPO, capture_output=True, text=True,
                         encoding='utf-8').stdout.split('\n')
new_all = open(REPO + r"\paper\sections\05_deployment.tex", encoding='utf-8').read().split('\n')


def slice_between(lines):
    a = next(k for k, l in enumerate(lines) if 'One inference of the full multi-scale' in l)
    b = next(k for k, l in enumerate(lines) if l.strip() == HEAD)
    return '\n'.join(lines[a:b])


def norm(t):
    t = re.sub(r'\\times\s*10\^\{?(-?\d+)\}?', r'e\1', t)   # \times10^{7} -> e7
    t = t.replace('{,}', '')                                 # 48{,}755{,}486
    t = t.replace('\\,', '').replace('\\', '')               # spacing + stray commands
    t = re.sub(r',(?=\d{3}\b)', '', t)                       # thousands commas
    return t


NUM = re.compile(r'(?<![\d.])\d+(?:\.\d+)?(?:e-?\d+)?')


def nums(t):
    return set(m.group().rstrip('.') for m in NUM.finditer(norm(t)))


SO, SN = nums(slice_between(old_all)), nums(slice_between(new_all))
print('原始 §5.5 数字 token:', len(SO))
print('精简 §5.5 数字 token:', len(SN))
print()
print('=== 精简中被删掉的数字 ===')
for m in sorted(SO - SN, key=lambda x: (len(x), x)):
    print('   ', m)
print()
print('=== 精简中新出现/拆出的数字 ===')
for m in sorted(SN - SO, key=lambda x: (len(x), x)):
    print('   ', m)
print()
print('=== 保留下来的数字 ===')
print('   ', ', '.join(sorted(SO & SN, key=lambda x: (len(x), x))))
