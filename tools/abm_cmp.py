#!/usr/bin/env python3
"""Compare forced-decode top-20 logprobs of BEAM_ABM variants vs ABM=0 (MAC). Usage: abm_cmp.py DIR"""
import sys, glob, os
D = sys.argv[1]
def rd(f):
    r = {}
    for line in open(f):
        if line.startswith("F "):
            p = line.split(); r[int(p[1])] = [(int(a), float(b)) for a, b in (x.split(":") for x in p[2:])]
    return r
for p in sorted({os.path.basename(f).split(".abm")[0] for f in glob.glob(D + "/*.abm0.log")}):
    ref = rd(f"{D}/{p}.abm0.log")
    for a in (1, 2):
        o = rd(f"{D}/{p}.abm{a}.log"); n = t1 = same = 0; md = 0.0; bit = 0
        for k, v in ref.items():
            w = o.get(k)
            if not w: continue
            n += 1; t1 += v[0][0] == w[0][0]; same += [x for x, _ in v] == [x for x, _ in w]
            dw = dict(w); md = max([md] + [abs(lp - dw[t]) for t, lp in v if t in dw]); bit += v == w
        print(f"{p} ABM{a} vs MAC: positions {n} top1 agree {t1}/{n} top20 set+order identical {same}/{n} bit-identical rows {bit}/{n} max|dlogprob| {md:.4f}")
