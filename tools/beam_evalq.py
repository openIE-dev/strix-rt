#!/usr/bin/env python3
# teacher-forced quality drift: baseline NVFP4 vs q2/q3 on logs/beam/seq.tsv; reads logs/beam/evalq/{name}.{base,q2,q3}.log
import math, sys, os
def load(p):
    d = {}
    if not os.path.exists(p): return d
    for l in open(p):
        if l.startswith("F "):
            f = l.split(); d[int(f[1])] = {int(a.split(":")[0]): float(a.split(":")[1]) for a in f[2:]}
    return d
O = "logs/beam/evalq"; names = os.environ.get("NAMES", "").split() or [l.split("\t")[0] for l in open("logs/beam/seq.tsv")]
for q in sys.argv[1:] or ["q2", "q3"]:
    print(f"== {q} vs nvfp4 (teacher-forced on baseline greedy continuation)")
    print(f"{'prompt':12} {'n':>3} {'top1':>6} {'dlogp(chosen)':>14} {'KL(top20)':>10} {'p(chosen)b':>10} {'p(chosen)q':>10}")
    T = [0, 0, 0.0, 0.0, 0.0, 0.0]
    for n in names:
        b = load(f"{O}/{n}.base.log"); m = load(f"{O}/{n}.{q}.log")
        s = [l for l in open("logs/beam/seq.tsv") if l.startswith(n + "\t")][0].split("\t"); ids = [int(x) for x in s[1].split(",")]
        c = [0, 0, 0.0, 0.0, 0.0, 0.0]
        for pos in sorted(b):
            if pos not in m: continue
            tb, tm = b[pos], m[pos]; tok = ids[pos + 1]; floor_m = min(tm.values()) - 0.5
            c[0] += 1; c[1] += max(tb, key=tb.get) == max(tm, key=tm.get)
            lb = tb.get(tok, min(tb.values()) - 0.5); lm = tm.get(tok, floor_m)
            c[2] += lm - lb; c[4] += math.exp(lb); c[5] += math.exp(lm)
            Z = sum(math.exp(v) for v in tb.values())
            c[3] += sum(math.exp(v) / Z * (v - math.log(Z) - (tm.get(k, floor_m) - math.log(sum(math.exp(tm.get(j, floor_m)) for j in tb)))) for k, v in tb.items())
        if c[0]:
            print(f"{n:12} {c[0]:3} {c[1]/c[0]:6.3f} {c[2]/c[0]:14.3f} {c[3]/c[0]:10.3f} {c[4]/c[0]:10.3f} {c[5]/c[0]:10.3f}")
            T = [T[i] + c[i] for i in range(6)]
    if T[0]: print(f"{'ALL':12} {T[0]:3} {T[1]/T[0]:6.3f} {T[2]/T[0]:14.3f} {T[3]/T[0]:10.3f} {T[4]/T[0]:10.3f} {T[5]/T[0]:10.3f}")
