exec(open("sim.py").read().split("def run(")[0])
import collections
def belady(K, warm=0.5):
    nw = int(len(S) * warm); nxt = {}; fut = [None] * len(S); last = {}
    for i in range(len(S) - 1, -1, -1):
        l, ex = S[i]; fut[i] = {e: last.get((l, e), 10**9) for e in ex}
        for e in ex: last[(l, e)] = i
    c = collections.defaultdict(dict); hit = tot = 0
    for i, (l, ex) in enumerate(S):
        cl = c[l]; h = 0
        for e in ex:
            if e in cl: h += 1
            cl[e] = fut[i][e]
        while len(cl) > K: v = max((x for x in cl if x not in ex), key=lambda x: cl[x]); del cl[v]
        if i >= nw: hit += h; tot += len(ex)
    return hit / tot
for K in (32, 64, 96, 128): print(K, "belady %.3f" % belady(K))
