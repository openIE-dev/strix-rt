import re, glob, collections, sys
# stream: list of (layer, [experts]) in execution order across prompts (set B order then A), each prompt is a fresh sequence
order = "paris_raw paris_chat math_chat code_py code_chat prose science_chat history german sql logic_chat json".split()
S = []
for n in order:
    try: t = open(f"logs/beam/prof/{n}.log").read()
    except FileNotFoundError: continue
    S += [(int(l), list(map(int, e.split(",")))) for l, e in re.findall(r"route (\d+) \[([\d, ]+)\]", t)]
L = sorted({l for l, _ in S}); print("accesses", len(S), "layers", len(L), "tokens~", len(S) // len(L))
def run(policy, K, warm_frac=0.5):
    nw = int(len(S) * warm_frac); hit = tot = 0
    st = policy(K)
    for i, (l, ex) in enumerate(S):
        h = st.access(l, ex)
        if i >= nw: hit += h; tot += len(ex)
    return hit / tot
class PerLRU:
    def __init__(s, K): s.K = K; s.c = collections.defaultdict(collections.OrderedDict)
    def access(s, l, ex):
        c = s.c[l]; h = 0
        for e in ex:
            if e in c: c.move_to_end(e); h += 1
            else:
                c[e] = 1
                if len(c) > s.K: c.popitem(last=False)
        return h
class GlobLRU:
    def __init__(s, K): s.K = K * 49; s.c = collections.OrderedDict()
    def access(s, l, ex):
        h = 0
        for e in ex:
            k = (l, e)
            if k in s.c: s.c.move_to_end(k); h += 1
            else:
                s.c[k] = 1
                if len(s.c) > s.K: s.c.popitem(last=False)
        return h
class DecayLFU:  # per-layer: evict min decayed count (counts decayed x d every 4 accesses of the layer), Edge0 hot_counts style
    def __init__(s, K, d=0.75, every=4): s.K = K; s.d = d; s.ev = every; s.cnt = collections.defaultdict(lambda: collections.defaultdict(float)); s.c = collections.defaultdict(set); s.n = collections.Counter()
    def access(s, l, ex):
        s.n[l] += 1; cn = s.cnt[l]; c = s.c[l]; h = 0
        if s.n[l] % s.ev == 0:
            for k in list(cn): cn[k] *= s.d
        for e in ex: cn[e] += 1
        for e in ex:
            if e in c: h += 1
            else:
                c.add(e)
                if len(c) > s.K:
                    v = min((x for x in c if x not in ex), key=lambda x: cn[x]); c.remove(v)
        return h
class Hybrid:  # per-layer LRU with frequency guard: evict the LRU-oldest among the lowest-quartile counts (no decay) -> approx LRFU
    def __init__(s, K, lam=0.9): s.K = K; s.lam = lam; s.sc = collections.defaultdict(dict); s.t = collections.Counter()
    def access(s, l, ex):
        s.t[l] += 1; t = s.t[l]; c = s.sc[l]; h = 0
        for e in ex:
            if e in c: h += 1; v, tl = c[e]; c[e] = (v * s.lam ** (t - tl) + 1, t)
            else:
                c[e] = (1.0, t)
                if len(c) > s.K:
                    v = min((x for x in c if x not in ex), key=lambda x: c[x][0] * s.lam ** (t - c[x][1])); del c[v]
        return h
class GlobLRFU(Hybrid):
    def __init__(s, K, lam=0.9): super().__init__(K * 49, lam); s.T = 0; s.g = {}
    def access(s, l, ex):
        s.T += 1; t = s.T; c = s.g; h = 0
        for e in ex:
            k = (l, e)
            if k in c: h += 1; v, tl = c[k]; c[k] = (v * s.lam ** ((t - tl) / 49) + 1, t)
            else:
                c[k] = (1.0, t)
                if len(c) > s.K:
                    v = min((x for x in c if not (x[0] == l and x[1] in ex)), key=lambda x: c[x][0] * s.lam ** ((t - c[x][1]) / 49)); del c[v]
        return h
for K in (32, 64, 96, 128):
    r = [("perLRU", run(PerLRU, K)), ("globLRU", run(GlobLRU, K)), ("decayLFU", run(DecayLFU, K)),
         ("LRFU.9", run(lambda k: Hybrid(k, .9), K)), ("LRFU.97", run(lambda k: Hybrid(k, .97), K))]
    if K in (64, 128): r.append(("globLRFU.97", run(lambda k: GlobLRFU(k, .97), K)))
    print(K, "  ".join(f"{n} {v:.3f}" for n, v in r), flush=True)
