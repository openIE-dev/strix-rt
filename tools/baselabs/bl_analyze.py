"""Analyze Base Labs J/decision and floor windows. Usage: bl_analyze.py"""
import json, os, glob, statistics as st
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs")
def g(p):
    d = json.load(open(p)); ch = {c["fabric"]: c for c in d["breakdown"]["channels"]}; ib = {c["fabric"]: c for c in d["idle_baseline"]["channels"]}
    return ch["socket"]["joules"], d["breakdown"]["seconds"], ib["socket"]["avg_w"], ch["gfx"]["joules"]
def diff(lo, hi, dd):
    (j0, t0, w0, g0), (j1, t1, w1, g1) = g(lo), g(hi); w = (w0 + w1) / 2
    return dict(gross=(j1 - j0) / dd, net=(j1 - j0 - w * (t1 - t0)) / dd, gfx=(g1 - g0) / dd, s=(t1 - t0) / dd, w=(j1 - j0) / (t1 - t0), idle=w)
def show(name, rs):
    if not rs: print(name, "no windows"); return
    m = {k: st.median(r[k] for r in rs) for k in rs[0]}
    print(f"{name:10} n={len(rs)} J/dec gross {m['gross']:.2f} [{min(r['gross'] for r in rs):.2f}-{max(r['gross'] for r in rs):.2f}] net {m['net']:.2f} gfx {m['gfx']:.2f} ms/dec {1e3*m['s']:.1f} W {m['w']:.1f} idle W {m['idle']:.1f}")
E = R + "/baselabs/energy"
for n in ("joey", "elephant"):
    rs = []
    for lo in sorted(glob.glob(f"{E}/{n}_r*.lo.json")):
        hi = lo[:-8] + ".hi.json"
        if not os.path.exists(hi): continue
        a, b = (list(map(int, open(p[:-5] + ".meta").read().split())) for p in (lo, hi)); rs.append(diff(lo, hi, b[0] * b[1] - a[0] * a[1]))
    show(n, rs)
F = R + "/cascade/floor"
for k in ("k8", "k1"):
    show(k, [diff(f"{F}/{k}_r{r}.lo.json", f"{F}/{k}_r{r}.hi.json", 8 * 40) for r in (1, 2) if os.path.exists(f"{F}/{k}_r{r}.hi.json")])
for t in ("idle_srv", "idle_none"):
    j, s, w, gj = g(f"{F}/{t}.json"); print(f"{t:10} {j/s:.2f} W socket ({gj/s:.2f} W gfx) over {s:.0f} s; pre-window idle baseline {w:.2f} W")
