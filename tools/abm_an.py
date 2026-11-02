"""abm_an.py: Beam ABM Part B (NL=8, 224-slot cache): GPU s/step and net J/token from N=200 minus N=40 windows."""
import json, glob, os, re, statistics as st
O = "logs/beam/abm"; busy = {}
if os.path.exists(O + "/busy_raw.txt"):
    for l in open(O + "/busy_raw.txt"):
        t, a0, b0, a1, b1 = l.split(); busy[t] = 32 * (int(a1) - int(a0)) / (int(b1) - int(b0))
def g(t):
    d = json.load(open(f"{O}/{t}.json")); ch = {c["fabric"]: c for c in d["breakdown"]["channels"]}; ib = {c["fabric"]: c for c in d["idle_baseline"]["channels"]}
    m = re.search(r"^total ([\d.]+)s\s+gpu ([\d.]+)s.*steps (\d+)", open(f"{O}/{t}.log").read(), re.M)
    return ch["socket"]["joules"], d["breakdown"]["seconds"], ib["socket"]["avg_w"], float(m[2]), int(m[3])
print("abm rep  gpu_ms/step  net_J/tok  gross_J/tok  s/tok  idleW  busy_cores(max)")
R = {}
for a in (0, 1, 2):
    for r in (1, 2):
        lo, hi = f"nl8_abm{a}_n40_r{r}", f"nl8_abm{a}_n200_r{r}"
        if not all(os.path.exists(f"{O}/{x}.json") for x in (lo, hi)): continue
        (j0, t0, w0, g0, s0), (j1, t1, w1, g1, s1) = g(lo), g(hi); n = s1 - s0; w = (w0 + w1) / 2
        v = dict(gpu=1000 * (g1 - g0) / n, net=(j1 - j0 - w * (t1 - t0)) / n, gross=(j1 - j0) / n, s=(t1 - t0) / n, w=w, b=max(busy.get(lo, -1), busy.get(hi, -1)))
        R.setdefault(a, []).append(v)
        print(f"{a:3d} {r:3d} {v['gpu']:11.2f} {v['net']:10.2f} {v['gross']:11.2f} {v['s']:6.3f} {w:6.1f} {v['b']:6.2f}")
for a, v in R.items():
    print(f"ABM{a} median: gpu {st.median(x['gpu'] for x in v):.2f} ms/step  net {st.median(x['net'] for x in v):.2f} J/tok  (n={len(v)})")
