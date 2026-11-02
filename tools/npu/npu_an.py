"""npu_an.py -> net/gross J per decision and per decoded token from logs/npu/e (socket channel, differential hi-lo)."""
import json, glob, os, sys, collections, statistics as st
D = sys.argv[1] if len(sys.argv) > 1 else "logs/npu"; R = collections.defaultdict(list); busy = {}
if os.path.exists(D + "/busy.txt"):
    for l in open(D + "/busy.txt"): t, v = l.split(); busy[t] = float(v)
def g(p):
    d = json.load(open(p)); ch = {c["fabric"]: c for c in d["breakdown"]["channels"]}; ib = {c["fabric"]: c for c in d["idle_baseline"]["channels"]}
    return ch["socket"]["joules"], d["breakdown"]["seconds"], ib["socket"]["avg_w"], {k: v["joules"] for k, v in ch.items()}
for lo in sorted(glob.glob(D + "/e/*.lo.json")):
    b = lo[:-8]; hi = b + ".hi.json"
    if not all(os.path.exists(x) for x in (hi, b + ".lo.meta", b + ".hi.meta")): continue
    (j0, t0, w0, c0), (j1, t1, w1, c1) = g(lo), g(hi); n = int(open(b + ".hi.meta").read()) - int(open(b + ".lo.meta").read())
    tag = os.path.basename(b); key = tag.rsplit("_r", 1)[0]; w = (w0 + w1) / 2
    R[key].append(dict(gross=(j1 - j0) / n, net=(j1 - j0 - w * (t1 - t0)) / n, s=(t1 - t0) / n, idle=w,
                       npu=(c1.get("npu", 0) - c0.get("npu", 0)) / n, gfx=(c1.get("gfx", 0) - c0.get("gfx", 0)) / n,
                       busy=max(busy.get(tag + ".lo", 0), busy.get(tag + ".hi", 0))))
print(f"{'config':44s} reps  gross J  net J  s/unit  npuJ   gfxJ  idleW busy")
for k in sorted(R):
    v = R[k]; m = lambda f: st.median(x[f] for x in v)
    print(f"{k:44s} {len(v):4d} {m('gross'):8.3f} {m('net'):6.3f} {m('s'):6.3f} {m('npu'):6.3f} {m('gfx'):6.3f} {m('idle'):5.1f} {max(x['busy'] for x in v):4.2f}")
for f in sorted(glob.glob(D + "/acc_*.jsonl")):
    rs = [json.loads(l) for l in open(f)]; print(os.path.basename(f), "acc %.3f n=%d" % (sum(r["pred"] == r["gold"] for r in rs) / len(rs), len(rs)))
