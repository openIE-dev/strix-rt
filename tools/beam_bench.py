#!/usr/bin/env python3
# summarise logs/beam/bench: tok/s (decode steps / wall) and differential net J/token (socket channel) per config
import json, re, glob, os, sys
O = "logs/beam/bench"
def ch(p):
    d = json.load(open(p))["breakdown"]["channels"]; return {c["fabric"]: c for c in d}
def info(p):
    t = open(p).read(); m = re.search(r"total ([\d.]+)s\s+gpu ([\d.]+)s\s+disk\(stream\) ([\d.]+)s \(([\d.]+) GB.*steps (\d+)", t)
    h = re.search(r"layer: hit (\d+)/(\d+)", t); sps = re.findall(r"per-step wall s: ([\d. ]+)", t)
    return (float(m[1]), float(m[4]), int(m[5]), (int(h[1]) / int(h[2]) if h and int(h[2]) else None), [float(x) for x in sps[0].split()] if sps else []) if m else None
cfgs = sorted({os.path.basename(f).rsplit("_n", 1)[0] for f in glob.glob(O + "/*_n40_r1.json")})
sock = sys.argv[1] if len(sys.argv) > 1 else None
print("channels:", list(ch(f"{O}/{cfgs[0]}_n40_r1.json")))
for c in cfgs:
    row = []
    for r in (1, 2):
        try: a, b = f"{O}/{c}_n40_r{r}", f"{O}/{c}_n2_r{r}"; ia, ib = info(a + ".log"), info(b + ".log"); ca, cb = ch(a + ".json"), ch(b + ".json")
        except Exception as e: print(c, r, repr(e)); continue
        if not ia or not ib: continue
        ds = 38; steps = ia[4][len(ib[4]):] if ia[4] else []
        row.append((ia[0], ia[1], ds, ia[3], {k: (ca[k]["net_joules"] - cb[k]["net_joules"]) / ds for k in ("socket",)}, sum(steps) / max(len(steps), 1), ca["socket"]["net_joules"] / 39, ca["socket"]["avg_w"]))
    for x in row: print(f"{c:12} wall {x[0]:.1f}s disk {x[1]:.1f} GB hit {x[3]} decode s/tok(mean steps) {x[5]:.3f}  Jtot/39tok {x[6]:.1f} avgW {x[7]:.1f} dJ/tok " + " ".join(f"{k}:{v:.1f}" for k, v in x[4].items()))
