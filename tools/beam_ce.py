#!/usr/bin/env python3
import json, re, sys
O = "logs/beam/cachee"
def ch(p): return {c["fabric"]: c for c in json.load(open(p))["breakdown"]["channels"]}["socket"]
def lg(p):
    t = open(p).read(); s = re.search(r"per-step wall s: ([\d. ]+)", t); h = re.findall(r"^cache \d+ slots/layer: hit (\d+)/(\d+)", t, re.M)
    return [float(x) for x in s[1].split()] if s else [], (int(h[-1][0]), int(h[-1][1])) if h else (0, 0)
for l in open(O + "/cfg.txt"):
    name, nl, ns = l.split(); nl, ns = int(nl), int(ns); res = []
    for r in (1, 2):
        try: a = f"{O}/{name}_n{nl}_r{r}"; b = f"{O}/{name}_n{ns}_r{r}"; ca, cb = ch(a + ".json"), ch(b + ".json"); sa, ha = lg(a + ".log"); sb, hb = lg(b + ".log")
        except Exception as e: print(name, r, e); continue
        d = nl - ns; dj = (ca["net_joules"] - cb["net_joules"]) / d; extra = sa[len(sb):]
        mh = (ha[0] - hb[0]) / max(ha[1] - hb[1], 1)
        res.append(dj); print(f"{name:10} rep{r} diffJ/tok {dj:6.2f}  s/tok(extra steps) {sum(extra)/max(len(extra),1):.3f}  marginal hit {mh:.3f}  net W long {ca['net_joules']/ (sum(sa)+1):.1f}")
    if res: print(f"{name:10} MEAN net J/token {sum(res)/len(res):.2f}")
