#!/usr/bin/env python3
"""tools/flint/analyze.py : summarize logs/flint (accuracy on the 800-item test split, differential socket energy).
J/decision and J/token are differential: (J_hi - J_lo) / (units_hi - units_lo); gross = SMU socket joules, net = minus idle_baseline W x window seconds.
Windows whose busy-core average (window + idle) > BUSY_MAX are dropped (other jobs on the box)."""
import glob, json, os, statistics as st, sys
D = os.path.expanduser("~/strix-rt/logs/flint"); BUSY_MAX = 1.5; DS = ["ag_news", "sst5", "mnli", "boolq"]
NAMES = {"flintq41": "Flint Q4_1 (official INT4 GGUF)", "flintnv": "Flint NVFP4 (official GGUF)", "qwen27": "Qwen3.8-27B UD-Q4_K_M"}
busy = {}
if os.path.exists(D + "/busy.txt"):
    for l in open(D + "/busy.txt"): busy[l.split()[0]] = float(l.rsplit(":", 1)[1])
def g(p):
    d = json.load(open(p)); ch = {c["fabric"]: c for c in d["breakdown"]["channels"]}; ib = {c["fabric"]: c for c in d["idle_baseline"]["channels"]}
    return ch["socket"]["joules"], d["breakdown"]["seconds"], ib["socket"]["avg_w"], ch["gfx"]["joules"]
def diff(base, ulo, uhi):
    tag = os.path.basename(base); b = max(busy.get(tag + ".lo", 0), busy.get(tag + ".hi", 0))
    (j0, t0, w0, g0), (j1, t1, w1, g1) = g(base + ".lo.json"), g(base + ".hi.json"); w = (w0 + w1) / 2; du = uhi - ulo
    return dict(gross=(j1 - j0) / du, net=(j1 - j0 - w * (t1 - t0)) / du, gfx=(g1 - g0) / du, idle_w=w, busy=b, ok=b <= BUSY_MAX)
def tok(p): return [json.loads(l) for l in open(p) if l.startswith('{"mode"')][-1]  # det-telemetry appends its table after the bench JSON line
def med(x): return st.median(x) if x else float("nan")
out = ["| model | test acc | decision ms | J/decision gross (net) | prefill tok/s | J/tok prefill gross (net) | decode tok/s | J/tok decode gross (net) | idle W | dropped windows |", "|" + "---|" * 10]
for n, name in NAMES.items():
    accs, ms = [], []
    for ds in DS:
        f = f"{D}/acc/{n}_{ds}.jsonl"
        if not os.path.exists(f): continue
        r = [json.loads(l) for l in open(f)]; t = [x for x in r if int(x["id"].rsplit("-", 1)[1]) >= 50]
        accs.append(sum(x["pred"] == x["gold"] for x in t) / len(t)); ms += [x["ms"] for x in t]
    acc = sum(accs) / len(accs) if len(accs) == 4 else float("nan")
    dec, drop, idle = [], [], []
    for ds in DS:
        for lo in glob.glob(f"{D}/energy/{n}_{ds}_r*.lo.json"):
            base = lo[:-8]
            if not os.path.exists(base + ".hi.json"): continue
            u = [eval(open(base + f".{w}.meta").read().replace(" ", "*")) for w in ("lo", "hi")]; e = diff(base, *u)
            (dec if e["ok"] else drop).append(e if e["ok"] else os.path.basename(base)); idle.append(e["idle_w"]) if e["ok"] else None
    jd = [st.mean([e["gross"] for e in dec]) if len(dec) >= 4 else float("nan"), st.mean([e["net"] for e in dec]) if len(dec) >= 4 else float("nan")]
    tk = {}
    for m, unit in (("pp", "prompt_tokens"), ("tg", "gen_tokens")):
        rows = []
        for lo in glob.glob(f"{D}/energy/{n}_{m}_r*.lo.json"):
            base = lo[:-8]
            if not os.path.exists(base + ".hi.json"): continue
            a, b = tok(base + ".lo.out"), tok(base + ".hi.out"); e = diff(base, a[unit], b[unit])
            tps = b["pp_tps"] if m == "pp" else b["tg_tps"]
            if e["ok"]: rows.append((e, tps)); idle.append(e["idle_w"])
            else: drop.append(os.path.basename(base))
        tk[m] = (med([r[1] for r in rows]), med([r[0]["gross"] for r in rows]), med([r[0]["net"] for r in rows]))
    pp, tg = tk["pp"], tk["tg"]; dr = ", ".join(drop) or "-"
    out.append(f"| {name} | {acc:.3f} | {med(ms):.0f} | {jd[0]:.2f} ({jd[1]:.2f}) | {pp[0]:.0f} | {pp[1]:.4f} ({pp[2]:.4f}) | {tg[0]:.1f} | {tg[1]:.3f} ({tg[2]:.3f}) | {med(idle):.1f} | {dr} |")
print("\n".join(out))
