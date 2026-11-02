#!/usr/bin/env python3
"""Cascade analysis (stdlib only). Reads logs/cascade/acc/*.jsonl (per-item tier outputs) and logs/cascade/energy/*.json (+ .meta) and prints
markdown tables: tier accuracy/calibration, J per decision per tier and dataset, threshold sweep (accuracy, escalation rate, J/decision,
J/correct), break-even escalation rate. Dev = first 50 items per dataset (temperature refit), test = remaining 200.
Energy per decision = differential (J_hi - J_lo)/(D_hi - D_lo) over decision counts (windows are whole passes of the same 100 items);
'gross' = socket joules, 'net' = gross - idle_baseline_W * delta_t (idle power removed). Cascade energy charges the small run for every decision
and the large run for every escalation (GreenGate: discarded small work is NOT free)."""
import json, glob, os, math, statistics as st, sys, collections
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs", "cascade")
BUSY_MAX = 1.5  # windows whose busy-core average exceeds this are dropped (other workers)
DS = ["ag_news", "sst5", "mnli", "boolq"]
def load(p): return {r["id"]: r for r in map(json.loads, open(p))} if os.path.exists(p) else {}
def split(i): return "dev" if int(i.rsplit("-", 1)[1]) < 50 else "test"
def temp(p, T):
    q = {k: max(v, 1e-9) ** (1 / T) for k, v in p.items()}; s = sum(q.values()); return {k: v / s for k, v in q.items()}
def ece(rs, nb=10):
    b = collections.defaultdict(list)
    for c, ok in rs: b[min(int(c * nb), nb - 1)].append((c, ok))
    n = len(rs); return sum(len(v) / n * abs(sum(c for c, _ in v) / len(v) - sum(o for _, o in v) / len(v)) for v in b.values())
def refit(r1):
    dev = [r for r in r1.values() if split(r["id"]) == "dev"]; best = (1e18, 1.0)
    for T in [x / 20 for x in range(8, 100)]:
        nll = -sum(math.log(max(temp(r["probs"], T)[r["gold"]], 1e-9)) for r in dev) / len(dev); best = min(best, (nll, T))
    return best[1]
def apply_T(r1, T):
    out = {}
    for i, r in r1.items():
        p = temp(r["probs"], T); pred = max(p, key=p.get); out[i] = dict(r, probs=p, pred=pred, conf=p[pred])
    return out
def energy():
    """-> {(tier, ds): {'gross': [J/dec per rep], 'net': [...]}}"""
    E = collections.defaultdict(lambda: {"gross": [], "net": [], "busy": []})
    busy = {}
    if os.path.exists(D + "/busy.txt"):
        for l in open(D + "/busy.txt"): t = l.split()[0]; busy[t] = float(l.rsplit(":", 1)[1])
    for f in sorted(glob.glob(D + "/energy/*_r*.lo.json")):
        base = f[:-8]; tag = os.path.basename(base); tier, rest = tag.split("_", 1); ds, rep = rest.rsplit("_r", 1)
        if tier != "t2" and rest.startswith(("ag_", "sst", "mnli", "boolq")) is False: pass
        # tag looks like kev08_ag_news_r1 / t2_boolq_r2
        hi = base + ".hi.json"
        if not os.path.exists(hi) or not os.path.exists(base + ".lo.meta") or not os.path.exists(base + ".hi.meta"): continue
        def g(p):
            d = json.load(open(p)); ch = {c["fabric"]: c for c in d["breakdown"]["channels"]}; ib = {c["fabric"]: c for c in d["idle_baseline"]["channels"]}
            return ch["socket"]["joules"], d["breakdown"]["seconds"], ib["socket"]["avg_w"], ch["gfx"]["joules"]
        def dec(p): a, b = map(int, open(p).read().split()); return a * b
        (j0, t0, w0, g0), (j1, t1, w1, g1) = g(base + ".lo.json"), g(hi); dd = dec(base + ".hi.meta") - dec(base + ".lo.meta")
        w = (w0 + w1) / 2; k = (tier, ds)
        if max(busy.get(tag + ".lo", 0), busy.get(tag + ".hi", 0)) > BUSY_MAX: E[k].setdefault("dropped", []).append(tag); continue
        E[k]["gross"].append((j1 - j0) / dd); E[k]["net"].append((j1 - j0 - w * (t1 - t0)) / dd)
        E[k].setdefault("gfx", []).append((g1 - g0) / dd); E[k].setdefault("win", []).append((t0, t1)); E[k].setdefault("idlew", []).append(w)
        E[k]["busy"].append(max(busy.get(tag + ".lo", 0), busy.get(tag + ".hi", 0))); E[k].setdefault("dd", []).append(dd)
    return E
def med(x): return st.median(x) if x else float("nan")
def main():
    tiers1 = [os.path.basename(p)[:-len("_ag_news.jsonl")] for p in sorted(glob.glob(D + "/acc/*_ag_news.jsonl")) if not os.path.basename(p).startswith("t2_")]
    r2 = {}; [r2.update(load(f"{D}/acc/t2_{d}.jsonl")) for d in DS]
    out = {}; E = energy()
    print("## Tier accuracy and calibration (test = 800 items, 200 per dataset)\n")
    print("| tier | acc (all) | " + " | ".join(f"acc {d}" for d in DS) + " | ECE raw | ECE refit | T refit |\n|---|---|" + "---|" * (len(DS) + 3))
    def row(name, r, T=None, rawr=None):
        te = [x for x in r.values() if split(x["id"]) == "test"]; a = sum(x["pred"] == x["gold"] for x in te) / len(te)
        per = [sum(x["pred"] == x["gold"] for x in te if x["ds"] == d) / max(1, sum(1 for x in te if x["ds"] == d)) for d in DS]
        e = ece([(x["conf"], x["pred"] == x["gold"]) for x in te])
        print(f"| {name} | {a:.3f} | " + " | ".join(f"{p:.3f}" for p in per) + f" | {rawr if rawr is not None else e:.3f} | {e:.3f} | {T if T else '-'} |")
        return a
    A2 = row("tier 2 Qwen3.8-27B Q4_K_M (letter logprobs)", r2) if r2 else None
    R1 = {}
    for t in tiers1:
        r = {}; [r.update(load(f"{D}/acc/{t}_{d}.jsonl")) for d in DS]
        if len(r) < 1000: continue
        te = [x for x in r.values() if split(x["id"]) == "test"]; raw = ece([(x["conf"], x["pred"] == x["gold"]) for x in te])
        T = refit(r); R1[t] = apply_T(r, T); row("tier 1 " + t, R1[t], T, raw)
    print("\n## Energy per decision (median of reps; J socket; gfx in parentheses; net = idle baseline removed)\n")
    print("| tier | " + " | ".join(DS) + " | mean over datasets (gross) | mean (net) | reps | max busy cores in a window |\n|---|" + "---|" * (len(DS) + 4))
    eg, en = {}, {}
    for tier in tiers1 + ["t2"]:
        cells = []; gs = []; ns = []; reps = []; bz = []
        for d in DS:
            v = E.get((tier, d))
            if not v or not v["gross"]: cells.append("-"); continue
            eg[(tier, d)] = med(v["gross"]); en[(tier, d)] = med(v["net"]); gs.append(eg[(tier, d)]); ns.append(en[(tier, d)]); reps.append(len(v["gross"])); bz.append(max(v["busy"]))
            cells.append(f"{med(v['gross']):.2f} ({med(v['gfx']):.2f}) [{min(v['gross']):.2f}-{max(v['gross']):.2f}]")
        dr = sum(len(E[(tier, d)].get("dropped", [])) for d in DS if (tier, d) in E)
        if gs: print(f"| {tier} | " + " | ".join(cells) + f" | {sum(gs)/len(gs):.2f} | {sum(ns)/len(ns):.2f} | {min(reps)} (dropped {dr}) | {max(bz):.1f} |")
    THR = [0, .5, .6, .7, .8, .9, .95, .97, .99, .995, 1.01]
    R2 = {"t2": r2, "kev27": R1.pop("kev27", {})}
    for (t, r1), (t2n, r2) in [(x, y) for x in R1.items() for y in R2.items() if y[1]]:
        for metric, ee in (("gross", eg), ("net", en)):
            if not all((t, d) in ee for d in DS) or not all((t2n, d) in ee for d in DS) or not r2: continue
            te = [i for i in r1 if split(i) == "test"]; n = len(te)
            def casc(thr, oracle=False):
                J = c = e = 0
                for i in te:
                    d = r1[i]["ds"]; J += ee[(t, d)]; use2 = (r1[i]["pred"] != r1[i]["gold"]) if oracle else r1[i]["conf"] < thr
                    if use2: J += ee[(t2n, d)]; e += 1; c += r2[i]["pred"] == r2[i]["gold"]
                    else: c += r1[i]["pred"] == r1[i]["gold"]
                return c / n, e / n, J / n
            es = sum(ee[(t, r1[i]["ds"])] for i in te) / n; el = sum(ee[(t2n, r1[i]["ds"])] for i in te) / n
            a2 = sum(r2[i]["pred"] == r2[i]["gold"] for i in te) / n; j2 = el; a1, _, j1 = casc(0)  # always-large = tier 2 alone, no tier-1 run
            print(f"\n## Cascade {t} -> {t2n}, {metric} socket joules (test n={n}); always-small {a1:.3f} acc, {j1:.2f} J/dec, {j1/a1:.2f} J/correct; always-large {a2:.3f} acc, {j2:.2f} J/dec, {j2/a2:.2f} J/correct\n")
            print("| threshold | escalated | accuracy | J/decision | J/correct | J/dec vs always-large | J/correct vs always-large |\n|---|---|---|---|---|---|---|")
            for thr in THR:
                a, e, j = casc(thr); print(f"| {thr} | {e:.3f} | {a:.3f} | {j:.2f} | {j/a:.2f} | {j/j2-1:+.1%} | {(j/a)/(j2/a2)-1:+.1%} |")
            a, e, j = casc(0, True); print(f"| oracle (escalate only wrong) | {e:.3f} | {a:.3f} | {j:.2f} | {j/a:.2f} | {j/j2-1:+.1%} | {(j/a)/(j2/a2)-1:+.1%} |")
            rstar = 1 - es / el
            print(f"\nBreak-even escalation rate (J/decision, router overhead ~0): r* = 1 - e_s/e_l = 1 - {es:.2f}/{el:.2f} = {rstar:.3f}; per dataset: " +
                  ", ".join(f"{d} {1 - ee[(t, d)] / ee[('t2', d)]:.3f}" for d in DS))
            # J/correct break-even: smallest threshold whose J/correct <= always-large
            ok = [thr for thr in THR if (lambda x: x[2] / x[0] <= j2 / a2)(casc(thr))]
            print("J/correct below always-large for thresholds: " + (", ".join(map(str, ok)) if ok else "none"))
if __name__ == "__main__": main()
