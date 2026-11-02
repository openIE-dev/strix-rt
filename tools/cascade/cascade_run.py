#!/usr/bin/env python3
"""Cascade client (stdlib only). Tier 1 = System One server (kev_serve POST /v1/systemone), tier 2 = llama-server chat completions
(first-token option-letter logprobs, no thinking, no prompt cache). One decision = one request.
  cascade_run.py run  --tier t1|t2 --url U [--model M] [--ds NAME] [--lo A --hi B] [--passes P] [--out F]
  cascade_run.py casc --thr T --t1url U1 --t1model M --t2url U2 [--ds NAME ...] [--lo A --hi B] [--out F]
Each output line: {id, ds, gold, pred, probs{key:p}, conf (=max p), ms, [esc, tier]}. conf is the max option probability."""
import argparse, json, math, os, sys, time, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
def items(ds, lo, hi):
    its = [json.loads(l) for l in open(os.path.join(HERE, "eval.jsonl"))]
    if ds: its = [i for i in its if i["ds"] in ds]
    out = []
    for d in dict.fromkeys(i["ds"] for i in its): out += [i for i in its if i["ds"] == d][lo:hi]
    return out
def post(url, body, timeout=600):
    r = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=timeout))
def t1(it, url, model):
    q = {"type": it["qtype"], "instructions": it["instructions"]}
    if it["qtype"] == "choice": q["criteria"] = it["options"]
    t0 = time.time(); r = post(url + "/v1/systemone", {"state": it["state"], "model": model, "questions": {"q": q}}); ms = (time.time() - t0) * 1e3
    a = r["answers"]["q"]
    if it["qtype"] == "noul": py = a["noul"]; probs = {"yes": py, "no": 1 - py}
    else: probs = a["probabilities"]
    s = sum(probs.values()) or 1.0; probs = {k: v / s for k, v in probs.items()}
    pred = max(probs, key=probs.get)
    return dict(pred=pred, probs=probs, conf=probs[pred], ms=ms)
L = "ABCDEFGH"
def t2(it, url, model):
    keys = list(it["options"]) if it["qtype"] == "choice" else ["yes", "no"]
    opts = "\n".join(f"{L[i]}. {k}" + (f": {it['options'][k]}" if it["qtype"] == "choice" and it["options"][k] else "") for i, k in enumerate(keys))
    prompt = f"{it['state']}\n\nQuestion: {it['instructions']}\nOptions:\n{opts}\nAnswer with the single letter of the best option and nothing else."
    body = {"model": model or "m", "messages": [{"role": "user", "content": prompt}], "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 20,
            "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time(); r = post(url + "/v1/chat/completions", body); ms = (time.time() - t0) * 1e3
    tl = r["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    p = {k: 0.0 for k in keys}
    for e in tl:
        t = e["token"].strip()
        if len(t) == 1 and t in L[:len(keys)]: p[keys[L.index(t)]] += math.exp(e["logprob"])
    s = sum(p.values())
    if s == 0: p = {k: 1.0 / len(keys) for k in keys}; s = 1.0
    p = {k: v / s for k, v in p.items()}; pred = max(p, key=p.get)
    return dict(pred=pred, probs=p, conf=p[pred], ms=ms)
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--tier"); ap.add_argument("--url"); ap.add_argument("--model", default="")
    ap.add_argument("--ds", action="append"); ap.add_argument("--lo", type=int, default=0); ap.add_argument("--hi", type=int, default=10**9)
    ap.add_argument("--passes", type=int, default=1); ap.add_argument("--out")
    ap.add_argument("--thr", type=float); ap.add_argument("--t1url"); ap.add_argument("--t1model", default=""); ap.add_argument("--t2url"); ap.add_argument("--t2model", default="m"); ap.add_argument("--t2kind", default="t2")
    a = ap.parse_args(); its = items(a.ds, a.lo, a.hi); f = open(a.out, "w") if a.out else None; n = 0; t0 = time.time()
    if a.cmd == "run":
        fn = t1 if a.tier == "t1" else t2
        for p in range(a.passes):
            for it in its:
                r = fn(it, a.url, a.model); n += 1
                if f and p == 0: f.write(json.dumps(dict(id=it["id"], ds=it["ds"], gold=it["gold"], **r)) + "\n")
    else:
        for it in its:
            r = t1(it, a.t1url, a.t1model); esc = r["conf"] < a.thr; tier = 1
            if esc: r2 = (t1 if a.t2kind == "t1" else t2)(it, a.t2url, a.t2model); r = dict(r2, ms=r["ms"] + r2["ms"]); tier = 2
            n += 1
            if f: f.write(json.dumps(dict(id=it["id"], ds=it["ds"], gold=it["gold"], esc=esc, tier=tier, **r)) + "\n")
    print(f"DONE {n} decisions in {time.time() - t0:.2f} s", flush=True)
if __name__ == "__main__": main()
