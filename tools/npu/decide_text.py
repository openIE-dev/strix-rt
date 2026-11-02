"""Zero-shot decision client for servers without logprobs (FLM). Same prompt as tools/cascade t2; pred = first letter of reply.
usage: decide_text.py URL MODEL OUT [passes] [per_ds_lo per_ds_hi]   (test split = items 50.. per dataset)"""
import json, sys, time, urllib.request
L = "ABCDEFGHIJ"
url, model, out = sys.argv[1:4]; passes = int(sys.argv[4]) if len(sys.argv) > 4 else 1
lo = int(sys.argv[5]) if len(sys.argv) > 5 else 50; hi = int(sys.argv[6]) if len(sys.argv) > 6 else 250
its, cnt = [], {}
for ln in open("tools/cascade/eval.jsonl"):
    it = json.loads(ln); c = cnt.get(it["ds"], 0); cnt[it["ds"]] = c + 1
    if lo <= c < hi: its.append(it)
f = open(out, "w"); n = 0; t0 = time.time()
for p in range(passes):
    for it in its:
        keys = list(it["options"]) if it["qtype"] == "choice" else ["yes", "no"]
        opts = "\n".join(f"{L[i]}. {k}" + (f": {it['options'][k]}" if it["qtype"] == "choice" and it["options"][k] else "") for i, k in enumerate(keys))
        prompt = f"{it['state']}\n\nQuestion: {it['instructions']}\nOptions:\n{opts}\nAnswer with the single letter of the best option and nothing else."
        body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1, "temperature": 0,
                "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}, "think": False}
        t = time.time()
        r = json.load(urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
        ms = (time.time() - t) * 1e3; txt = (r["choices"][0]["message"].get("content") or "").strip()
        let = next((ch for ch in txt if ch in L[:len(keys)]), None); pred = keys[L.index(let)] if let else keys[0]
        n += 1
        if p == 0: f.write(json.dumps(dict(id=it["id"], ds=it["ds"], gold=it["gold"], pred=pred, txt=txt, ms=ms, usage=r.get("usage"))) + "\n")
print(f"DONE {n} decisions in {time.time() - t0:.2f} s", flush=True)
