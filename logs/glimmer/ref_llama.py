# llama-server greedy reference with per-step top-2 probabilities (margin). usage: ref_llama.py PORT N OUT.json
import json, sys, urllib.request
port, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
def post(p, d):
    r = urllib.request.Request(f"http://127.0.0.1:{port}{p}", json.dumps(d).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=60))
res = []
for pr in json.load(open(sys.argv[4] if len(sys.argv) > 4 else "prompts.json")):
    ids = post("/tokenize", {"content": pr, "add_special": False})["tokens"]
    r = post("/completion", {"prompt": ids, "n_predict": n, "temperature": 0, "top_k": 1, "n_probs": 2, "post_sampling_probs": False, "cache_prompt": False, "ignore_eos": True})
    gen = [t["id"] for t in r["completion_probabilities"]]
    marg = [t["top_logprobs"][0]["logprob"] - t["top_logprobs"][1]["logprob"] for t in r["completion_probabilities"]]
    res.append({"prompt": pr, "ids": ids, "gen": gen, "margin": marg, "text": r["content"], "timings": r.get("timings")})
    print(len(ids), gen[:8], round(min(marg), 3), repr(r["content"][:80]), flush=True)
json.dump(res, open(out, "w"))
