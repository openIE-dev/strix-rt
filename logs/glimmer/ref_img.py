# image prompt greedy ref via llama-server /completion with multimodal_data. usage: ref_img.py PORT N OUT.json IMAGE
import json, sys, urllib.request, base64
port, n, out, img = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
def post(p, d):
    r = urllib.request.Request(f"http://127.0.0.1:{port}{p}", json.dumps(d).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=120))
mk = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/props", timeout=10)).get("media_marker", "<__media__>")
pr = mk + "\nThis image shows"
r = post("/completion", {"prompt": {"prompt_string": pr, "multimodal_data": [base64.b64encode(open(img, "rb").read()).decode()]}, "n_predict": n, "temperature": 0, "top_k": 1, "n_probs": 2, "post_sampling_probs": False, "cache_prompt": False, "ignore_eos": True})
gen = [t["id"] for t in r["completion_probabilities"]]
marg = [t["top_logprobs"][0]["logprob"] - t["top_logprobs"][1]["logprob"] for t in r["completion_probabilities"]]
json.dump([{"prompt": pr, "image": img, "gen": gen, "margin": marg, "text": r["content"], "timings": r.get("timings"), "n_prompt": r.get("tokens_evaluated")}], open(out, "w"))
print("img", r.get("tokens_evaluated"), gen[:8], round(min(marg), 3), repr(r["content"][:120]), flush=True)
