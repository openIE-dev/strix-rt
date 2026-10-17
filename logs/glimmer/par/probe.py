# probe.py PORT TAG: teacher-forced top logprobs at the mismatch positions (prompt idx, pos, ours_tok) against one llama-server
import json, sys, urllib.request, math
port, tag = sys.argv[1], sys.argv[2]
R = json.load(open("../ref_llama.json"))
cases = [(2, 86, 1574), (5, 73, 11624), (0, 43, 28449)]
def post(p, d):
    r = urllib.request.Request(f"http://127.0.0.1:{port}{p}", json.dumps(d).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=900))
for pi, pos, ours in cases:
    r = R[pi]; ids = r["ids"] + r["gen"][:pos]; ref = r["gen"][pos]
    o = post("/completion", {"prompt": ids, "n_predict": 1, "temperature": 0, "top_k": 1, "n_probs": 200, "post_sampling_probs": False, "cache_prompt": False})
    tl = {t["id"]: t["logprob"] for t in o["completion_probabilities"][0]["top_logprobs"]}
    top = o["completion_probabilities"][0]["id"]; a = tl.get(ref); b = tl.get(ours)
    print("PROBE", tag, "prompt", pi, "pos", pos, "top", top, "lp_ref", a, "lp_ours", b, "diff(ref-ours)", None if a is None or b is None else round(a - b, 4), flush=True)
