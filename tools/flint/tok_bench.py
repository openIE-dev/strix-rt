#!/usr/bin/env python3
"""Token-throughput load for llama-server (stdlib only), used inside det-telemetry windows.
  tok_bench.py URL pp NPROMPT REPS : REPS requests, each a fresh NPROMPT-token prompt, max_tokens 1 (prefill-dominated)
  tok_bench.py URL tg NGEN REPS    : REPS requests, short fixed prompt, NGEN tokens generated with ignore_eos (decode-dominated)
Prints totals from the server timings: prompt tokens, prompt ms, generated tokens, generation ms."""
import json, sys, time, urllib.request, random
url, mode, n, reps = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
def post(body):
    r = urllib.request.Request(url + "/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=3600))
words = "the of and to in is that for it as with was on be by at this from or have an they which one you were her all she there would their we him been has when who will more no if out so said what up its about into than them can only other new some could time these two may then do first any my now such like our over man me even most made after also did many before must through back years where much your way well down should because each just those people how too little state good very make world still own see men work long get here between both life being under never day same another know while last might us great old year off come since against go came right used take three".split()
P = T = G = GT = 0.0; t0 = time.time()
for i in range(reps):
    if mode == "pp":
        rnd = random.Random(i); prompt = " ".join(rnd.choice(words) for _ in range(n))
        r = post({"prompt": prompt, "n_predict": 1, "cache_prompt": False, "temperature": 0})
    else:
        r = post({"prompt": "Write a long story about a lighthouse keeper.", "n_predict": n, "ignore_eos": True, "cache_prompt": False, "temperature": 0})
    t = r["timings"]; P += t["prompt_n"]; T += t["prompt_ms"]; G += t["predicted_n"]; GT += t["predicted_ms"]
print(json.dumps(dict(mode=mode, n=n, reps=reps, prompt_tokens=P, prompt_ms=T, gen_tokens=G, gen_ms=GT, wall_s=time.time() - t0,
                      pp_tps=P / T * 1e3 if T else None, tg_tps=G / GT * 1e3 if GT else None)), flush=True)
