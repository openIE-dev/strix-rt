"""Decode load: gen.py URL MODEL MAXTOK REPS META -> writes total completion tokens to META. Fixed prompt, temp 0, no prompt cache."""
import json, sys, urllib.request
url, model, mt, reps, meta = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
P = "Write a long, detailed essay about the history of the printing press, its inventors, and its effect on European science and religion."
tot = 0
for r in range(reps):
    body = {"model": model, "messages": [{"role": "user", "content": P}], "max_tokens": mt, "temperature": 0, "cache_prompt": False,
            "ignore_eos": True, "chat_template_kwargs": {"enable_thinking": False}, "think": False}
    d = json.load(urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1200))
    tot += d["usage"]["completion_tokens"]
open(meta, "w").write(f"{tot}\n"); print("TOKENS", tot, flush=True)
