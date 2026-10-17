# serve1_test.py URL [image.png]: text chat (greedy, repeated -> deterministic), streaming, completions, optional image chat
import sys, json, base64, time, urllib.request
U = sys.argv[1]
def post(p, b):
    r = urllib.request.Request(U + p, json.dumps(b).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=600).read())
msg = [{"role": "user", "content": "Write one sentence about the moon landing."}]
outs = []
for i in range(2):
    t = time.time(); r = post("/v1/chat/completions", {"model": "m", "messages": msg, "max_tokens": 48, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
    outs.append(r["choices"][0]["message"]["content"]); print(f"chat {i}: {time.time()-t:.2f}s usage {r['usage']} -> {outs[-1]!r}")
print("DETERMINISTIC", outs[0] == outs[1])
r = urllib.request.urlopen(urllib.request.Request(U + "/v1/chat/completions", json.dumps({"model": "m", "messages": msg, "max_tokens": 16, "temperature": 0, "stream": True,
    "chat_template_kwargs": {"enable_thinking": False}}).encode(), {"Content-Type": "application/json"}), timeout=600)
txt = "".join(json.loads(l[6:])["choices"][0]["delta"].get("content", "") for l in r.read().decode().splitlines() if l.startswith("data: {") and json.loads(l[6:])["choices"])
print("STREAM", repr(txt), "prefix-ok", outs[0].startswith(txt.rstrip()[:20]))
r = post("/v1/completions", {"model": "m", "prompt": "The capital of France is", "max_tokens": 8, "temperature": 0}); print("COMPLETION", repr(r["choices"][0]["text"]))
r = post("/v1/chat/completions", {"model": "m", "messages": msg, "max_tokens": 24, "temperature": 0.8, "seed": 7, "chat_template_kwargs": {"enable_thinking": False}}); print("SAMPLED", repr(r["choices"][0]["message"]["content"]))
if len(sys.argv) > 2:
    b = base64.b64encode(open(sys.argv[2], "rb").read()).decode()
    m2 = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + b}}, {"type": "text", "text": "Describe this image in one paragraph."}]}]
    t = time.time(); r = post("/v1/chat/completions", {"model": "m", "messages": m2, "max_tokens": 96, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
    print(f"IMAGE {time.time()-t:.2f}s usage {r['usage']} -> {r['choices'][0]['message']['content']!r}")
