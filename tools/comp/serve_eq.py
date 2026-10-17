"""strix-serve check: greedy outputs of N different prompts served one at a time (C=1) must equal the outputs when
all N are served concurrently (batched decode). Also checks chat SSE streaming and /v1/models.
usage: python3 eq_test.py URL [N=8] [maxt=48]"""
import json, sys, threading, urllib.request, time
url = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 8; MAXT = int(sys.argv[3]) if len(sys.argv) > 3 else 48
def post(path, js, stream=False):
    r = urllib.request.urlopen(urllib.request.Request(url + path, data=json.dumps(js).encode(), headers={"Content-Type": "application/json"}), timeout=900)
    return r if stream else json.loads(r.read())
topics = ["the moon landing", "how GPUs schedule wavefronts", "a recipe for bread", "the history of Rome", "why the sky is blue",
          "sorting algorithms", "the water cycle", "black holes", "jazz music", "volcanoes", "the printing press", "honey bees"]
prompts = [f"Write a short paragraph about {topics[i % len(topics)]}" + (" in detail" * (i // len(topics))) + ".\n" for i in range(N)]
def run(i, out): out[i] = post("/v1/completions", {"prompt": prompts[i], "max_tokens": MAXT, "temperature": 0, "ignore_eos": True})["choices"][0]["text"]
print("models:", post.__name__ and json.loads(urllib.request.urlopen(url + "/v1/models").read())["data"][0]["id"])
seq = [None] * N; t = time.time()
for i in range(N): run(i, seq)
ts = time.time() - t
con = [None] * N; t = time.time(); th = [threading.Thread(target=run, args=(i, con)) for i in range(N)]
[x.start() for x in th]; [x.join() for x in th]; tc = time.time() - t
ok = sum(a == b for a, b in zip(seq, con))
for i in range(N):
    if seq[i] != con[i]: print(f"MISMATCH {i}:\n  C=1 {seq[i]!r}\n  C={N} {con[i]!r}")
print(f"greedy batched == single-stream: {ok}/{N}  (sequential {ts:.1f}s, concurrent {tc:.1f}s)")
print("sample:", repr(seq[0][:200]))
r = post("/v1/chat/completions", {"messages": [{"role": "user", "content": "Say hello in French, one word."}], "max_tokens": 16, "temperature": 0, "stream": True,
         "chat_template_kwargs": {"enable_thinking": False}, "stream_options": {"include_usage": True}}, stream=True)
txt = ""; usage = None
for line in r:
    line = line.decode().strip()
    if not line.startswith("data:") or line == "data: [DONE]": continue
    o = json.loads(line[5:])
    usage = o.get("usage") or usage
    for c in o["choices"]: txt += c.get("delta", {}).get("content", "")
print("chat stream:", repr(txt), "usage", usage)
