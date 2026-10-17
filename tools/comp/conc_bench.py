# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx"]
# ///
"""Soothill-style concurrency bench (soothill.io 2026-08-10 method) against an OpenAI-compatible /v1/completions server.
Deterministic ~480-word prompts (unique per request so prefix caches cannot help), 64 output tokens (ignore_eos),
temperature 0, seed 12345, streaming. Per concurrency C: 2 warm-up waves, then 10 measured waves of C simultaneous requests.
Reports total completion tok/s (all completion tokens / summed wave wall time), mean TTFT, mean e2e, success rate.
usage: uv run conc_bench.py URL MODEL [C,..] [--waves 10] [--warm 2] [--out file.json]"""
import asyncio, json, sys, time, random, argparse, httpx
WORDS = ("strix halo memory bandwidth unified lpddr5x compute unit wavefront kernel dispatch tensor matrix vector cache "
         "latency throughput scheduler token prompt decode prefill expert router attention layer norm residual").split()
def prompt(i):
    r = random.Random(1000 + i)
    body = " ".join(r.choice(WORDS) for _ in range(470))
    return f"Request {i}. Summarize the following notes in plain English.\n\n{body}\n\nSummary:"
async def one(cl, url, model, i, maxt):
    t0 = time.perf_counter(); ttft = None; n = 0; usage = None
    js = {"model": model, "prompt": prompt(i), "max_tokens": maxt, "temperature": 0, "seed": 12345, "stream": True,
          "ignore_eos": True, "stream_options": {"include_usage": True}, "cache_prompt": False}
    try:
        async with cl.stream("POST", url + "/v1/completions", json=js, timeout=600) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data:"): continue
                d = line[5:].strip()
                if d == "[DONE]": break
                o = json.loads(d)
                if o.get("usage"): usage = o["usage"]
                for c in o.get("choices") or []:
                    if c.get("text"):
                        if ttft is None: ttft = time.perf_counter() - t0
                        n += 1
        ct = (usage or {}).get("completion_tokens") or n
        return dict(ok=True, ttft=ttft, e2e=time.perf_counter() - t0, ctoks=ct, ptoks=(usage or {}).get("prompt_tokens"))
    except Exception as e:
        return dict(ok=False, err=repr(e)[:200], e2e=time.perf_counter() - t0, ctoks=0)
async def main():
    ap = argparse.ArgumentParser(); ap.add_argument("url"); ap.add_argument("model"); ap.add_argument("conc", nargs="?", default="1,2,4,8")
    ap.add_argument("--waves", type=int, default=10); ap.add_argument("--warm", type=int, default=2); ap.add_argument("--maxt", type=int, default=64)
    ap.add_argument("--out"); a = ap.parse_args(); res = []; rid = 0
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=64)) as cl:
        for C in [int(c) for c in a.conc.split(",")]:
            wall = 0.0; rs = []
            for w in range(a.warm + a.waves):
                t = time.perf_counter(); out = await asyncio.gather(*[one(cl, a.url, a.model, rid + k, a.maxt) for k in range(C)]); rid += C
                if w >= a.warm: wall += time.perf_counter() - t; rs += out
            ok = [r for r in rs if r["ok"]]; ct = sum(r["ctoks"] for r in ok)
            row = dict(conc=C, requests=len(rs), success_rate=len(ok) / len(rs), completion_tok_s=ct / wall if wall else 0,
                       mean_ttft_ms=1e3 * sum(r["ttft"] for r in ok) / max(1, len(ok)) if ok and all(r["ttft"] for r in ok) else None,
                       mean_e2e_ms=1e3 * sum(r["e2e"] for r in ok) / max(1, len(ok)), mean_ctoks=ct / max(1, len(ok)),
                       mean_ptoks=(sum(r["ptoks"] or 0 for r in ok) / max(1, len(ok))), wall_s=wall, errors=[r["err"] for r in rs if not r["ok"]][:3])
            print(json.dumps(row), flush=True); res.append(row)
    if a.out: json.dump(res, open(a.out, "w"), indent=1)
asyncio.run(main())
