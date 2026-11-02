#!/usr/bin/env python3
"""Export the cascade energy tables in ML.ENERGY Benchmark v3 / leaderboard form (stdlib only).
Writes logs/mlenergy/<model>/<task>/results.json (benchmark-v3 required fields, one per clean repeat window, median repeat
chosen), logs/mlenergy/leaderboard.json ({task: {configurations: [...]}} as in ml-energy/leaderboard build_data.py) and a
flat logs/mlenergy/runs.csv (LLMRun fields). Mapping and deviations from ML.ENERGY's GPU-only H100/B200 methodology:
  - one request = one decision (one System One question / one tier-2 option-letter answer); one decision = one output token,
    so energy_per_token_joules == energy_per_request_joules and avg_output_len = 1.
  - energy is differential whole-socket energy (Strix Halo SoC: CPU + iGPU + memory controller; tools/energy/record.py socket channel):
    (J of the 'hi' pass - J of the 'lo' pass) / (decisions_hi - decisions_lo). gpu_only_* gives the gfx channel alone.
  - 'steady state' = the hi-minus-lo difference window; windows with > BUSY_MAX foreign busy cores are excluded (analyze.py).
  - ITL fields are null (no token streaming); total_input_tokens is null (not recorded per window).
Usage: tools/cascade/mlenergy_export.py [OUTDIR]"""
import collections, csv, json, os, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(analyze.D, "..", "mlenergy")
MODELS = {  # tier: model_id, nickname, total B, active B, arch, precision
    "kev08": ("local/kev-0.8b", "Kev-0.8B (System One)", 0.8, 0.8, "Dense", "bfloat16"),
    "kev4b": ("local/kev-4b", "Kev-4B (System One)", 4.0, 4.0, "Dense", "bfloat16"),
    "kev9q8": ("local/kev-9b", "Kev-9B (System One)", 9.0, 9.0, "Dense", "q8_0"),
    "kev27": ("local/kev-27b", "Kev-27B (System One)", 27.0, 27.0, "Dense", "q8_0"),
    "laya": ("local/laya-f16", "Laya (System One)", None, None, "Dense", "float16"),
    "t2": ("local/Qwen3.8-27B-UD-Q4_K_M", "Qwen3.8 27B (option-letter logprob, llama.cpp Vulkan)", 27.0, 27.0, "Dense", "q4_k_m"),
}
def main():
    E = analyze.energy(); rows = []; lb = collections.defaultdict(list)
    for (tier, ds), e in sorted(E.items()):
        if not e.get("dd"): continue
        reps = []
        for i, dd in enumerate(e["dd"]):
            (t0, t1) = e["win"][i]; dur = t1 - t0; jd = e["gross"][i]
            reps.append(dict(steady_state_energy=jd * dd, steady_state_energy_per_token=jd, steady_state_duration=dur,
                             completed=dd, num_prompts=dd, total_output_tokens=dd, total_input_tokens=None, duration=dur,
                             output_throughput=dd / dur, request_throughput=dd / dur, total_token_throughput=dd / dur,
                             net_energy_per_request=e["net"][i], gpu_only_energy_per_request=e["gfx"][i], idle_power_watts=e["idlew"][i]))
        med = sorted(reps, key=lambda r: r["steady_state_energy_per_token"])[(len(reps) - 1) // 2]
        mid, nick, tb, ab, arch, prec = MODELS.get(tier, (tier, tier, None, None, "Dense", "unknown"))
        d = os.path.join(OUT, mid.replace("/", "__"), ds); os.makedirs(d, exist_ok=True)
        json.dump(dict(med, repeats=reps, n_repeats=len(reps), dropped_windows=e.get("dropped", []),
                       methodology="strix-rt tools/cascade/mlenergy_export.py (see docstring)"), open(os.path.join(d, "results.json"), "w"), indent=1)
        jr = st.median(r["steady_state_energy_per_token"] for r in reps)
        cfg = dict(model_id=mid, nickname=nick, task=ds, gpu_model="AMD Ryzen AI Max+ 395 (Strix Halo, Radeon 8060S gfx1151, 128 GB)", num_gpus=1,
                   total_params_billions=tb, activated_params_billions=ab, architecture=arch, weight_precision=prec,
                   max_num_seqs=1, max_num_batched_tokens=None, tensor_parallel=1, expert_parallel=1, data_parallel=1,
                   energy_per_token_joules=jr, energy_per_request_joules=jr, avg_output_len=1.0,
                   avg_power_watts=st.median(r["steady_state_energy"] / r["steady_state_duration"] for r in reps),
                   output_throughput_tokens_per_sec=st.median(r["output_throughput"] for r in reps),
                   request_throughput_req_per_sec=st.median(r["request_throughput"] for r in reps),
                   median_itl_ms=None, p90_itl_ms=None, p95_itl_ms=None, p99_itl_ms=None, avg_batch_size=1.0,
                   steady_state_energy_joules=sum(r["steady_state_energy"] for r in reps), steady_state_duration_seconds=sum(r["steady_state_duration"] for r in reps),
                   completed_requests=sum(r["completed"] for r in reps), total_output_tokens=sum(r["completed"] for r in reps),
                   num_request_repeats=len(reps), is_stable=len(reps) >= 2, unstable_reason="" if len(reps) >= 2 else f"{len(reps)} clean repeat(s)",
                   domain="llm", energy_scope="socket (SoC package)", net_energy_per_request_joules=st.median(r["net_energy_per_request"] for r in reps),
                   gpu_only_energy_per_request_joules=st.median(r["gpu_only_energy_per_request"] for r in reps))
        rows.append(cfg); lb[ds].append({k: v for k, v in cfg.items() if k != "task"})
    os.makedirs(OUT, exist_ok=True)
    json.dump({t: {"configurations": c, "task": t, "task_display_name": t} for t, c in lb.items()}, open(os.path.join(OUT, "leaderboard.json"), "w"), indent=1)
    with open(os.path.join(OUT, "runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows: print(f'{r["nickname"][:22]:22} {r["task"]:8} J/req {r["energy_per_request_joules"]:8.2f}  W {r["avg_power_watts"]:6.1f}  req/s {r["request_throughput_req_per_sec"]:6.2f}  reps {r["num_request_repeats"]}')
    print("wrote", OUT)
if __name__ == "__main__": main()
