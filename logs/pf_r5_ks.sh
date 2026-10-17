#!/bin/sh
# Round 5: cooperative K-slab (PF_KS) A/B: GEMM microbench + lfm2 correctness + pp512, perf level high
cd ~/strix-rt; D=/sys/class/drm/card1/device
Q=$MODELS/Qwen3.8-27B-UD-Q4_K_M.gguf; L=$MODELS/LFM2.5-24B-A2B-Q4_K_M.gguf
busy() { pgrep -f "release/(q38|qtest|lfm|npu_|tern_|bench|pf_)|npu/target[a-z_]*/release|llama-bench|llama-cli|build-.*/bin/" >/dev/null; }
w=0; while busy; do sleep 5; w=$((w+5)); done; echo "[waited ${w}s; start $(date +%T)]"
echo high | sudo tee $D/power_dpm_force_performance_level >/dev/null
for ks in 1 2; do echo "== PF_KS=$ks"; PF_KS=$ks ITERS=10 ./target/release/pf_gemm $Q 512; PF_KS=$ks ITERS=10 ./target/release/pf_gemm $L 512; done
for ks in 1 2; do echo "== lfm check PF_KS=$ks"; PF_KS=$ks REPS=3 ./target/release/pf_lfm $L 512 16 2>&1 | grep -E "greedy|logits|prefill|rep-to"; done
for r in 1 2 3; do for ks in 1 2; do echo "PF_KS=$ks lfm: $(PF_KS=$ks REPS=6 NOREF=1 ./target/release/pf_lfm $L 512 1 2>&1 | grep ^prefill)"; echo "PF_KS=$ks q38: $(PF_KS=$ks REPS=4 NOREF=1 ./target/release/pf_q38 $Q 512 1 2>&1 | grep ^prefill)"; done; done
echo auto | sudo tee $D/power_dpm_force_performance_level >/dev/null; echo "restored $(cat $D/power_dpm_force_performance_level); other GPU job at end: $(busy && echo YES || echo no) $(date +%T)"
