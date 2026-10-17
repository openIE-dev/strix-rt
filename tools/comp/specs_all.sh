#!/bin/bash
# speculative decoding: llama.cpp MTP (Qwen3.8 nextn head) vs strix-rt MTP; same 12-token prompt ids, greedy, 128 tokens
cd ~/strix-rt; S=tools/comp/spec.sh; V=~/llama-latest/build-vulkan/bin; H=~/llama-latest/build-rocm713/bin
Q=$MODELS/Qwen3.8-27B-UD-Q4_K_M.gguf; P="$(cat logs/q38_prompt_ids.txt)"
$S $V vk_q38_base $Q "$P"
for k in 1 2 3; do $S $V vk_q38_mtp$k $Q "$P" --spec-type draft-mtp --spec-draft-n-max $k; done
( source tools/comp/rocm713.env; $S $H hip_q38_base $Q "$P"; for k in 1 2 3; do $S $H hip_q38_mtp$k $Q "$P" --spec-type draft-mtp --spec-draft-n-max $k; done )
ZEROCOPY=dmabuf MTP=1,2,3 RUNS=2 target-comp/release/q38 $Q "$P" 128 2>&1 | grep -E "baseline|MTP k"
for q in PQ2_0 PTQ1_0; do echo "bonsai $q (MTP graft from $Q)"; ZEROCOPY=dmabuf MTP_GGUF=$Q MTP=1,2 RUNS=2 target-comp/release/q38 $MODELS/bonsai2/Ternary-Bonsai-2-27B-$q.gguf "$P" 128 2>&1 | grep -E "baseline|MTP k"; done
