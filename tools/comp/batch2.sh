#!/bin/bash
# batch 2: HIP (ROCm 7.13) llama-bench, gpt-oss-20b, energy for HIP, HIP MTP, gpt-oss eagle3
cd ~/strix-rt; R=tools/comp/run.sh; V=~/llama-latest/build-vulkan/bin; H=~/llama-latest/build-rocm713/bin; G=$MODELS/bench/gpt-oss-20b-MXFP4.gguf
$R lb_vk tools/comp/lb.sh $V vk_gptoss $G
for t in q38:Qwen3.8-27B-UD-Q4_K_M lfm:LFM2.5-24B-A2B-Q4_K_M mcv:minicpmv47/MiniCPM-V-4_7-35B-A3B-Q4_K_M q36:bench/Qwen3.6-35B-A3B-Q8_0 gptoss:bench/gpt-oss-20b-MXFP4; do
  $R lb_hip tools/comp/lb_hip.sh $H hip_${t%%:*} $MODELS/${t#*:}.gguf; done
$R energy2 tools/comp/energy_all.sh q38:none:$MODELS/Qwen3.8-27B-UD-Q4_K_M.gguf lfm:none:$MODELS/LFM2.5-24B-A2B-Q4_K_M.gguf mcv:none:$MODELS/minicpmv47/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf q36:none:$MODELS/bench/Qwen3.6-35B-A3B-Q8_0.gguf gptoss:none:$G
P="$(cat logs/q38_prompt_ids.txt)"; Q=$MODELS/Qwen3.8-27B-UD-Q4_K_M.gguf
$R spec2 bash -c "source tools/comp/rocm713.env; tools/comp/spec.sh $H hip_q38_base $Q \"$P\"; for k in 1 2 3; do tools/comp/spec.sh $H hip_q38_mtp\$k $Q \"$P\" --spec-type draft-mtp --spec-draft-n-max \$k; done"
GP="\"The history of the Roman Empire can be summarized as follows:\""
$R spec2 bash -c "tools/comp/spec.sh $V vk_gptoss_base $G ; tools/comp/spec.sh $V vk_gptoss_eagle3 $G  --spec-type draft-eagle3 -md $MODELS/bench/eagle3-gpt-oss-20b-Q8_0.gguf -ngld 999"
