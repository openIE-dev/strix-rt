#!/bin/bash
# batch 4: ROCm cells for Bonsai PTQ1_0 (prism ROCm 7.13 build), Bonsai energy, vLLM jump investigation
cd ~/strix-rt; R=tools/comp/run.sh; F=~/src/prism-llama
$R lb_bonsai tools/comp/lb_hip.sh $F/build-rocm713/bin hip_bonsai_PTQ1_0 $MODELS/bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf
VB_OVERRIDE=$F/build-vulkan/bin HB_OVERRIDE=$F/build-rocm713/bin $R energy4 tools/comp/energy_all.sh bonsai_PQ2_0:q38:$MODELS/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf bonsai_PTQ1_0:q38:$MODELS/bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf
$R vllmj tools/comp/vllm_jump.sh
