#!/bin/bash
# one lock session (perf high): GLM_SKIP attribution with the Q4_K pair default, 481 tokens (1 gate/up, 2 down, 4 attention, 8 lm_head, 16 qkv, 32 o-proj; outputs wrong)
cd ~/strix-rt; M=$MODELS/new/glimmer/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf
for s in 0 1 2 4 8 16 32 0; do GLM_SKIP=$s bash logs/glimmer/esample.sh sk$s target-glm/release/glm_run $M @logs/glimmer/p/0.ids 481 | sed "s/^/skip$s /" | cut -c1-200; done
