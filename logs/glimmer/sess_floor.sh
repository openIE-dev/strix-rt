#!/bin/bash
# one lock session (perf high): power floor with GEMVs removed. GLM_SKIP 63 = all GEMV groups + attention (wrong output)
cd ~/strix-rt; M=$MODELS/new/glimmer/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf
for s in 63 15 0; do GLM_SKIP=$s bash logs/glimmer/esample.sh fl$s target-glm/release/glm_run $M @logs/glimmer/p/0.ids 1025 | sed "s/^/skip$s /" | cut -c1-200; done
