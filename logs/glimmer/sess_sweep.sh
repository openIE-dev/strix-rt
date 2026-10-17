#!/bin/bash
# one lock session (perf high): GEMV_PROJ lanes-per-row,workgroup sweep, 192 decode tokens
cd ~/strix-rt; M=$MODELS/new/glimmer/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf
for v in 32,128 16,128 8,128 4,128 32,256 16,256 8,256 32,64 32,128; do GEMV_PROJ=$v bash logs/glimmer/esample.sh sw${v/,/_} target-glm/release/glm_run $M @logs/glimmer/p/0.ids 193 | sed "s/^/proj $v /" | cut -c1-200; done
