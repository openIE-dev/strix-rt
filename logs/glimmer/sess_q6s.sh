#!/bin/bash
# one lock session (perf high): Q6 quad lanes sweep, 321 tokens tok/s + power
cd ~/strix-rt; M=$MODELS/new/glimmer/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf
for c in "5 16,256" "71 16,256" "71 8,128" "71 32,128" "71 8,256" "5 16,256" "71 4,128"; do set -- $c
  GEMVQ_Q4P=$1 GEMV_PROJ=$2 bash logs/glimmer/esample.sh q6s target-glm/release/glm_run $M @logs/glimmer/p/0.ids 321 | sed "s/^/Q4P=$1 proj=$2 /" | cut -c1-200
done
