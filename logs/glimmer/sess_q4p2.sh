#!/bin/bash
# one lock session (perf high): GEMV_PROJ lanes sweep with GEMVQ_Q4P=1, plus Q4P=0 reference at 16,128; 481 tokens
cd ~/strix-rt; M=$MODELS/new/glimmer/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf
for c in "0 32,128" "1 16,128" "1 8,128" "1 16,256" "0 16,128" "1 32,128" "0 32,128"; do set -- $c
  GEMVQ_Q4P=$1 GEMV_PROJ=$2 bash logs/glimmer/esample.sh q4pb target-glm/release/glm_run $M @logs/glimmer/p/0.ids 481 | sed "s/^/Q4P=$1 proj=$2 /" | cut -c1-200
  grep "^ids:" /tmp/esample_q4pb.out | md5sum | cut -c1-8
done
