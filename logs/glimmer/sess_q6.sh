#!/bin/bash
# one lock session (perf high): GEMVQ_Q4P bit0 Q4_K pairs, bit1 Q6_K aligned loads; screening 481 tokens + ids md5
cd ~/strix-rt; M=$MODELS/new/glimmer/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf
for c in "1 8,128" "3 8,128" "3 16,256" "3 16,128" "1 8,128" "3 8,128"; do set -- $c
  GEMVQ_Q4P=$1 GEMV_PROJ=$2 bash logs/glimmer/esample.sh q6 target-glm/release/glm_run $M @logs/glimmer/p/0.ids 481 | sed "s/^/Q4P=$1 proj=$2 /" | cut -c1-200
  grep "^ids:" /tmp/esample_q6.out | md5sum | cut -c1-8
done
