#!/bin/bash
# one lock session (perf high): GEMVQ_Q4P=5 (float pair + xs) vs 13 (integer-dot pair, q8 x); 481 tokens, ids saved
cd ~/strix-rt; M=$MODELS/new/glimmer/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf; mkdir -p logs/glimmer/idot2
for r in 1 2; do for v in 5 13; do
  GEMVQ_Q4P=$v bash logs/glimmer/esample.sh id bash -c "target-glm/release/glm_run $M @logs/glimmer/p/0.ids 481" | sed "s/^/Q4P=$v rep$r /" | cut -c1-200
  grep "^ids:" /tmp/esample_id.out > logs/glimmer/idot2/ids_q$v.txt; md5sum < logs/glimmer/idot2/ids_q$v.txt | cut -c1-8
done; done
