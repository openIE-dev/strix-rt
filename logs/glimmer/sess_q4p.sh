#!/bin/bash
# one lock session (perf high): GEMVQ_Q4P=0/1 A/B/A/B: energy sample (481 decode tokens ~37 s) + id parity between variants
cd ~/strix-rt; M=$MODELS/new/glimmer/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf
for r in 1 2; do for v in 0 1; do
  GEMVQ_Q4P=$v bash logs/glimmer/esample.sh q4p$v target-glm/release/glm_run $M @logs/glimmer/p/0.ids 481 | sed "s/^/Q4P=$v rep$r /" | cut -c1-220
  grep "^ids:" /tmp/esample_q4p$v.out | md5sum | sed "s/^/Q4P=$v idsmd5 /"
done; done
