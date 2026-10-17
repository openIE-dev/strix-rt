#!/bin/bash
# one lock session (perf high): GEMVQ_Q4P=1 (pairs, proj 16,256) vs 5 (pairs + precomputed sum(x)); 481 tokens, ids md5
cd ~/strix-rt; M=$MODELS/new/glimmer/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf
for r in 1 2; do for v in 1 5; do
  GEMVQ_Q4P=$v bash logs/glimmer/esample.sh xs bash -c "target-glm/release/glm_run $M @logs/glimmer/p/0.ids 481" | sed "s/^/Q4P=$v rep$r /" | cut -c1-200
  grep "^ids:" /tmp/esample_xs.out | md5sum | cut -c1-8
done; done
