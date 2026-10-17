#!/bin/bash
# one lock session (perf high): GEMVQ_Q4P 5 vs 71 (Q6_K quad kernel), 481 tokens ids md5 + tok/s, then forced parity at 71
cd ~/strix-rt; M=$MODELS/new/glimmer/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf
for v in 5 71; do GEMVQ_Q4P=$v target-glm/release/glm_run $M @logs/glimmer/p/0.ids 481 > /tmp/q6q$v.out 2>&1; echo "Q4P=$v $(grep "^ids:" /tmp/q6q$v.out | md5sum | cut -c1-8) $(grep -o "decode [0-9]* tok GPU.*tok/s |" /tmp/q6q$v.out | head -1)"; done
GEMVQ_Q4P=71 bash logs/glimmer/sess_par13.sh
