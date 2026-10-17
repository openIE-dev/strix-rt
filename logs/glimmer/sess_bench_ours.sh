#!/bin/bash
# one lock session (perf high): strix-rt glimmer pp512 / pp2048 (chunk 512) and tg128 (prompt 9 tok), 3 runs each
cd ~/strix-rt; M=$MODELS/new/glimmer/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf
for n in 512 2048; do echo "== pp$n"; MAX_CTX=4096 PF_CHUNK=512 RUNS=3 timeout 120 target-glm/release/glm_run $M @logs/glimmer/p/pp$n.ids 2 2>&1 | grep -E "^prompt"; done
echo "== tg128"; RUNS=3 timeout 120 target-glm/release/glm_run $M @logs/glimmer/p/0.ids 129 2>&1 | grep -E "^prompt"
