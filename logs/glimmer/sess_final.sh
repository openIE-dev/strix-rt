#!/bin/bash
# one lock session (perf high): default (Q4P=1, proj 16,256) parity vs llama.cpp refs, then differential energy default vs GEMVQ_Q4P=0 (old kernel, proj 32,128) x3
cd ~/strix-rt; M=$MODELS/new/glimmer/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf; G=target-glm/release/glm_run
for i in 0 1 2 3 4 5; do
  echo "== prompt $i"; FORCED=1 timeout 100 $G $M @logs/glimmer/p/$i.ids 128 @logs/glimmer/p/$i.ref 2>&1 | grep -E "FORCED"
  timeout 100 $G $M @logs/glimmer/p/$i.ids 128 @logs/glimmer/p/$i.ref 2>&1 | grep -E "parity|decode"
done
for r in 1 2 3; do
  GEMVQ_Q4P=0 tools/comp/energy480.sh fin_old_r$r "$G $M @logs/glimmer/p/0.ids @N@" | tail -1
  tools/comp/energy480.sh fin_new_r$r "$G $M @logs/glimmer/p/0.ids @N@" | tail -1
done
