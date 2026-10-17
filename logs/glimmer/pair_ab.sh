#!/bin/bash
# pair_ab.sh TAG VA VB : one lock-session A/B pair of differential energy (N=480-64) GEMVQ_Q4P=VA then VB
cd ~/strix-rt; M=$MODELS/new/glimmer/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf; G=target-glm/release/glm_run
for v in $2 $3; do GEMVQ_Q4P=$v tools/comp/energy480.sh $1_q$v "$G $M @logs/glimmer/p/0.ids @N@" | tail -1
  echo "TOKS q$v $(grep -o "= [0-9.]* tok/s" logs/comp/energy/$1_q$v.n480.out | head -1)"; done
