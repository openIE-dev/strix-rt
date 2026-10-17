#!/bin/bash
# one lock session (perf high): differential energy N=480-64, GEMVQ_Q4P=5 vs 13, A/B x3
cd ~/strix-rt; M=$MODELS/new/glimmer/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf; G=target-glm/release/glm_run
for r in 1 2 3; do for v in 5 13; do
  GEMVQ_Q4P=$v tools/comp/energy480.sh id${v}_r$r "$G $M @logs/glimmer/p/0.ids @N@" | tail -1
done; done
