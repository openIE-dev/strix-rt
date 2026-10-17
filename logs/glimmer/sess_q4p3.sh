#!/bin/bash
# one lock session (perf high): differential decode energy (N=480-64), base vs Q4P variants, A/B x3
cd ~/strix-rt; M=$MODELS/new/glimmer/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf; G=target-glm/release/glm_run
for r in 1 2 3; do
  tools/comp/energy480.sh base_r$r "$G $M @logs/glimmer/p/0.ids @N@" | tail -1
  GEMVQ_Q4P=1 GEMV_PROJ=16,256 tools/comp/energy480.sh q4p16_256_r$r "$G $M @logs/glimmer/p/0.ids @N@" | tail -1
  GEMVQ_Q4P=1 GEMV_PROJ=8,128 tools/comp/energy480.sh q4p8_128_r$r "$G $M @logs/glimmer/p/0.ids @N@" | tail -1
done
