#!/bin/bash
cd ~/strix-rt; M=$MODELS/new/glimmer/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf
for i in 0 1 2 3 4 5; do echo "== prompt $i"; FORCED=1 timeout 100 target-glm/release/glm_run $M @logs/glimmer/p/$i.ids 128 @logs/glimmer/p/$i.ref 2>&1 | grep FORCED | cut -c1-200; done
