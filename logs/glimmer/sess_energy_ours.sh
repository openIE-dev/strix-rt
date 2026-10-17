#!/bin/bash
# one lock session (perf high): differential decode energy for strix-rt glimmer (N=320 minus N=64)
cd ~/strix-rt; M=$MODELS/new/glimmer/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf
tools/comp/energy.sh glimmer_ours "target-glm/release/glm_run $M @logs/glimmer/p/0.ids @N@"
