#!/bin/bash
# one lock session: GLM_FUSE_FFN 0 vs 1, 481 tokens, ids md5 + tok/s, then forced parity with fusion on
cd ~/strix-rt; M=$MODELS/new/glimmer/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf
for v in 0 1; do GLM_FUSE_FFN=$v target-glm/release/glm_run $M @logs/glimmer/p/0.ids 481 > /tmp/fuse$v.out 2>&1; grep "^ids:" /tmp/fuse$v.out | md5sum | cut -c1-8; grep -o "decode [0-9]* tok GPU.*tok/s |" /tmp/fuse$v.out | head -1; tail -2 /tmp/fuse$v.out | cut -c1-150; done
GLM_FUSE_FFN=1 bash logs/glimmer/sess_par13.sh
