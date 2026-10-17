#!/bin/bash
# one GPU session: 26B greedy+forced parity vs chat reference, then vision smoke test
cd ~/strix-rt
python3 -c "
import struct
w,h=320,240
d=bytearray()
for y in range(h):
  for x in range(w):
    d+=bytes((220,30,30) if 100<x<220 and 60<y<180 else (30,60,200))
open('logs/g4/test.ppm','wb').write(b'P6 %d %d 255\n'%(w,h)+d)"
logs/wait_quiet.sh
GPU_LOCK_TAG=g4-26bpar-vision logs/gpu_lock.sh bash -c '
RC=vk bash logs/g4/par.sh 26bc $MODELS/new/gemma4/26b-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf 64
timeout 60 ./target/release/g4_vision $MODELS/new/gemma4/e4b/mmproj-gemma-4-E4B-it-Q8_0.gguf logs/g4/test.ppm 280 /tmp/g4v_emb.f32
echo done'
