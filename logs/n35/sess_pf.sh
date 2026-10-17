#!/bin/bash
# one lock session: batched-prefill parity (PF=1) vs llama refs
cd ~/strix-rt; M=$MODELS/new/nemotron35vl/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Q4_0.gguf; free -g | head -2
ZEROCOPY=0 STEP_CHUNK=16 MAX_CTX=256 PF=1 timeout 280 target-n35/release/n35 $M PARITY logs/n35/ref_llama.txt 2>&1 | tail -14
