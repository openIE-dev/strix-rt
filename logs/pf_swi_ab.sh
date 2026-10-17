#!/bin/bash
# A/B: fused-SwiGLU A via shared (PF_SWI_ADIR=0, new default) vs direct (1, old); LLM prefill parity/speed + qimage
D=/sys/class/drm/card1/device; echo high | sudo tee $D/power_dpm_force_performance_level >/dev/null
cd ~/strix-rt; M=$MODELS/Qwen3.8-27B-UD-Q4_K_M.gguf
for L in 512 2048; do for a in 1 0; do echo "== pf_q38 L=$L PF_SWI_ADIR=$a"; PF_SWI_ADIR=$a target/release/pf_q38 $M $L 2>&1 | tail -6; done; done
for a in 1 0; do echo "== qi prof PF_SWI_ADIR=$a"; PF_SWI_ADIR=$a target/release/qi --size 1024 --stage prof 2>&1 | grep -E "^(qkv|o |swiglu|down|flash)|full"; done
echo "== qi 512 parity"; target/release/qi --golden ~/qi_ref/golden512 --size 512 --out /tmp/q5c.png 2>&1 | grep -E "step 39|PSNR|denoise"
echo auto | sudo tee $D/power_dpm_force_performance_level >/dev/null; sudo dmesg -T | grep timeout | tail -1 | cut -c1-30; echo ALLDONE
