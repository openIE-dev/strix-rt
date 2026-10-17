#!/bin/bash
# energy_all.sh NAME:KIND:MODEL ... (KIND q38|lfm|mcv|none) -> differential J/token for llama.cpp Vulkan, HIP (ROCm 7.13) and strix-rt
cd ~/strix-rt; E=tools/comp/energy.sh; V=~/llama-latest/build-vulkan/bin; H=~/llama-latest/build-rocm713/bin
for spec in "$@"; do IFS=: read n k m <<< "$spec"; VB=${VB_OVERRIDE:-$V}; HB=${HB_OVERRIDE:-$H}
  $E vk_$n "$VB/llama-bench -m $m -ngl 999 -fa on -p 0 -n @N@ -r 1"
  $E hip_$n "source tools/comp/rocm713.env; $HB/llama-bench -m $m -ngl 999 -fa on -p 0 -n @N@ -r 1"
  case $k in
  q38|lfm) $E srt_$n "ZEROCOPY=dmabuf target-comp/release/$k $m \"\$(cat logs/${k}_prompt_ids.txt)\" @N@" ;;
  mcv) $E srt_$n "ZEROCOPY=dmabuf PREFILL=1 target-comp/release/mcv_ref $m logs/comp/mcvref/L12 @N@" ;;
  esac
done
