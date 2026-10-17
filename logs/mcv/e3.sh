#!/bin/bash
# e2e parity on 3 images: our vision + LLM vs llama.cpp mtmd greedy reference
cd ~/strix-rt; M=$MODELS/minicpmv47/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf; R=~/mcv_ref; B=target-k6/release/mcv_ref
for im in fox gold chess; do d=$R/ref_$im; [ $im = fox ] && d=$R/ref_img
  OWN_VIS=$R/$im.ppm timeout 300 $B $M $d 120 > logs/mcv/e3_${im}_greedy.log 2>&1
  OWN_VIS=$R/$im.ppm FORCED=1 timeout 300 $B $M $d > logs/mcv/e3_${im}_forced.log 2>&1
  OWN_VIS=$R/$im.ppm PREFILL=1 REPS=5 timeout 300 $B $M $d 120 > logs/mcv/e3_${im}_pf.log 2>&1
  FORCED=1 timeout 300 $B $M $d > logs/mcv/e3_${im}_refemb_forced.log 2>&1
  echo "$im done"
done
for im in fox gold chess; do timeout 300 $L -m $M --mmproj $MODELS/minicpmv47/mmproj-model-f16.gguf --image $R/$im.png -p "Describe this image in one paragraph." -n 64 --temp 0 -ngl 99 > logs/mcv/e3_${im}_mtmdcli.log 2>&1; done
echo E3DONE
