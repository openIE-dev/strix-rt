#!/bin/bash
# logs/dm_session.sh NAME  : one model, whole session under the GPU lock (called via gpu_lock.sh) at perf level high.
cd ~/strix-rt; M=$MODELS/decision; B=./target/release/dm_energy; D=logs/dm; mkdir -p $D
NAME=$1
case $NAME in
 kev08) A="kev $M/kev-0.8b-bf16.gguf $M/kev-0.8b-head.bin crates/kev/ref/golden.bin 248061";;
 kev4b) A="kev $M/kev-4b-bf16.gguf $M/kev-4b-head.bin crates/kev/ref/4b/golden.bin 248061";;
 kev9q8) A="kev $M/kev-9b-q8_0.gguf $M/kev-9b-head.bin crates/kev/ref/9b/golden.bin 248061";;
 kev9bf) A="kev $M/kev-9b-bf16.gguf $M/kev-9b-head.bin crates/kev/ref/9b/golden.bin 248061";;
 kev27) A="kev $M/kev-27b-q8_0.gguf $M/kev-27b-head.bin crates/kev/ref/27b/golden.bin 248061";;
 laya) A="laya $M/laya-f16.gguf $M/laya/tokenizer/tokenizer.json $M/laya/rl_agent_config.json crates/laya/ref/golden.json";;
esac
modes="full state"; [ $NAME = laya ] && modes="req"
for mode in $modes; do
  if [ $NAME = laya ]; then CMD="$B $A @N@"; else CMD="$B $A @N@ $mode"; fi
  c=$(echo "$CMD" | sed "s/@N@/6/")
  ms=$( { /usr/bin/time -f %e $c; } 2>&1 | awk "/^DM/{for(i=1;i<=NF;i++) if(\$i ~ /ms\/(iter|request)/) {gsub(/\(/,\"\",\$(i-1)); v=\$(i-1)}} END{print v}")
  NHI=$(python3 -c "import math;print(max(8,math.ceil(40000/float(\"$ms\"))))"); NLO=$((NHI/4))
  echo "[$NAME $mode: calib $ms ms/iter -> NLO=$NLO NHI=$NHI]" | tee -a $D/summary.log
  for r in 1 2 3; do tools/comp/dm_energy.sh ${NAME}_${mode}_r$r $NLO $NHI "$CMD" 2>&1 | tee -a $D/summary.log; done
done
