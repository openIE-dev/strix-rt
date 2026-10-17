#!/bin/bash
# FA acceptance: teacher-forced agreement FA=0 vs FA=1 (must be within 1), 5 prompts per model
cd ~/strix-rt; B=target-fa2/release; M=$MODELS/minicpmv47/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf; Q=$MODELS/Qwen3.8-27B-UD-Q4_K_M.gguf
for d in ~/mcv_ref/ref_chess ~/mcv_ref/ref_gold ~/mcv_ref/ref_txt logs/comp/mcvref/L512 logs/comp/mcvref/L2048; do
  n=$(basename $d); PF_FA=0 PREFILL=1 timeout 90 $B/mcv_ref $M $d 48 2>/dev/null | grep "^OURS" | tr -d "OURS[]" > /tmp/fa8_$n.ids
  for fa in 0 1; do echo "== MiniCPM $n FA=$fa $(PF_FA=$fa PREFILL=1 PFORCED=/tmp/fa8_$n.ids timeout 90 $B/mcv_ref $M $d 2>&1 | grep -E "PFORCED|rror|panic")"; done
done
for c in "1 256" "2 512" "3 512" "4 768" "5 1024"; do set -- $c
  for fa in 0 1; do echo "== Q38 seed $1 L$2 FA=$fa $(PF_SEED=$1 PF_FA=$fa PF_FORCED=1 REPS=1 timeout 150 $B/pf_q38 $Q $2 32 2>&1 | grep -E "PF_FORCED|rror|panic")"; done
done
echo DONE
