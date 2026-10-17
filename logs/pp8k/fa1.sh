#!/bin/bash
# FA parity + pp8192: PF_FA=0 (scalar pf_attn) vs 1 (coopmat flash attention)
cd ~/strix-rt; B=target-fa/release; Q=$MODELS/Qwen3.8-27B-UD-Q4_K_M.gguf; M=$MODELS/minicpmv47/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf
echo "== q38 parity L512"; PF_FA=1 timeout 200 $B/pf_q38 $Q 512 16 2>&1 | tail -4
for fa in 0 1; do echo "== FA=$fa"
 PF_FA=$fa PREFILL=1 REPS=1 timeout 200 $B/mcv_ref $M logs/comp/mcvref/L512 2 2>&1 | grep -E "MATCH|prefill|rror|panic" | head -4
 PF_FA=$fa PF_TS=1 PREFILL=1 REPS=2 MAX_CTX=8400 timeout 200 $B/mcv_ref $M logs/comp/mcvref/L8192 2 2>&1 | grep -E "TS ms|prefill|rror|panic" | tail -4
 PF_FA=$fa NOREF=1 REPS=2 timeout 200 $B/pf_q38 $Q 8192 2>&1 | tail -2
done; echo DONE
