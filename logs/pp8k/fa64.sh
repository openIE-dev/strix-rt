#!/bin/bash
cd ~/strix-rt; B=target-fa2/release; L=$MODELS/LFM2.5-24B-A2B-Q4_K_M.gguf
for c in "32 8 512 0" "32 8 512 7680" "32 8 500 3"; do echo "== fa_check64 $c"; FA_D=64 timeout 20 $B/fa_check $c 2>&1 | tail -3; done
for fa in 0 1; do echo "== LFM parity FA64=$fa"; PF_FA64=$fa REPS=1 timeout 100 $B/pf_lfm $L 512 16 2>&1 | grep -E "agreement|top5 ref|rror|panic"; done
for L2 in 512 8192; do for fa in 0 1 0 1; do echo "== LFM pp$L2 FA64=$fa $(PF_FA64=$fa NOREF=1 REPS=3 timeout 100 $B/pf_lfm $L $L2 1 2>&1 | grep ^prefill | cut -c1-120)"; done; done
echo DONE
