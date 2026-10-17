#!/bin/bash
# LFM prefill speed A/B: spd.sh LEN ROUNDS "ENV=.. ENV=.." ...  (alternating, 15 s cool-down between runs)
cd ~/strix-rt; B=target-fa3/release; L=$MODELS/LFM2.5-24B-A2B-Q4_K_M.gguf; n=$1; R=$2; shift 2
for r in $(seq $R); do for c in "$@"; do echo "== L$n [$c] $(env $c REPS=3 timeout 90 $B/pf_lfm $L $n 1 2>&1 | grep -oE "pp$n [0-9.]+ tok/s|rror.*|panic.*")"; sleep 15; done; done; echo DONE
