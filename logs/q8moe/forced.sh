#!/bin/bash
cd ~/strix-rt; FORCED=1 timeout 120 target-q8/release/mcv_ref $MODELS/bench/Qwen3.6-35B-A3B-Q8_0.gguf ~/mcv_ref/ref_q36txt 128 2>&1 | grep -E "FORCED|rror|mismatch" | tail -5
echo DONE
