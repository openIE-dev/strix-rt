#!/bin/bash
# E4B image parity vs llama-mtmd-cli (logs/g4/mmref.sh): greedy + teacher-forced with the image embeddings injected
cd ~/strix-rt; D=$MODELS/new/gemma4/e4b; E=$D/gemma-4-E4B-it-Q8_0.gguf; MP=$D/mmproj-gemma-4-E4B-it-Q8_0.gguf
logs/wait_quiet.sh
GPU_LOCK_TAG=g4-mm logs/gpu_lock.sh bash -c "
G4_PF=1 IMG=$MP,logs/g4/mm_test.ppm timeout 60 ./target/release/g4_run $E @logs/g4/mm_prompt_ids.txt 66 @logs/g4/mm_ref_ids.txt | grep -v ^weights | cut -c1-400
G4_PF=1 FORCED=1 IMG=$MP,logs/g4/mm_test.ppm timeout 60 ./target/release/g4_run $E @logs/g4/mm_prompt_ids.txt 65 @logs/g4/mm_ref1_ids.txt | grep -E 'FORCED|image' | cut -c1-400"
