#!/bin/bash
# tools/spec/driver.sh: J/token of llama.cpp speculative decoding vs plain decode (Vulkan), 2 reps, each config under gpu_lock.
cd ~/strix-rt; G=$MODELS/new/glimmer; Q=$MODELS/Qwen3.8-27B-UD-Q4_K_M.gguf; O=$MODELS/bench/gpt-oss-20b-MXFP4.gguf
S="logs/gpu_lock.sh bash tools/spec/spec_session.sh"
for rep in 1 2; do
  GPU_LOCK_TAG="spec glim $rep" $S glim_base $rep -m $G/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf
  GPU_LOCK_TAG="spec glim $rep" $S glim_dflash $rep -m $G/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf --spec-type draft-dflash -md $G/dflash-Muse-Glimmer-30B-Q4_K_M.gguf -ngld 999
  GPU_LOCK_TAG="spec q38 $rep" $S q38_base $rep -m $Q
  GPU_LOCK_TAG="spec q38 $rep" $S q38_mtp3 $rep -m $Q --spec-type draft-mtp --spec-draft-n-max 3
  GPU_LOCK_TAG="spec oss $rep" $S oss_base $rep -m $O
  GPU_LOCK_TAG="spec oss $rep" $S oss_eagle2 $rep -m $O --spec-type draft-eagle3 -md $MODELS/bench/eagle3-gpt-oss-20b-Q8_0.gguf -ngld 999 --spec-draft-n-max 2
done; echo DRIVER_DONE
