#!/bin/bash
# gpt-oss-20b: llama.cpp Vulkan baseline vs EAGLE3 draft (ggml-org eagle3-gpt-oss-20b-Q8_0), greedy 128 tokens
cd ~/strix-rt; V=~/llama-latest/build-vulkan/bin; G=$MODELS/bench/gpt-oss-20b-MXFP4.gguf; D=$MODELS/bench/eagle3-gpt-oss-20b-Q8_0.gguf
P='"The history of the Roman Empire can be summarized as follows:"'
tools/comp/spec.sh $V vk_gptoss_base $G "$P"
for k in 2 4; do tools/comp/spec.sh $V vk_gptoss_eagle3_n$k $G "$P" --spec-type draft-eagle3 -md $D -ngld 999 --spec-draft-n-max $k; done
