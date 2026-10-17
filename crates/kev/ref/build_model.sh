#!/bin/bash
# build_model.sh SIZE (0.8b|4b|9b): golden (PyTorch fp32 CPU) -> ref/SIZE/{golden.bin}, head -> $MODELS/decision/kev-SIZE-head.bin,
# LoRA-merged bf16 GGUF -> $MODELS/decision/kev-SIZE-bf16.gguf
set -e; S=$1; D=$MODELS/decision; R=~/strix-rt/crates/kev/ref; O=$R/$S; mkdir -p $O
B=$D/Qwen3.5-${S^^}-Base; B=${B/0.8B/0.8B}; [ $S = 0.8b ] && B=$D/Qwen3.5-0.8B-Base
cd ~/src/kev
.venv/bin/python $R/ref_kev.py $D/kev-$S $O/golden.json 2>&1 | grep -v -i "warn\|Loading\|falling back"
.venv/bin/python $R/pack.py $O && mv $O/head.bin $D/kev-$S-head.bin && mv $O/golden_head.pt $D/kev-$S-head.pt
.venv/bin/python $R/merge_lora.py $B $D/kev-$S $D/kev-$S-merged
cd ~/llama.cpp && .venv/bin/python convert_hf_to_gguf.py $D/kev-$S-merged --outtype bf16 --outfile $D/kev-$S-bf16.gguf 2>&1 | tail -1
rm -rf $D/kev-$S-merged; echo BUILT $S
