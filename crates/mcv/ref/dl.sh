#!/bin/bash
cd $MODELS/minicpmv47
G=https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-gguf/resolve/main
H=https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B/resolve/main
for i in 1 2 3 4 5 6; do
  curl -fL -sS --retry 5 -C - -o MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf $G/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf && break
  echo "retry $i rc=$?"; sleep 10
done
for f in config.json preprocessor_config.json processor_config.json tokenizer.json tokenizer_config.json generation_config.json; do
  curl -fL -sS --retry 3 -o $f $H/$f || echo "miss $f"
done
ls -la
echo DONE2
