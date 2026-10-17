#!/bin/bash
cd $MODELS/new/glimmer; B=https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF/resolve/main
for f in Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf mmproj-Muse-Glimmer-30B-Q4_K_M.gguf dflash-Muse-Glimmer-30B-Q4_K_M.gguf; do
  for i in 1 2 3 4 5 6 7 8; do curl -sL --max-time 600 -C - -o $f "$B/$f" && break; echo "$f attempt $i rc=$?"; sleep 5; done
  ls -l $f
done
echo DLDONE
