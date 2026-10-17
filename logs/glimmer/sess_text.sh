#!/bin/bash
# one lock session: text prompts (and the image prompt) straight into glm_run via the Rust tokenizer/ViT; ids parity vs llama refs (json:file#idx)
cd ~/strix-rt; M=$MODELS/new/glimmer/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf; R=logs/glimmer
for i in 0 1 2 3 4 5; do
  python3 -c "import json;open(\"/tmp/glm_p$i.txt\",\"w\").write(json.load(open(\"$R/prompts.json\"))[$i])"
  echo "== prompt $i"; NO_BOS=1 FORCED=1 timeout 100 target-glm/release/glm_run $M textfile:/tmp/glm_p$i.txt 128 json:$R/ref_llama.json#$i 2>&1 | grep -E "^prompt:|FORCED|rror"
  NO_BOS=1 timeout 100 target-glm/release/glm_run $M textfile:/tmp/glm_p$i.txt 128 json:$R/ref_llama.json#$i 2>&1 | grep -E "parity|^text"
done
echo "== image prompt"
IMAGE=$MODELS/new/glimmer/test_image.png FORCED=1 timeout 150 target-glm/release/glm_run $M "text:<__media__>
This image shows" 128 json:$R/ref_llama_img.json#0 2>&1 | grep -E "^image|^prompt:|FORCED|rror"
IMAGE=$MODELS/new/glimmer/test_image.png timeout 150 target-glm/release/glm_run $M "text:<__media__>
This image shows" 128 json:$R/ref_llama_img.json#0 2>&1 | grep -E "parity|^text"
