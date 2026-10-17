cd ~/strix-rt
P="$(cat logs/q38_prompt_ids.txt)"
for v in 65536 32768 16384; do echo "== MTP_VOCAB=$v"; ZEROCOPY=dmabuf MTP_VOCAB=$v MTP=1,2,3 RUNS=1 ./target/release/q38 $MODELS/Qwen3.8-27B-UD-Q4_K_M.gguf "$P" 128; done
