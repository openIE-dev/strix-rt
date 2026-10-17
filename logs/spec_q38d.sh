cd ~/strix-rt
ZEROCOPY=dmabuf MTP=2,1,3 RUNS=3 ./target/release/q38 $MODELS/Qwen3.8-27B-UD-Q4_K_M.gguf "$(cat logs/q38_prompt_ids.txt)" 128
