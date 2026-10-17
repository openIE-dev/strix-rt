cd ~/strix-rt
P="$(cat logs/q38_prompt_ids.txt)"
M=$MODELS/Qwen3.8-27B-UD-Q4_K_M.gguf
echo "== Bonsai PTQ1_0 + grafted MTP"
ZEROCOPY=dmabuf MTP_GGUF=$M MTP=1,2,3 RUNS=3 ./target/release/q38 $MODELS/bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf "$P" 128
echo "== Bonsai PQ2_0 + grafted MTP"
ZEROCOPY=dmabuf MTP_GGUF=$M MTP=1,2,3 RUNS=3 ./target/release/q38 $MODELS/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf "$P" 128
echo "== Qwen3.8 Q4_K_M"
ZEROCOPY=dmabuf MTP=1,2,3 RUNS=3 ./target/release/q38 $M "$P" 128
