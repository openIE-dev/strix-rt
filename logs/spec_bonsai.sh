cd ~/strix-rt
P="$(cat logs/q38_prompt_ids.txt)"
M=$MODELS/Qwen3.8-27B-UD-Q4_K_M.gguf
for q in PTQ1_0 PQ2_0; do
echo "== Bonsai $q + grafted MTP"
ZEROCOPY=dmabuf MTP_GGUF=$M MTP=1,2,3 RUNS=3 ./target/release/q38 $MODELS/bonsai2/Ternary-Bonsai-2-27B-$q.gguf "$P" 128
ZEROCOPY=dmabuf FORCE=logs/bonsai_ref_cpu_${q}_ids.txt ./target/release/q38 $MODELS/bonsai2/Ternary-Bonsai-2-27B-$q.gguf "$P" | grep agree
done
