cd ~/strix-rt
B=$MODELS/bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf
P="$(cat logs/q38_prompt_ids.txt)"
for nr in ${NRS:-1 2 4}; do for tern in ${TERNS:-8,64 4,64}; do
  echo "== GEMVQ_NBNR=$nr GEMV_TERN=$tern"
  env ZEROCOPY=dmabuf MTP_GGUF=$MODELS/Qwen3.8-27B-UD-Q4_K_M.gguf VPROF=1 GEMVQ_NBNR=$nr GEMV_TERN=$tern ./target/release/q38 $B "$P" 8 2>&1 | grep -E "verify|rror|panic"
done; done
