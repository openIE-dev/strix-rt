cd ~/strix-rt
B=$MODELS/bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf
P="$(cat logs/q38_prompt_ids.txt)"
for cfg in "TERN_LDS=0 GEMVQ_NBNR=1 GEMV_TERN=8,64" "TERN_LDS=1 GEMVQ_NBNR=2 GEMV_TERN=8,64" "TERN_LDS=1 GEMVQ_NBNR=2 GEMV_TERN=8,256" "TERN_LDS=1 GEMVQ_NBNR=1 GEMV_TERN=8,256" "TERN_LDS=1 GEMVQ_NBNR=4 GEMV_TERN=8,256" "TERN_LDS=1 GEMVQ_NBNR=2 GEMV_TERN=4,256" "TERN_LDS=1 GEMVQ_NBNR=2 GEMV_TERN=16,512"; do
  echo "== $cfg  (other GPU job: $(pgrep -f 'release/(pf_|npu_|tern_)|npu/target/release' >/dev/null && echo YES || echo no))"
  env ZEROCOPY=dmabuf MTP_GGUF=$MODELS/Qwen3.8-27B-UD-Q4_K_M.gguf VPROF=1 $cfg timeout 120 ./target/release/q38 $B "$P" 8 2>&1 | grep -E "verify|rror|panic"
done
