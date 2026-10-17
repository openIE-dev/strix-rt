cd ~/strix-rt; M=$MODELS/Qwen3.8-27B-UD-Q4_K_M.gguf; P="$(cat logs/q38_prompt_ids.txt)"
for c in 4,64 4,128 8,64 8,128 8,256 16,64 16,128 16,256 32,128 32,256; do
  echo "== $c tern=$(ps aux | grep -c '[t]ern_')"
  GEMV_PROJ=$c GEMV_OUT=$c GEMV_SWIGLU=$c GEMV_DOWN=$c GEMV_HEAD=$c ZEROCOPY=dmabuf RUNS=1 PROFILE=1 timeout 60 ./target/release/q38 $M "$P" 16 2>&1 | grep -E "proj|out|ffn|lm_head|total"
done
