# 26B debug session: llama.cpp CPU eval-callback dump, ROCm reference text, our per-layer residual dumps
cd ~/strix-rt; M=$MODELS/new/gemma4/26b-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf; L=~/llama-latest
PR="The history of the Roman Empire can be summarized as follows:"
GGML_VK_VISIBLE_DEVICES=none timeout 120 $L/build-vulkan/bin/llama-eval-callback -m $M -ngl 0 -p "$PR" -n 1 > logs/g4/26b_evalcb_cpu.txt 2>&1; echo "evalcb rc $?"
