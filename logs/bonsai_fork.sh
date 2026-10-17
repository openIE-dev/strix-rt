cd ~/strix-rt; F=~/src/prism-llama; PR="The history of the Roman Empire can be summarized as follows:"
for q in PQ2_0 PTQ1_0; do
  M=$MODELS/bonsai2/Ternary-Bonsai-2-27B-$q.gguf
  $F/build-vulkan/bin/llama-bench -m $M -ngl 99 -p 512 -n 128 -r 3 > logs/bonsai_fork_vulkan_$q.log 2>&1
  $F/build-hip/bin/llama-bench -m $M -ngl 99 -p 512 -n 128 -r 3 > logs/bonsai_fork_hip_$q.log 2>&1
  $F/build-vulkan/bin/llama-simple -m $M -n 128 "$PR" > logs/bonsai_ref_vk_$q.txt 2> logs/bonsai_ref_vk_$q.err
  GGML_VK_VISIBLE_DEVICES=none $F/build-vulkan/bin/llama-simple -m $M -ngl 0 -n 128 "$PR" > logs/bonsai_ref_cpu_$q.txt 2> logs/bonsai_ref_cpu_$q.err
  $F/build-vulkan/bin/llama-tokenize -m $M -f logs/bonsai_ref_cpu_$q.txt --ids --log-disable 2>/dev/null | tail -1 > logs/bonsai_ref_cpu_${q}_ids.txt
  $F/build-vulkan/bin/llama-tokenize -m $M -f logs/bonsai_ref_vk_$q.txt --ids --log-disable 2>/dev/null | tail -1 > logs/bonsai_ref_vk_${q}_ids.txt
done
echo done > logs/bonsai_fork.done
