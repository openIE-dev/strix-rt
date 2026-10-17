# Gemma 4 llama.cpp references: logs/g4/ref.sh <tag> <model> [n]  (run under logs/gpu_lock.sh)
cd ~/strix-rt; B=~/llama-latest/build-vulkan/bin; T=$1; M=$2; N=${3:-64}
PR="The history of the Roman Empire can be summarized as follows:"
$B/llama-tokenize -m $M -p "$PR" --ids --log-disable 2>/dev/null | tail -1 > logs/g4/${T}_prompt_ids.txt
timeout 120 $B/llama-simple -m $M -n $N "$PR" > logs/g4/${T}_ref_vk.txt 2> logs/g4/${T}_ref_vk.err
$B/llama-tokenize -m $M -f logs/g4/${T}_ref_vk.txt --ids --log-disable --no-bos 2>/dev/null | tail -1 > logs/g4/${T}_ref_vk_ids.txt
