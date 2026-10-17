#!/bin/bash
# E4B audio parity vs llama-mtmd-cli: reference transcript (Vulkan), then ours greedy + teacher-forced with the
# CPU gemma4a encoder embeddings injected (AUD=, marker 999998)
cd ~/strix-rt; B=~/llama-latest/build-vulkan/bin; D=$MODELS/new/gemma4/e4b; E=$D/gemma-4-E4B-it-Q8_0.gguf; MP=$D/mmproj-gemma-4-E4B-it-Q8_0.gguf; O=logs/g4
W=${W:-$HOME/pk_ref/clips16/ls1.wav}
echo "[2, 105, 9731, 107, 98, 107, 106, 107, 105, 2364, 107, 256000, 999998, 258883, 5183, 17038, 672, 9855, 236761, 106, 107, 105, 4368, 107]" > $O/au_prompt_ids.txt
logs/wait_quiet.sh
GPU_LOCK_TAG=g4-au4 logs/gpu_lock.sh bash -c "
timeout 120 $B/llama-mtmd-cli -m $E --mmproj $MP --audio $W -p '<__media__>Transcribe this audio.' --jinja --temp 0 -n 120 -ngl 999 --seed 1 -v > $O/au4_ref_vk.txt 2> $O/au4_ref_vk.err; echo mtmd rc \$?
grep -iE 'audio|n_tokens|encod|chunk' $O/au4_ref_vk.err | grep -v '^load\|tensor' | tail -n 6
$B/llama-tokenize -m $E -f $O/au4_ref_vk.txt --ids --log-disable --no-bos 2>/dev/null | tail -1 > $O/au4_ref_ids.txt
python3 -c \"import ast;r=ast.literal_eval(open('$O/au4_ref_ids.txt').read());r=r[1:] if r and r[0]==107 else r;open('$O/au4_ref1_ids.txt','w').write(str(r))\"
cat $O/au4_ref_vk.txt; echo; cat $O/au4_ref1_ids.txt; echo
G4_PF=1 AUD=$MP,$W timeout 90 ./target/release/g4_run $E @$O/au_prompt_ids.txt 120 @$O/au4_ref1_ids.txt | grep -v ^weights | cut -c1-500
G4_PF=1 FORCED=1 AUD=$MP,$W timeout 90 ./target/release/g4_run $E @$O/au_prompt_ids.txt 120 @$O/au4_ref1_ids.txt | grep -E 'FORCED|audio' | cut -c1-500"
echo alldone
