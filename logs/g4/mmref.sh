#!/bin/bash
# logs/g4/mmref.sh: E4B + mmproj image reference via llama-mtmd-cli (Vulkan), prompt ids for g4_run IMG (run under gpu_lock)
cd ~/strix-rt; B=~/llama-latest/build-vulkan/bin; D=$MODELS/new/gemma4/e4b; M=$D/gemma-4-E4B-it-Q8_0.gguf; MP=$D/mmproj-gemma-4-E4B-it-Q8_0.gguf; O=logs/g4
python3 - <<'PY'
w,h=480,384
d=bytearray()
for y in range(h):
  for x in range(w):
    c=(245,245,240)
    if (x-150)**2+(y-190)**2<90**2: c=(220,30,30)
    if 290<x<430 and 120<y<260: c=(30,70,210)
    if y>330: c=(40,150,60)
    d+=bytes(c)
open('logs/g4/mm_test.ppm','wb').write(b'P6 %d %d 255\n'%(w,h)+d)
PY
timeout 200 $B/llama-mtmd-cli -m $M --mmproj $MP --image $O/mm_test.ppm -p "<__media__>Describe this image." --jinja --temp 0 -n 64 -ngl 999 --seed 1 -v > $O/mm_ref_vk.txt 2> $O/mm_ref_vk.err; echo "rc $?"
grep -iE 'image|token|slice|encod' $O/mm_ref_vk.err | tail -n 8
$B/llama-tokenize -m $M -p "<|turn>user
<|image>" --ids --log-disable 2>/dev/null | tail -1 > $O/mm_pre_ids.txt
$B/llama-tokenize -m $M -p "<image|>Describe this image.<turn|>
<|turn>model
" --ids --log-disable --no-bos 2>/dev/null | tail -1 > $O/mm_post_ids.txt
$B/llama-tokenize -m $M -f $O/mm_ref_vk.txt --ids --log-disable --no-bos 2>/dev/null | tail -1 > $O/mm_ref_ids.txt
cat $O/mm_pre_ids.txt $O/mm_post_ids.txt; cat $O/mm_ref_vk.txt
