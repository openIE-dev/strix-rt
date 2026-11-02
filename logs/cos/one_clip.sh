#!/bin/bash
# one_clip.sh TAG SIZE STEPS : one fresh-process Cosmos3 clip (9 frames, CFG 6) with the fixed "red vintage car" prompt; logs to $O/TAG.log, wall to $O/TAG.wall
cd ~/strix-rt; s=$(date +%s.%N)
timeout 200 target-cos/release/cos gen --prompt "A red vintage car driving along a coastal road at sunset, with a lighthouse on a cliff in the distance" --size "$2" --frames 9 --steps "$3" --out /tmp/cos_e_$1 > "$O/$1.log" 2>&1
echo "$1 rc=$? wall $(echo "$(date +%s.%N) - $s" | bc)" >> "$O/walls.txt"
