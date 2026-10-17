#!/bin/bash
# A/B at the same source (HEAD worktrees): pf_deq fixed (target-dq) vs 280d9e2 pf_deq (target-dqold)
cd ~/strix-rt; D=logs/dq
commit() { for i in 1 2 3 4 5; do git add $D/$1.log && git -c user.name="David Charlot" -c user.email=you@example.com commit -qm "logs/dq: $1" -- $D/$1.log && return; sleep 3; done; }
r() { tag=$1; shift; bash logs/gpu_idle_run.sh $D/$tag.log "$@"; commit $tag; }
r qi512_old timeout 900 target-dqold/release/qi --golden ~/qi_ref/golden512 --out $D/qi512_old.png
r qi512_new timeout 900 target-dq/release/qi --golden ~/qi_ref/golden512 --out $D/qi512_new.png
r yue2_old timeout 900 target-dqold/release/yue2 parity --ref ~/yue_ref/out_ref
echo DONE > $D/DONE_AB
