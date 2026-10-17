#!/bin/bash
# pf_deq BF16/F16 alignment fix: parity checks with target-dq (fixed shader), each under the GPU lock
cd ~/strix-rt; T=target-dq/release; MD=$MODELS/decision; D=logs/dq
commit() { for i in 1 2 3 4 5; do git add $D/$1.log && git -c user.name="David Charlot" -c user.email=you@example.com commit -qm "logs/dq: $1" -- $D/$1.log && return; sleep 3; done; }
r() { tag=$1; shift; bash logs/gpu_idle_run.sh $D/$tag.log "$@"; commit $tag; }
P=/sys/class/drm/card1/device/power_dpm_force_performance_level
r kev08 BENCH_STATE=2048 $T/kev_parity $MD/kev-0.8b-bf16.gguf $MD/kev-0.8b-head.bin crates/kev/ref/golden.bin 248061
logs/gpu_lock.sh sh -c "echo high | sudo tee $P >/dev/null; bash logs/gpu_idle_run.sh $D/kev9bbf16.log BENCH_STATE=2048 $T/kev_parity $MD/kev-9b-bf16.gguf $MD/kev-9b-head.bin crates/kev/ref/9b/golden.bin 248061; echo auto | sudo tee $P >/dev/null"; commit kev9bbf16
r qi512 timeout 900 $T/qi --golden ~/qi_ref/golden512 --out $D/qi512.png
r yue2 timeout 900 target-dq/release/yue2 parity --ref ~/yue_ref/out_ref
echo DONE > $D/DONE
