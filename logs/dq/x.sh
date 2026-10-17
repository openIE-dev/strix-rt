#!/bin/bash
# qimage 512 parity at fbedf9e (last commit with 22.71 dB, already after 280d9e2) + fixed pf_deq
cd ~/strix-rt; D=logs/dq
bash logs/gpu_idle_run.sh $D/qi512s_fbedf9e_fix.log timeout 900 target-dqx/release/qi --golden ~/qi_ref/golden512 --size 512 --out $D/qi512_x.png
git add $D/qi512s_fbedf9e_fix.log $D/x.sh; git -c user.name="David Charlot" -c user.email=you@example.com commit -qm "logs/dq: qi512 at fbedf9e + fixed pf_deq" -- $D/qi512s_fbedf9e_fix.log $D/x.sh
# HEAD qimage (owner build in target-qi, built after the pf_deq fix)
bash logs/gpu_idle_run.sh $D/qi512s_head.log timeout 900 target-qi/release/qi --golden ~/qi_ref/golden512 --size 512 --out $D/qi512_head.png
git add $D/qi512s_head.log; git -c user.name="David Charlot" -c user.email=you@example.com commit -qm "logs/dq: qi512 HEAD" -- $D/qi512s_head.log
