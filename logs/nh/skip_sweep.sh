#!/bin/bash
cd ~/strix-rt
for k in 0 512 1024 2048 0; do GPU_LOCK_WAIT=1800 logs/wait_quiet.sh logs/gpu_lock.sh bash logs/nh/skip_one.sh $k 9>&-; done; echo DONE
