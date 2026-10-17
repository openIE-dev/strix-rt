#!/bin/bash
# tools/comp/serve_split.sh BENCH TAG KIND MODEL [args..]: one run.sh serve session (= one gpu_lock hold, server restart)
# per concurrency level, releasing the GPU lock between levels so other workers are not blocked for 14+ min.
cd ~/strix-rt; for C in ${CS:-1 2 4 8}; do CONC=$C tools/comp/run.sh serve "$@"; done
