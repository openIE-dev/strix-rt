#!/bin/sh
# round 2: e2e (ktts x2 runs of 6 reps) then LM gemv sweep, both GPU-idle gated
cd ~/strix-rt
sh logs/kt/kt_e2e.sh
sh logs/kt/lmsweep.sh
