#!/bin/bash
cd ~/strix-rt
for e in "PF_ADIR=0" "PF_ADIR=0 PF_MI=2 PF_NJ=4 PF_WR=4" "PF_ADIR=0 PF_LS4=9" "PF_ADIR=0 PF_MI=2 PF_NJ=4 PF_WR=8"; do echo "== $e"; logs/qi_prof.sh $e 2>&1 | grep -E "^(qkv|o |swiglu|down)|full|rror"; done
