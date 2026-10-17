#!/bin/bash
cd ~/strix-rt
for e in "X=1" "PF_KS=1" "PF_KS=2" "PF_LS4=5" "PF_LS4=9" "PF_SG=64"; do echo "== $e"; logs/qi_prof.sh $e 2>&1; done
