#!/bin/bash
cd ~/strix-rt
for e in "QI_HPAD=0" "QI_HPAD=64" "QI_HPAD=128" "QI_HPAD=256"; do echo "== $e"; logs/qi_prof.sh $e 2>&1 | grep -E "^(qkv|o |swiglu|down)|full|rror"; done
