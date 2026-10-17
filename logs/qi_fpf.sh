#!/bin/bash
cd ~/strix-rt
for e in QI_FLASH_PF=0 QI_FLASH_PF=1; do echo "== $e"; logs/qi_prof.sh $e 2>&1 | grep -E "^(swiglu|flash)|full|rror|timeout"; done
