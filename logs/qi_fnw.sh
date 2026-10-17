#!/bin/bash
cd ~/strix-rt
for e in ${@:-QI_FLASH_NW=8 QI_FLASH_NW=4}; do echo "== $e"; logs/qi_prof.sh $e 2>&1 | grep -E "^(swiglu|gate|up|flash|ew)|full|rror|timeout"; done
