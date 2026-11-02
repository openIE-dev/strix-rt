#!/bin/bash
# chain3: redo windows moved to energy/dirty (busy > 1.5 cores), after the running kev27 r3 driver.
cd ~/strix-rt; while pgrep -f "driver.sh e1 kev27 3" >/dev/null; do sleep 60; done
D=tools/cascade/driver.sh; $D e1 kev27 1; $D e1 kev27 2; $D e2 3
echo "$(date +%T) chain3 done" >> logs/cascade/session.log
