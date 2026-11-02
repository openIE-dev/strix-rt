#!/bin/bash
# chain2: after chain.sh finishes, Kev-27B energy (alternative tier 2), 2 repeats.
cd ~/strix-rt; while pgrep -f "bash tools/cascade/chain.sh" >/dev/null; do sleep 60; done
tools/cascade/driver.sh e1 kev27 1; tools/cascade/driver.sh e1 kev27 2
echo "$(date +%T) chain2 done" >> logs/cascade/session.log
