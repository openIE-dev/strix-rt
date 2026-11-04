#!/bin/bash
# tools/flint/chain.sh : wait for the Flint downloads, then accuracy (no gate) and energy sessions for all three models.
cd ~/strix-rt; until grep -q DONE ~/models/new/flint/dl.log; do sleep 30; done
bash tools/flint/driver.sh "flintq41 1 acc" "flintnv 1 acc" "flintq41 1" "flintnv 1" "qwen27 1"
