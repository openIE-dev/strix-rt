#!/bin/bash
cd ~/strix-rt; B=target-fa/release/fa_check
for c in "16 2 512 0" "16 2 128 0" "16 2 128 384" "24 4 512 0" "16 2 500 0" "16 2 512 0 1000" "16 2 512 512"; do echo "== $c"; timeout 20 $B $c 2>&1 | tail -3; done; echo DONE
