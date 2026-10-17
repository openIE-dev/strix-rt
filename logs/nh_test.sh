#!/bin/bash
# nh_test.sh MODEL "IDS" N REFSEQ : strix-rt nemotron greedy run + teacher-forced agreement (run under wait_quiet + gpu_lock)
cd ~/strix-rt
timeout 100 target-nh/release/nh "$1" "$2" "$3" 2>&1 | grep -v "^weights"
[ -n "$4" ] && FORCE="$4" timeout 100 target-nh/release/nh "$1" "$2" 2>&1 | grep -v "^weights"
