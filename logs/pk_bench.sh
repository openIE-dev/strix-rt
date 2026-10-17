#!/bin/bash
# Parakeet Redux benchmark: strix-rt (idle GPU, perf level high, median of 7) + CPU references + WER table.
cd ~/strix-rt; R=${1:-r1}; D=/sys/class/drm/card1/device; C=~/pk_ref/clips16
CL="$C/ls0.wav $C/ls1.wav $C/ls2.wav $C/ls3.wav $C/ls4.wav $C/ls5.wav $C/ls6.wav $C/ls7.wav $C/kt_ours.wav $C/kt_e2e.wav"
. logs/pf_busy.sh; waitidle; echo "[start $(date +%T)]"; echo high | sudo tee $D/power_dpm_force_performance_level >/dev/null
./target/release/pk --reps 7 $CL > logs/pk_${R}_ours.jsonl
echo auto | sudo tee $D/power_dpm_force_performance_level >/dev/null; echo "== perf level $(cat $D/power_dpm_force_performance_level)"
[ -f logs/pk_ref_times.json ] || (cd crates/parakeet/ref && ~/pk_ref/.venv/bin/python rtf_ref.py ~/strix-rt/logs/pk_ref_times.json $CL)
~/pk_ref/.venv/bin/python crates/parakeet/ref/wer.py logs/pk_${R}_ours.jsonl logs/pk_ref_times.json crates/parakeet/ref/refs.tsv
echo "[done $(date +%T)]"
