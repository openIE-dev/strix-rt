#!/bin/bash
# Cosmos3-Nano 256x256x9, 8 steps, CFG 6: strix-rt (2 runs) vs diffusers ROCm bf16 (denoise on GPU, VAE fp32 on CPU), energy per phase from the energy recorder.
cd ~/strix-rt; L=logs/cos/bench1
echo high | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level >/dev/null
trap "echo auto | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level >/dev/null" EXIT
for r in 1 2; do T0=$(date +%s.%N); timeout 60 tools/energy/record.py record --hz 20 --csv $L.ours$r.csv -- target-cos/release/cos --golden $HOME/cosmos_ref/g256 --stage run --out $L.ours$r > $L.ours$r.log 2>&1
  echo "== ours run $r"; grep -E "denoise|vae decode|video vs|prefix gpu" $L.ours$r.log; python3 logs/cos/phase_energy.py $L.ours$r.csv $T0 $L.ours$r.log; done
T0=$(date +%s.%N); timeout 170 tools/energy/record.py record --hz 20 --csv $L.ref.csv -- $HOME/cosmos_ref/.venv/bin/python logs/cos/ref_bench.py 256 256 9 8 vae_cpu > $L.ref.log 2>&1
echo "== diffusers"; grep -E "^denoise|PHASE" $L.ref.log; python3 logs/cos/phase_energy.py $L.ref.csv $T0 $L.ref.log
