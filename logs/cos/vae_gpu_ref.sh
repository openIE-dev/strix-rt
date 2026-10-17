#!/bin/bash
# diffusers Wan2.2 VAE decode on ROCm (bf16, as in the pipeline) alone, 256x256x9 latent, repeated in-process; energy recorder.
cd ~/strix-rt; L=logs/cos/vae_gpu_ref
echo high | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level >/dev/null
trap "echo auto | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level >/dev/null" EXIT
T0=$(date +%s.%N); REPS=3 timeout 280 tools/energy/record.py record --hz 10 --csv $L.csv -- $HOME/cosmos_ref/.venv/bin/python $HOME/cosmos_ref/vae_ref.py logs/cos/g256_run.lat.f32 3 16 16 $HOME/cosmos_ref/v256gpu cuda bf16 > $L.log 2>&1
grep -E "vae decode" $L.log; python3 logs/cos/phase_energy.py $L.csv $T0 $L.log
