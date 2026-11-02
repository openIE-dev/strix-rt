#!/bin/bash
# yue_session.sh NWIN : NWIN tools/energy/record.py windows (hz 20, idle 5 s) of one `yue2 gen` (30 s clip, full pipeline: ABC plan + semantic AR + NAR 32-step flow + VAE) each. Run under wait_quiet + gpu_lock + perf_high.
cd ~/strix-rt; O=logs/yue2_energy; mkdir -p $O
for w in $(seq 1 $1); do
  tools/energy/record.py record --hz 20 --idle 5 --json $O/w$w.json -- bash -c "s=\$(date +%s.%N); timeout 400 target-ye/release/yue2 gen --ref $HOME/yue_ref/out_ref --seed 1234 --out /tmp/yue_e$w.wav > $O/w$w.log 2>&1; echo w$w rc=\$? wall \$(echo \"\$(date +%s.%N) - \$s\" | bc) >> $O/walls.txt"
  uptime >> $O/load.txt
done
