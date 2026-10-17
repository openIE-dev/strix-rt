#!/bin/bash
# one lock session: differential energy of the image-encode path (CPU preprocess + ViT GPU) = (E(runs_hi) - E(runs_lo)) / (hi - lo); load/warm-up cancel.
# usage: sess_imgenergy.sh <img> <lo> <hi>      socket = SMU average_socket_power integrated (not wall power)
cd ~/strix-rt/logs/n35; I=$1; LO=$2; HI=$3; B=$(basename $I | sed "s/\..*//"); O=imgenergy
for N in $LO $HI; do RUNS=$N tools/energy/record.py record --hz 20 --json $O/$B.r$N.json -- ~/strix-rt/target/release/n35img $MODELS/new/nemotron35vl/mmproj_st $I > $O/$B.r$N.out 2>&1; done
python3 imgenergy_calc.py $O $B $LO $HI
