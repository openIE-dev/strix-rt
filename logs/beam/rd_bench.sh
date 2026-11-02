#!/bin/bash
cd ~/strix-rt; O=logs/beam/rd; mkdir -p $O; rm -f $O/*.log
IDS=785,6722,315,9625,374
B=target-beam/release/beam; M=$MODELS/new/beam/nvfp4
run() { name=$1; shift; env BEAM_EVICT=1 "$@" $B $M $IDS 16 > $O/$name.log 2>&1; }
run base_t8_c30M BEAM_RD_THREADS=8 BEAM_RD_CH=30000000
run buf_t16_c2M BEAM_RD_THREADS=16 BEAM_RD_CH=2097152
run buf_t32_c1M BEAM_RD_THREADS=32 BEAM_RD_CH=1048576
run dir_t16_c4M BEAM_DIRECT=1 BEAM_RD_THREADS=16 BEAM_RD_CH=4194304
run dir_t32_c2M BEAM_DIRECT=1 BEAM_RD_THREADS=32 BEAM_RD_CH=2097152
run dir_t16_c4M_pf2 BEAM_DIRECT=1 BEAM_RD_THREADS=16 BEAM_RD_CH=4194304 BEAM_PF_DEPTH=2
run buf_t16_c2M_pf2 BEAM_RD_THREADS=16 BEAM_RD_CH=2097152 BEAM_PF_DEPTH=2
echo done > $O/done
