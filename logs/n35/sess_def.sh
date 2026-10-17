#!/bin/bash
# default (MTP on) vs MTP=0 plain, same binary, exact ids compare. usage: sess_def.sh PROMPT N
cd ~/strix-rt; N=${2:-128}; M=$MODELS/new/nemotron35vl/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Q4_0.gguf; P=$(cat logs/nh/${1:-p30}.ids)
run() { env ZEROCOPY=0 STEP_CHUNK=16 MAX_CTX=1024 "$@" timeout 200 target-n35/release/n35 $M "$P" $N 2>&1 | grep -E "^ids|^MTP|^prompt|rror|panick" | cut -c1-400; }
A=$(run MTP=0); B=$(run MTP=1)
echo "$A" | grep -v "^ids" ; echo "$B" | grep -v "^ids"
if [ "$(echo "$A" | grep ^ids)" == "$(echo "$B" | grep ^ids)" ] && [ -n "$(echo "$A" | grep ^ids)" ]; then echo "EXACT $1 N=${2:-128}"; else echo "MISMATCH $1"; fi
