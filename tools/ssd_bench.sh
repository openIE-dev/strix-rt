#!/bin/bash
# Cold NVMe read benchmarks. Drops page cache before each cold test (needs sudo).
set -x
F=$HOME/strix-fio.dat
drop() { sync; sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'; }
[ -f $F ] || fio --name=mk --filename=$F --size=16G --rw=write --bs=4M --direct=1 --ioengine=io_uring --iodepth=16 --output-format=terse >/dev/null
drop; dd if=$MODELS/q7.gguf of=/dev/null bs=16M 2>&1 | tail -1
drop; dd if=$F of=/dev/null bs=16M 2>&1 | tail -1
for cfg in "seq 1M 32 1" "seq 4M 16 1" "seq 1M 32 4" "seq 4M 32 4" "rand 2M 16 1" "rand 2M 32 4" "rand 128k 64 4" "rand 4k 128 4"; do
  set -- $cfg; rw=read; [ $1 = rand ] && rw=randread
  drop; fio --name=t --filename=$F --size=16G --rw=$rw --bs=$2 --iodepth=$3 --numjobs=$4 --group_reporting --direct=1 --ioengine=io_uring --runtime=15 --time_based=0 --output-format=normal 2>&1 | grep -E "READ:|IOPS=" | head -2
  echo "  ^ io_uring O_DIRECT $1 bs=$2 qd=$3 jobs=$4"
done
drop; fio --name=t --filename=$F --size=16G --rw=read --bs=1M --iodepth=32 --numjobs=1 --direct=0 --ioengine=io_uring --runtime=15 2>&1 | grep -E "READ:"; echo "  ^ io_uring BUFFERED seq 1M qd32"
