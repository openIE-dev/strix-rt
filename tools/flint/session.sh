#!/bin/bash
# tools/flint/session.sh NAME REP [acc|energy] : run INSIDE logs/gpu_lock.sh + tools/comp/perf_high.sh (see driver.sh).
# Recursive Flint vs Qwen3.8-27B on llama-server (Vulkan). Resumable: skips outputs that exist.
#   1) accuracy: tools/cascade decision eval (1000 items, first-token letter logprobs, no thinking), once per model
#   2) J/decision: differential socket energy, 2 pass counts over the same 100 items per dataset (same method as tools/cascade e2)
#   3) J/token: differential windows; prefill = REPS fresh 2048-token prompts (lo/hi REPS), decode = 1 request x 128 vs 640 generated tokens
# Every window: quiet gate (<1.0 busy cores) + det-telemetry record --hz 20 --idle 5 (SMU socket + gfx).
cd ~/strix-rt; D=logs/flint; mkdir -p $D/acc $D/energy; C=tools/cascade/cascade_run.py; TB=tools/flint/tok_bench.py
LS=${LS:-$HOME/llama-clef/build-vulkan/bin/llama-server}; PORT=8130; N=100; WIN_MS=50000; DS="ag_news sst5 mnli boolq"
n=$1; rep=$2; mode=${3:-energy}; case $n in
  flintq41) export CASCADE_T2_FMT=flint; G=$HOME/models/new/flint/flint-int4-q4_1.gguf;;
  flintnv)  export CASCADE_T2_FMT=flint; G=$HOME/models/new/flint/flint-nvfp4.gguf;;
  qwen27)   G=$HOME/models/Qwen3.8-27B-UD-Q4_K_M.gguf;;
esac
PIDS=""; trap "kill \$PIDS 2>/dev/null; wait 2>/dev/null" EXIT
log() { echo "$(date +%T) $*" | tee -a $D/session.log; }
$LS -m $G --alias m -ngl 999 -fa on --port $PORT --no-webui -c 8192 -b 4096 -ub 2048 -np 1 --jinja > $D/server_$n.log 2>&1 & PIDS="$PIDS $!"; U=http://127.0.0.1:$PORT
for i in $(seq 1 300); do curl -sf $U/health >/dev/null && break; sleep 2; done; curl -sf $U/health >/dev/null || { log "$n server failed"; exit 4; }
python3 $C run --tier t2 --url $U --model m --lo 0 --hi 3 >/dev/null; log "$n up"
window() { tag=$1; shift
  python3 logs/cpubusy.py gate 1.0 600 >> $D/gate.txt || { log "gate timeout before $tag"; exit 3; }
  b0=$(python3 logs/cpubusy.py snap)
  /usr/local/bin/det-telemetry record --hz 20 --idle 5 --json $D/energy/$tag.json -- "$@" > $D/energy/$tag.out 2>&1
  b1=$(python3 logs/cpubusy.py snap)
  python3 -c "import sys;a=list(map(int,sys.argv[1:3]));b=list(map(int,sys.argv[3:5]));print(sys.argv[5],\"avg busy cores incl. our run (window+idle): %.2f\"%(32*(b[0]-a[0])/(b[1]-a[1])))" $b0 $b1 $tag >> $D/busy.txt; }
# 1) accuracy
for ds in $DS; do [ -s $D/acc/${n}_$ds.jsonl ] || python3 $C run --tier t2 --url $U --model m --ds $ds --out $D/acc/${n}_$ds.jsonl | tee -a $D/session.log; done
[ $mode = acc ] && { log "session $n acc done"; exit 0; }
# 3) J/token
for w in lo hi; do tag=${n}_pp_r$rep.$w; [ -s $D/energy/$tag.json ] && continue; k=4; [ $w = hi ] && k=16; window $tag python3 $TB $U pp 2048 $k; done
for w in lo hi; do tag=${n}_tg_r$rep.$w; [ -s $D/energy/$tag.json ] && continue; k=128; [ $w = hi ] && k=640; window $tag python3 $TB $U tg $k 1; done
# 2) J/decision
for ds in $DS; do [ -s $D/energy/${n}_${ds}_r$rep.hi.json ] && continue
  r=$(python3 $C run --tier t2 --url $U --model m --ds $ds --lo 0 --hi 20 | sed -n "s/.* in \([0-9.]*\) s/\1/p")
  read lo hi <<< $(python3 -c "import sys,math; s=float(sys.argv[1])/20; hi=max(2,math.ceil($WIN_MS/1000/(s*$N))); print(max(1,hi//4), hi)" $r)
  log "$n $ds r$rep passes lo=$lo hi=$hi"
  for w in lo hi; do p=${!w}; window ${n}_${ds}_r$rep.$w python3 $C run --tier t2 --url $U --model m --ds $ds --lo 0 --hi $N --passes $p; echo "$p $N" > $D/energy/${n}_${ds}_r$rep.$w.meta; done
done
log "session $n r$rep done"
