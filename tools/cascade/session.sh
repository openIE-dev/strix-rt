#!/bin/bash
# tools/cascade/session.sh KIND ARGS : run INSIDE logs/gpu_lock.sh + tools/comp/perf_high.sh (see driver.sh). Resumable: skips outputs that exist.
#   acc  T1NAME...          accuracy pass over all 1000 items for tier 1 models (no energy)
#   acc2                    tier-2 accuracy pass (llama-server Qwen3.8-27B Q4_K_M)
#   e1   T1NAME REP         tier-1 differential energy windows, per dataset
#   e2   REP                tier-2 differential energy windows, per dataset
#   val  T1NAME THR...      live cascade (tier 1 then tier 2) over the TEST items, one tools/energy/record.py window per threshold
# Per energy window: strict gate (<1.0 busy cores system-wide for 2x4 s, up to 600 s; else exit 3 so the driver retries),
# tools/energy/record.py record --hz 20 (SMU socket + gfx, idle baseline), busy-core average over the window appended to busy.txt.
cd ~/strix-rt; M=$MODELS/decision; K=target/release/kev_serve; C=tools/cascade/cascade_run.py; D=logs/cascade; mkdir -p $D/acc $D/energy $D/val
Q2=$MODELS/Qwen3.8-27B-UD-Q4_K_M.gguf; LS=~/llama-latest/build-vulkan/bin/llama-server
P1=8110; P2=8120; DS="ag_news sst5 mnli boolq"; N=100; WIN_MS=50000
kind=$1; shift; PIDS=""; trap 'kill $PIDS 2>/dev/null; wait 2>/dev/null' EXIT
log() { echo "$(date +%T) $*" | tee -a $D/session.log; }
start_t1() { # NAME [PORT] -> sets URL1 MODEL1 (PORT default P1)
  PORT=${2:-$P1}; case $1 in
    kev08) a="$M/kev-0.8b-bf16.gguf $M/kev-0.8b-head.bin"; MODEL1=kev-0.8b;;
    kev4b) a="$M/kev-4b-bf16.gguf $M/kev-4b-head.bin"; MODEL1=kev-4b;;
    kev9q8) a="$M/kev-9b-q8_0.gguf $M/kev-9b-head.bin"; MODEL1=kev-9b;;
    kev27) a="$M/kev-27b-q8_0.gguf $M/kev-27b-head.bin"; MODEL1=kev-27b;;
    laya) a="-  x"; export LAYA=$M/laya-f16.gguf; MODEL1=laya;;
    # open-weight rivals (Nov 2026): served by upstream llama-server /v1/systemone (Vulkan), not kev_serve
    d3edge) G=vllm-d3-edge-BF16.gguf;; d3lite) G=vllm-d3-lite-BF16.gguf;; d3flash) G=vllm-d3-flash-Q8_0.gguf;; pplxflash) G=pplx-flash-v1.2-conv-Q8_0.gguf;; pplx27) G=pplx-decider-v1.2-conv-Q8_0.gguf;; drexmax) G=drex-max-v1-Q8_0/drex-max-v1-Q8_0.gguf;;
  esac
  URL1=http://127.0.0.1:$PORT
  case $1 in d3edge|d3lite|d3flash|pplxflash|pplx27|drexmax) MODEL1=$1; HZ=/health; LSX=$LS; [ $1 = drexmax ] && LSX=~/llama-drex/build-vulkan/bin/llama-server
      $LSX -m $MODELS/rivals/$G --alias $1 -ngl 999 -fa on --port $PORT --no-webui -c 8192 -b 4096 -ub 4096 -np 1 > $D/server_$1.log 2>&1 & PIDS="$PIDS $!";;
    *) HZ=/healthz; $K $a 127.0.0.1:$PORT $MODEL1 > $D/server_$1.log 2>&1 & PIDS="$PIDS $!";;
  esac
  for i in $(seq 1 240); do curl -sf $URL1$HZ >/dev/null && break; sleep 1; done; curl -sf $URL1$HZ >/dev/null || { log "t1 $1 failed to start"; exit 4; }
  python3 $C run --tier t1 --url $URL1 --model $MODEL1 --lo 0 --hi 5 >/dev/null; log "tier1 $1 up"
}
start_t2() { $LS -m $Q2 --alias m -ngl 999 -fa on --port $P2 --no-webui -c 4096 -np 1 > $D/server_t2.log 2>&1 & PIDS="$PIDS $!"; URL2=http://127.0.0.1:$P2
  for i in $(seq 1 300); do curl -sf $URL2/health >/dev/null && break; sleep 2; done; curl -sf $URL2/health >/dev/null || { log "t2 failed to start"; exit 4; }
  python3 $C run --tier t2 --url $URL2 --model m --lo 0 --hi 3 >/dev/null; log "tier2 up"; }
window() { # TAG CMD... : gated, recorded window; sets WMS (wall seconds)
  tag=$1; shift
  python3 logs/cpubusy.py gate 1.0 600 >> $D/gate.txt || { log "gate timeout before $tag"; exit 3; }
  b0=$(python3 logs/cpubusy.py snap)
  /usr/local/bin/tools/energy/record.py record --hz 20 --idle 5 --json $D/energy/$tag.json -- "$@" > $D/energy/$tag.out 2>&1
  b1=$(python3 logs/cpubusy.py snap)
  python3 - $b0 $b1 $tag >> $D/busy.txt <<PY
import sys; a=list(map(int,sys.argv[1:3])); b=list(map(int,sys.argv[3:5])); print(sys.argv[5], "avg busy cores incl. our run (window+idle): %.2f" % (32*(b[0]-a[0])/(b[1]-a[1])))
PY
  uptime >> $D/load.txt
}
passes() { # TIER URL MODEL DS -> echo "LO HI" passes (calibrated so the hi window is ~WIN_MS)
  r=$(python3 $C run --tier $1 --url $2 --model $3 --ds $4 --lo 0 --hi 20 | sed -n "s/.* in \([0-9.]*\) s/\1/p"); python3 - $r $N <<PY
import sys,math; s=float(sys.argv[1])/20; hi=max(2,math.ceil($WIN_MS/1000/(s*int(sys.argv[2])))); print(max(1,hi//4), hi)
PY
}
case $kind in
  acc) for t in "$@"; do start_t1 $t; for ds in $DS; do [ -s $D/acc/${t}_$ds.jsonl ] || python3 $C run --tier t1 --url $URL1 --model $MODEL1 --ds $ds --out $D/acc/${t}_$ds.jsonl | tee -a $D/session.log; done; kill $PIDS; wait; PIDS=""; sleep 2; done;;
  acc2) start_t2; for ds in $DS; do [ -s $D/acc/t2_$ds.jsonl ] || python3 $C run --tier t2 --url $URL2 --model m --ds $ds --out $D/acc/t2_$ds.jsonl | tee -a $D/session.log; done;;
  e1) t=$1; rep=$2; start_t1 $t; for ds in $DS; do
        [ -s $D/energy/${t}_${ds}_r$rep.hi.json ] && continue
        read lo hi <<< $(passes t1 $URL1 $MODEL1 $ds); log "$t $ds r$rep passes lo=$lo hi=$hi"
        for w in lo hi; do p=${!w}; window ${t}_${ds}_r$rep.$w python3 $C run --tier t1 --url $URL1 --model $MODEL1 --ds $ds --lo 0 --hi $N --passes $p; echo "$p $N" > $D/energy/${t}_${ds}_r$rep.$w.meta; done
      done;;
  e2) rep=$1; start_t2; for ds in $DS; do
        [ -s $D/energy/t2_${ds}_r$rep.hi.json ] && continue
        read lo hi <<< $(passes t2 $URL2 m $ds); log "t2 $ds r$rep passes lo=$lo hi=$hi"
        for w in lo hi; do p=${!w}; window t2_${ds}_r$rep.$w python3 $C run --tier t2 --url $URL2 --model m --ds $ds --lo 0 --hi $N --passes $p; echo "$p $N" > $D/energy/t2_${ds}_r$rep.$w.meta; done
      done;;
  val) t=$1; shift; start_t1 $t; U1=$URL1; M1=$MODEL1; T2ARG=""
       if [ -n "$T2" ]; then start_t1 $T2 $P2; U2=$URL1; T2ARG="--t2kind t1 --t2model $MODEL1"; else start_t2; U2=$URL2; fi
       tg=${T2:-t2}
       for thr in "$@"; do [ -s $D/val/${t}_${tg}_$thr.json ] && continue
         tag=val_${t}_${tg}_$thr; window $tag python3 $C casc --thr $thr --t1url $U1 --t1model $M1 --t2url $U2 $T2ARG --lo 50 --hi 250 --out $D/val/${t}_${tg}_$thr.jsonl
         mv $D/energy/$tag.json $D/val/${t}_${tg}_$thr.json; mv $D/energy/$tag.out $D/val/${t}_${tg}_$thr.out; done;;
  b8) t=$1; rep=$2; start_t1 $t; F=$D/b8; mkdir -p $F  # 8 noul questions per state: K=8 in one request vs K=1 (8 requests); 10 vs 50 states
      for k in 8 1; do for w in lo hi; do n=10; [ $w = hi ] && n=50; tag=${t}_k${k}_r$rep.$w; [ -s $F/$tag.json ] && continue
        python3 logs/cpubusy.py gate 1.0 600 >> $D/gate.txt || { log "gate timeout before $tag"; exit 3; }; b0=$(python3 logs/cpubusy.py snap)
        /usr/local/bin/tools/energy/record.py record --hz 20 --idle 5 --json $F/$tag.json -- python3 tools/cascade/floor_s1.py $URL1 $MODEL1 $k $n > $F/$tag.out 2>&1
        b1=$(python3 logs/cpubusy.py snap); python3 -c "import sys;a=list(map(int,sys.argv[1:3]));b=list(map(int,sys.argv[3:5]));print(sys.argv[5],\"avg busy cores incl. our run (window+idle): %.2f\"%(32*(b[0]-a[0])/(b[1]-a[1])))" $b0 $b1 $tag >> $F/busy.txt
      done; done;;
esac
log "session $kind $* done"
