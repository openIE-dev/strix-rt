#!/bin/bash
# tools/npu/npu_session.sh BACKEND NAME REP   (run inside logs/gpu_lock.sh)
#   BACKEND flm  : NAME = FLM tag (NPU)          BACKEND vk : NAME = gguf path (llama.cpp Vulkan iGPU)
# Differential windows (tools/energy/record.py socket, 20 Hz, idle 5 s): decisions (1 vs 4 passes over 100 test items), decode (64 vs 320 tokens x2).
cd ~/strix-rt; D=logs/npu; mkdir -p $D/e; B=$1; NM=$2; REP=$3; T=$(basename $NM .gguf | tr ':/' '__')_$B; PORT=8131; PIDS=""
trap 'kill $PIDS 2>/dev/null; wait 2>/dev/null' EXIT
log() { echo "$(date +%T) $*" | tee -a $D/session.log; }
if [ $B = flm ]; then flm serve $NM --port $PORT > $D/server_$T.log 2>&1 & PIDS=$!; MODEL=$NM
else ~/llama-latest/build-vulkan/bin/llama-server -m $NM --alias m -ngl 999 -fa on -c 4096 -np 1 --port $PORT --no-webui > $D/server_$T.log 2>&1 & PIDS=$!; MODEL=m; fi
U=http://127.0.0.1:$PORT
for i in $(seq 1 120); do curl -sf $U/v1/models >/dev/null && break; sleep 1; done
python3 tools/npu/decide_text.py $U $MODEL /dev/null 1 0 3 >/dev/null || { log "$T server failed"; exit 4; }
win() { tag=$1; shift
  python3 logs/cpubusy.py gate 1.0 120 >> $D/gate.txt || log "gate timeout $tag (continuing, not quiet)"
  b0=$(python3 logs/cpubusy.py snap)
  /usr/local/bin/tools/energy/record.py record --hz 20 --idle 5 --json $D/e/$tag.json -- "$@" > $D/e/$tag.out 2>&1
  b1=$(python3 logs/cpubusy.py snap)
  python3 -c "import sys;a=list(map(int,sys.argv[1:3]));b=list(map(int,sys.argv[3:5]));print(sys.argv[5],'%.2f'%(32*(b[0]-a[0])/(b[1]-a[1])))" $b0 $b1 $tag >> $D/busy.txt; }
[ -s $D/acc_$T.jsonl ] || python3 tools/npu/decide_text.py $U $MODEL $D/acc_$T.jsonl 1 50 250 | tee -a $D/session.log
for k in lo:1 hi:4; do n=${k#*:}; tg=${T}_dec_r${REP}.${k%:*}
  win $tg python3 tools/npu/decide_text.py $U $MODEL /dev/null $n 50 75; echo $((n*100)) > $D/e/$tg.meta; done
for k in lo:64 hi:320; do n=${k#*:}; tg=${T}_gen_r${REP}.${k%:*}
  win $tg python3 tools/npu/gen.py $U $MODEL $n 2 $D/e/$tg.meta; done
log "$T rep $REP done"
