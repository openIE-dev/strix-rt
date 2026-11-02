#!/bin/bash
# tools/spec/spec_session.sh TAG REP SERVER-ARGS... (inside gpu_lock): llama-server Vulkan, decode energy differential 64 vs 320 tokens (x2 requests),
# tools/energy/record.py socket 20 Hz + idle 5 s, quiet gate before each window; server log kept for draft acceptance.
cd ~/strix-rt; D=logs/spec; mkdir -p $D/e; T=$1; REP=$2; shift 2; PORT=8132; PIDS=""
trap 'kill $PIDS 2>/dev/null; wait 2>/dev/null' EXIT
log() { echo "$(date +%T) $*" | tee -a $D/session.log; }
~/llama-latest/build-vulkan/bin/llama-server --alias m -ngl 999 -fa on -c 4096 -np 1 --port $PORT --no-webui "$@" > $D/server_${T}_r$REP.log 2>&1 & PIDS=$!
U=http://127.0.0.1:$PORT
for i in $(seq 1 300); do curl -sf $U/health >/dev/null && break; sleep 2; done; curl -sf $U/health >/dev/null || { log "$T server failed"; exit 4; }
python3 tools/npu/gen.py $U m 32 1 /dev/null > /dev/null || { log "$T warm failed"; exit 4; }
for k in lo:64 hi:320; do n=${k#*:}; tg=${T}_gen_r${REP}.${k%:*}
  python3 logs/cpubusy.py gate 1.0 120 >> $D/gate.txt || log "gate timeout $tg (continuing, not quiet)"
  b0=$(python3 logs/cpubusy.py snap)
  /usr/local/bin/tools/energy/record.py record --hz 20 --idle 5 --json $D/e/$tg.json -- python3 tools/npu/gen.py $U m $n 2 $D/e/$tg.meta > $D/e/$tg.out 2>&1
  b1=$(python3 logs/cpubusy.py snap)
  python3 -c "import sys;a=list(map(int,sys.argv[1:3]));b=list(map(int,sys.argv[3:5]));print(sys.argv[5],'%.2f'%(32*(b[0]-a[0])/(b[1]-a[1])))" $b0 $b1 $tg >> $D/busy.txt
done
log "$T rep $REP done: $(grep -i "accept" $D/server_${T}_r$REP.log | tail -1 | cut -c1-160)"
