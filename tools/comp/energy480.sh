#!/bin/bash
# tools/comp/energy.sh TAG CMD-with-@N@ : differential decode energy. Runs CMD with N=64 and N=480 under tools/energy/record.py record
# (socket = SMU average_socket_power integrated) and reports (J320 - J64) / 256 = J per decoded token (load/prefill/warm-up cancel out).
T=$1; shift; O=~/strix-rt/logs/comp/energy; mkdir -p $O
for N in 64 480; do C=$(echo "$*" | sed "s/@N@/$N/g")
  tools/energy/record.py record --hz 20 --json $O/$T.n$N.json -- bash -c "$C" > $O/$T.n$N.out 2>&1; done
python3 - $O/$T.n64.json $O/$T.n480.json $T <<'PY'
import json,sys
def g(f):
    d=json.load(open(f)); ch={c["fabric"]:c for c in d["breakdown"]["channels"]}; return ch["socket"]["joules"], ch["gfx"]["joules"], d.get("duration_s") or d.get("duration")
a,b=g(sys.argv[1]),g(sys.argv[2])
print(f"ENERGY {sys.argv[3]}: socket {a[0]:.1f} J (N=64) {b[0]:.1f} J (N=480) -> {(b[0]-a[0])/416:.3f} J/token socket, {(b[1]-a[1])/416:.3f} J/token gfx")
PY
