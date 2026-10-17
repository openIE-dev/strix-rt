#!/bin/bash
# tools/comp/dm_energy.sh TAG NLO NHI CMD-with-@N@ : differential energy (J(NHI)-J(NLO))/(NHI-NLO) = J per iteration, energy recorder socket+gfx.
# Run inside gpu_lock.sh + perf_high.sh. Also prints window length of each run.
T=$1; NLO=$2; NHI=$3; shift 3; O=~/strix-rt/logs/dm/energy; mkdir -p $O
for N in $NLO $NHI; do C=$(echo "$*" | sed "s/@N@/$N/g")
  tools/energy/record.py record --hz 20 --json $O/$T.n$N.json -- bash -c "$C" > $O/$T.n$N.out 2>&1; done
python3 - $O/$T.n$NLO.json $O/$T.n$NHI.json $T $NLO $NHI <<"PY"
import json,sys
def g(f):
    d=json.load(open(f)); ch={c["fabric"]:c for c in d["breakdown"]["channels"]}; return ch["socket"]["joules"], ch["gfx"]["joules"], d.get("duration_s") or d.get("duration")
a,b=g(sys.argv[1]),g(sys.argv[2]); lo,hi=int(sys.argv[4]),int(sys.argv[5])
print(f"ENERGY {sys.argv[3]}: socket {a[0]:.1f} J ({a[2]} s) -> {b[0]:.1f} J ({b[2]} s): {(b[0]-a[0])/(hi-lo):.4f} J/iter socket, {(b[1]-a[1])/(hi-lo):.4f} J/iter gfx")
PY
