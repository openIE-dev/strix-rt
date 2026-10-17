#!/bin/bash
# prefill energy: differential of N x pp2048 vs N x pp512 (load/idle cancel), energy recorder 20 Hz, perf high. usage: pp_energy.sh TAG N "ENVS"
cd ~/strix-rt; T=$1; N=$2; shift 2; M=$MODELS/new/gemma4/26b-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf; O=logs/g4/ppe; mkdir -p $O
for L in 512 2048; do
 tools/energy/record.py record --hz 20 --json $O/$T.$L.json -- bash -c "for i in \$(seq $N); do env $* G4_QMMH=1 G4_PF=1 MAX_CTX=4096 ./target-g4p/release/g4_run $M rand:$L 2 2>&1 | grep prompt; done" > $O/$T.$L.out 2>&1
done
python3 - $O/$T.512.json $O/$T.2048.json $T $N <<'PY'
import json,sys
def g(f):
    d=json.load(open(f)); ch={c["fabric"]:c for c in d["breakdown"]["channels"]}; return ch["socket"]["joules"], ch["gfx"]["joules"], d.get("duration_s") or d.get("duration")
a,b=g(sys.argv[1]),g(sys.argv[2]); n=int(sys.argv[4]); tok=n*(2048-512)
print(f"PPENERGY {sys.argv[3]}: socket {a[0]:.0f}/{b[0]:.0f} J gfx {a[1]:.0f}/{b[1]:.0f} J dur {a[2]}/{b[2]} s -> {(b[0]-a[0])/tok:.4f} J/tok socket, {(b[1]-a[1])/tok:.4f} J/tok gfx over {tok} tokens")
PY
