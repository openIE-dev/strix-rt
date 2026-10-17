#!/bin/bash
# logs/g4/esample.sh TAG CMD...: run CMD while sampling tools/energy/record.py snapshot every 0.2 s; print median-window averages of
# gfx_w, socket_w, gfxclk, fclk, uclk, gfx_activity over samples with gfx_activity > 50 (the steady part of the run)
T=$1; shift; F=/tmp/esample_$T.jsonl; : > $F
( while true; do tools/energy/record.py snapshot 2>/dev/null | tr -d '\n' >> $F; echo >> $F; sleep 0.2; done ) & SP=$!
"$@" > /tmp/esample_$T.out 2>&1
kill $SP 2>/dev/null; wait $SP 2>/dev/null
python3 - $F $T <<'PY'
import json,sys
r=[]
for l in open(sys.argv[1]):
    try: g=json.loads(l)['gm']
    except Exception: continue
    if (g.get('gfx_activity_pct') or 0) > 50: r.append(g)
k=['gfx_w','socket_w','gfxclk_mhz','fclk_mhz','uclk_mhz','gfx_activity_pct']
if r: print(f"ESAMPLE {sys.argv[2]} n={len(r)} "+' '.join(f"{x}={sum(s[x] for s in r)/len(r):.1f}" for x in k))
else: print(f"ESAMPLE {sys.argv[2]} no active samples")
PY
grep -o 'decode [0-9]* tok GPU.*tok/s |' /tmp/esample_$T.out | head -n 1
