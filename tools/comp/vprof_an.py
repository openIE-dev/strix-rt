import gzip,json,sys,glob,collections
for c in ("c4","c5"):
    f=glob.glob(f"logs/comp/vprof/{c}/dp0*.gz")[0]; ev=json.load(gzip.open(f))["traceEvents"]
    k=[e for e in ev if e.get("ph")=="X" and e.get("cat") in ("kernel","gpu_op")]
    steps=[e for e in ev if e.get("ph")=="X" and "execute_model" in e.get("name","") ]
    tot=sum(e["dur"] for e in k); span=(max(e["ts"]+e["dur"] for e in k)-min(e["ts"] for e in k))
    d=collections.Counter(); n=collections.Counter()
    for e in k: d[e["name"][:90]]+=e["dur"]; n[e["name"][:90]]+=1
    print(f"== {c}: kernels {len(k)} gpu_busy {tot/1e6:.2f}s span {span/1e6:.2f}s exec_model calls {len(steps)}")
    for nm,t in d.most_common(12): print(f"  {t/1e6:7.3f}s {n[nm]:7d}  {nm}")
