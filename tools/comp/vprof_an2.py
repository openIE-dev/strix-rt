import gzip,json,glob,statistics as st,collections
for c in ("c4","c5"):
    ev=json.load(gzip.open(glob.glob(f"logs/comp/vprof/{c}/dp0*.gz")[0]))["traceEvents"]
    k=sorted([e for e in ev if e.get("ph")=="X" and e.get("cat") in ("kernel","gpu_op")],key=lambda e:e["ts"])
    streams=collections.Counter((e["args"].get("stream"),) for e in k)
    by=collections.defaultdict(list)
    for e in k: by[e["name"][:60]].append(e["dur"])
    print(f"== {c} streams {dict(streams)}")
    for nm in list(by)[:0]: pass
    for nm,v in sorted(by.items(),key=lambda x:-sum(x[1]))[:6]: print(f"  med {st.median(v):8.1f}us p10 {sorted(v)[len(v)//10]:8.1f} p90 {sorted(v)[9*len(v)//10]:8.1f} n {len(v):6d} {nm}")
    # overlap: kernels starting before previous ends
    ov=sum(1 for a,b in zip(k,k[1:]) if b["ts"]<a["ts"]+a["dur"]); print("  overlapping starts",ov)
    w=[e for e in k if e["name"].startswith("void wvSplitK")]
    print("  wvSplitK dims sample", [e["args"].get("grid") for e in w[:3]], [e["args"].get("block") for e in w[:2]])
