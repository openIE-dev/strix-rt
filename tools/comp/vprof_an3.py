import gzip,json,glob,statistics as st,collections
for c in ("c4","c5"):
    ev=json.load(gzip.open(glob.glob(f"logs/comp/vprof/{c}/dp0*.gz")[0]))["traceEvents"]
    k=[e for e in ev if e.get("ph")=="X" and e.get("cat") in ("kernel","gpu_op")]
    for pre in ("_matmul_NNT_bf16xbf16xmxfp4_32x256x128x1_swiglu","void at::native::index_elementwise"):
        g=collections.defaultdict(list)
        for e in k:
            if e["name"].startswith(pre): g[str(e["args"].get("grid"))].append(e["dur"])
        print(c,pre[:30],[(gr,len(v),round(st.median(v),1)) for gr,v in sorted(g.items(),key=lambda x:-sum(x[1]))[:5]])
