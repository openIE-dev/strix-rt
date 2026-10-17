import json,sys,re
# usage: imgenergy_calc.py <dir> <name> <lo> <hi>
d,b,lo,hi=sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4])
def g(n):
    j=json.load(open(f"{d}/{b}.r{n}.json")); ch={c["fabric"]:c for c in j["breakdown"]["channels"]}; return ch["socket"]["joules"], ch["gfx"]["joules"], j["breakdown"]["seconds"]
a,c=g(lo),g(hi)
t=open(f"{d}/{b}.r{hi}.out").read(); m=re.search(r"median ([0-9.]+) ms \(GPU",t); p=re.search(r"preprocess CPU median ([0-9.]+) ms",t)
print(f"IMGENERGY {b}: windows {a[2]:.1f}s ({lo} runs, {a[0]:.0f} J) and {c[2]:.1f}s ({hi} runs, {c[0]:.0f} J) -> {(c[0]-a[0])/(hi-lo):.2f} J/image socket, {(c[1]-a[1])/(hi-lo):.2f} J/image gfx; ViT GPU median {m.group(1) if m else None} ms, CPU preprocess median {p.group(1) if p else None} ms")
