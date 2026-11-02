import csv,re,sys,json
d=sys.argv[1]; w=sys.argv[2]; K=int(sys.argv[3])
rows=list(csv.DictReader(open(f"{d}/w{w}.csv"))); t=[float(r["t_s"]) for r in rows]; p=[float(r["socket_w"]) for r in rows]
idle=[c for c in json.load(open(f"{d}/w{w}.json"))["idle_baseline"]["channels"] if c["fabric"]=="socket"][0]["avg_w"]
ph=[]
for k in range(1,K+1):
    for l in open(f"{d}/w{w}_c{k}.log"):
        m=re.match(r"PHASE (\w+) ([\d.]+) ([\d.]+)",l)
        if m: ph.append((k,m[1],float(m[2]),float(m[3])))
end_ep=ph[-1][3]+0.3; off=t[-1]-end_ep
def E(a,b):
    e=0
    for i in range(1,len(t)):
        lo=max(t[i-1],a+off); hi=min(t[i],b+off)
        if hi>lo: e+=p[i]*(hi-lo)
    return e
tot={}
for k,n,a,b in ph: tot.setdefault(n,[0,0]); tot[n][0]+=b-a; tot[n][1]+=E(a,b)-idle*(b-a)
cs=sum(v[1] for v in tot.values()); 
net=sum(max(0,0) for _ in [0])
j=json.load(open(f"{d}/w{w}.json"))["breakdown"]; s=[c for c in j["channels"] if c["fabric"]=="socket"][0]
print(d,w,"idle",round(idle,1),"per-clip: "+", ".join(f"{n} {v[0]/K:.2f}s {v[1]/K:.0f}J net" for n,v in tot.items()),"| phases total net %.0f J/clip, window net %.0f J/clip => load+other %.0f J"%(cs/K,s["net_joules"]/K,(s["net_joules"]-cs)/K))
