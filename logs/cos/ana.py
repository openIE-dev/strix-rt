import json,glob,sys,csv,re
d=sys.argv[1]; K=int(sys.argv[2]); out=[]
for f in sorted(glob.glob(d+"/w?.json")):
    w=f.split("/")[-1][:-5]; j=json.load(open(f)); b=j["breakdown"]
    ch={c["fabric"]:c for c in b["channels"]}; ib={c["fabric"]:c for c in j["idle_baseline"]["channels"]}
    s=ch["socket"]; sec=b["seconds"]
    print(f"{w}: window {sec:.1f} s, K={K}: socket gross {s['joules']:.0f} J ({s['avg_w']:.1f} W), idle {ib['socket']['avg_w']:.1f} W, net {s['net_joules']:.0f} J => {s['net_joules']/K:.0f} J/clip net, {s['joules']/K:.0f} J/clip gross, {sec/K:.1f} s/clip")
    out.append(s['net_joules']/K)
if out:
    out.sort(); print("net J/clip min/median/max: %.0f / %.0f / %.0f"%(out[0],out[len(out)//2],out[-1]))
