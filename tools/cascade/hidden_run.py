"""Collect tier-1 <decide> hidden states (kev_serve "hidden": true) for all eval items -> JSONL (id, ds, gold, probs, h)."""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__)); from cascade_run import items, post
url, model, out = sys.argv[1:4]
with open(out, "w") as f:
    for it in items(None, 0, 250):
        q = {"type": it["qtype"], "instructions": it["instructions"]}
        if it["qtype"] == "choice": q["criteria"] = it["options"]
        r = post(url + "/v1/systemone", {"state": it["state"], "model": model, "questions": {"q": q}, "hidden": True})
        a = r["answers"]["q"]; probs = {"yes": a["noul"], "no": 1 - a["noul"]} if it["qtype"] == "noul" else a["probabilities"]
        f.write(json.dumps(dict(id=it["id"], ds=it["ds"], gold=it["gold"], probs=probs, h=[round(x, 4) for x in r["hidden"]["q"]])) + "\n")
print("DONE", out)
