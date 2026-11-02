"""Batch floor on POST /v1/systemone (any System One server: kev_serve or llama-server). Usage: floor_s1.py URL MODEL K NSTATES
Same 8 noul questions as floor.py; K=8 sends them in one request per state, K=1 sends 8 requests per state (8*NSTATES decisions either way)."""
import json, sys, urllib.request
sys.path.insert(0, __file__.rsplit("/", 1)[0]); from cascade_run import items
from floor_q import Q
url, model, K, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
def post(state, qs):
    b = json.dumps({"model": model, "state": state, "questions": {f"q{i}": {"type": "noul", "instructions": q} for i, q in qs}}).encode()
    return json.loads(urllib.request.urlopen(urllib.request.Request(url + "/v1/systemone", b, {"Content-Type": "application/json"}), timeout=600).read())
d = 0
for it in items(["ag_news"], 0, n):
    for i in range(0, 8, K): d += len(post(it["state"], list(enumerate(Q))[i:i + K])["answers"])
print("DONE", d, "decisions")
