"""Per-request energy floor load: POST /v1/decisions with K questions per state. Usage: floor.py URL K NSTATES
Same 8 predicate questions; K=8 sends them in one request per state, K=1 sends 8 requests per state (8*NSTATES decisions either way)."""
import json, sys, urllib.request
sys.path.insert(0, __file__.rsplit("/", 1)[0] + "/../cascade"); from cascade_run import items
Q = ["Is this text about sports?", "Is this text about business or money?", "Is the tone negative?", "Does it mention a person by name?",
     "Is it about science or technology?", "Is it about politics?", "Is it longer than two sentences?", "Is the writer expressing an opinion?"]
url, K, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
st = items(["ag_news"], 0, n)
def post(state, qs):
    b = json.dumps({"model": "kev-0.8b", "input": state, "questions": [{"type": "predicate", "instructions": q} for q in qs]}).encode()
    return json.loads(urllib.request.urlopen(urllib.request.Request(url + "/v1/decisions", b, {"Content-Type": "application/json"})).read())
d = 0
for it in st:
    s = it["state"]
    for i in range(0, 8, K): r = post(s, Q[i:i + K]); d += len(r["answers"])
print("DONE", d, "decisions")
