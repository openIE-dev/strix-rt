"""Reference outputs from Base Labs official runtime (OneJev qev @81ce62f + serve.patch), CPU fp32, sequential fork.
Per eval item: prefix/suffix token ids, slot ids, raw slot logits, calibrated probs. Usage: ref.py MODEL_DIR OUT N_PER_DS"""
import json, sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cascade")); from cascade_run import items
import torch; torch.set_num_threads(int(os.environ.get("NT", "8")))
from qev.engine import DecisionEngine
from qev.schema import SystemOneRequest
from qev.prompt import render_question
mp, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
e = DecisionEngine(mp, device="cpu", dtype=os.environ.get("DT","float32"), head_dtype=os.environ.get("DT","float32"), fork_mode="sequential")
print("info", e.info, flush=True)
with open(out, "w") as f:
    f.write(json.dumps({"meta": True, "slot_ids": e.slot_ids, "info": str(e.info)}) + "\n")
    for it in items(None, 0, n):
        q = {"type": it["qtype"], "instructions": it["instructions"]}
        if it["qtype"] == "choice": q["criteria"] = it["options"]
        req = SystemOneRequest(state=it["state"], questions={"q": q})
        rq = render_question("q", req.questions["q"], style=e.style)
        pre, suf = e._prefix_and_suffix_ids(req.state, [rq])
        t = time.time(); br = e._run_sequential(req.state, [rq])[0]; dt = time.time() - t
        resp, _ = e.decide(req)
        a = resp.answers["q"].model_dump()
        f.write(json.dumps(dict(id=it["id"], ds=it["ds"], gold=it["gold"], prefix=pre, suffix=suf[0], labels=br.labels,
                                logits=br.logits, mass=br.allowed_mass, answer=a, sec=round(dt, 2))) + "\n"); f.flush()
print("DONE", out)
