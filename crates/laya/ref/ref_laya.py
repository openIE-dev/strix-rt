"""Golden outputs for Laya (convaiinnovations/laya @ 7b928d8, ModernBERT-large + typed decision head), fp32 CPU."""
import sys, json, torch
import laya
from laya.common import DecisionModel
M, OUT = sys.argv[1], sys.argv[2]
agent = laya.load(M, device="cpu")
dm = next(v for v in vars(agent).values() if isinstance(v, DecisionModel))
dm.float().eval()
cap = []
def hook(mod, args, kwargs, out):
    kw = dict(kwargs); names = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]
    for n, a in zip(names, args): kw[n] = a
    enc = mod.encoder(input_ids=kw["input_ids"], attention_mask=kw["attention_mask"]).last_hidden_state
    cap.append({k: (v.tolist() if torch.is_tensor(v) else v) for k, v in kw.items() if k in names + ["position_ids", "option_ids"]} |
               {"logits": out[0].tolist(), "act_logits": out[1].tolist(), "enc_cls": enc[:, 0].tolist(), "enc_mean": enc.mean(1).tolist()})
dm.register_forward_hook(hook, with_kwargs=True)
qs = {"refund": {"type": "noul", "instructions": "Does the customer ask for a refund?"},
      "topic": {"type": "choice", "instructions": "What is the main topic?", "criteria": ["billing", "shipping damage", "account access", "product question"]},
      "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["none", "low", "medium", "high", "critical"]}}
states = ["Hi, my order #4471 arrived with a cracked screen. I want my money back, not a replacement.",
          "The server returned 503 for 12 minutes after the deploy at 14:02; error rate peaked at 38%.",
          " ".join(["Incident log line %d: checkout latency p99 rose to %d ms after the cache cluster failover; customers report duplicate charges on cards ending %04d." % (i, 900 + 37 * i, 1000 + 13 * i) for i in range(14)])]
res = []
with torch.no_grad():
    for s in states:
        try: r = agent.system_one(s, qs)
        except Exception as e:
            print("retry without levels:", e); qs["urgency"] = {"type": "score", "instructions": "How urgent is this?"}; r = agent.system_one(s, qs)
        res.append(r); print(json.dumps(r.get("answers", r), default=str)[:600])
json.dump({"states": states, "questions": qs, "results": res, "forward": cap}, open(OUT, "w"), default=str)
print("captured forwards", len(cap), [len(c["input_ids"]) for c in cap])
