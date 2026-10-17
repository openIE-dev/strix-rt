"""Golden outputs for Kev-0.8B (jaredpalmer/kev-0.8b @ bf75a6a8) via the upstream kev package, fp32 CPU.
Writes golden.json: per question row token ids/positions, decide/option offsets, final hidden at those offsets, logits, probs.
Also writes a merged (LoRA folded) HF checkpoint for GGUF conversion."""
import json, sys, torch, torch.nn.functional as F
from kev import checkpoint
from kev.model import rows_of
CK = sys.argv[1]; OUT = sys.argv[2]
m, tok = None, None
r = checkpoint.load(CK, "cpu", checkpoint.LoadOptions(dtype=getattr(torch, __import__("os").environ.get("KEV_DTYPE", "float32")), merge=True))
m, tok = (r[1], r[0]) if hasattr(r[0], "convert_tokens_to_ids") else (r[0], r[1])
m.eval()
recs = [
 {"state": "Hi, my order #4471 arrived with a cracked screen. I want my money back, not a replacement.",
  "questions": [{"instr": "Does the customer ask for a refund?", "options": ["yes", "no"], "label": 0},
                {"instr": "What is the main topic?", "options": ["billing", "shipping damage", "account access", "product question"], "label": 1}]},
 {"state": "Octopus\u0027s Garden", "questions": [{"instr": "The text is the title of a song by the Beatles. Who sings the lead vocal on it?",
   "options": ["John", "Paul", "George", "Ringo", "John and Paul duet"], "label": 3}]},
 {"state": "The server returned 503 for 12 minutes after the deploy at 14:02; error rate peaked at 38%.",
  "questions": [{"instr": "How severe is this incident?", "options": ["sev1", "sev2", "sev3", "sev4"], "label": 1}]},
]
out = {"model": "jaredpalmer/kev-0.8b", "temperature": float(m.head.temperature), "records": []}
with torch.no_grad():
    for rec in recs:
        enc = m.encode(tok, rec)
        S, Sp, rows = rows_of(enc)
        qs = []
        for r_ in rows:
            ids = S + r_["ids"]; pos = Sp + r_["pos"]
            h = m.lm(input_ids=torch.tensor([ids]), position_ids=torch.tensor([pos])).last_hidden_state[0].float()
            d = len(S) + r_["decide"]; oi = [len(S) + o for o in r_["opts"]]
            z = m.head(h[d], h[torch.tensor(oi)])
            qs.append({"ids": ids, "pos": pos, "decide": d, "opts": oi, "h_decide": h[d].tolist(), "h_opts": h[oi].tolist(),
                       "logits": z.tolist(), "probs": F.softmax(z, -1).tolist()})
        ref = [p.tolist() for p in m.probs(enc)]
        out["records"].append({"rec": rec, "questions": qs, "probs_upstream": ref})
        print(rec["state"][:40], [q["probs"] for q in qs])
json.dump(out, open(OUT, "w"))
torch.save({"q.weight": m.head.q.weight, "q.bias": m.head.q.bias, "k.weight": m.head.k.weight, "k.bias": m.head.k.bias,
            "temperature": torch.tensor(m.head.temperature)}, OUT.replace(".json", "_head.pt"))
