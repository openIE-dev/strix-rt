"""Plain causal HF fp32 slot logits for the items of a ref.jsonl (diagnostic: separates engine drift from the bidir mask).
Usage: causal_ref.py MODEL_DIR REF.jsonl OUT.jsonl"""
import json, sys, torch
from transformers import AutoModelForCausalLM
torch.set_num_threads(8)
m = AutoModelForCausalLM.from_pretrained(sys.argv[1], dtype=torch.float32).eval()
L = [json.loads(l) for l in open(sys.argv[2])]; meta, its = L[0], L[1:]
with open(sys.argv[3], "w") as f:
    f.write(json.dumps(meta) + "\n")
    for it in its:
        ids = torch.tensor([it["prefix"] + it["suffix"]])
        with torch.no_grad(): lg = m(input_ids=ids).logits[0, -1]
        n = len(it["logits"]); it = dict(it, logits=[float(lg[s]) for s in meta["slot_ids"][:n]]); f.write(json.dumps(it) + "\n")
