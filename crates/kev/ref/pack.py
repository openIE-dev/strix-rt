"""golden.json + golden_head.pt -> golden.bin / head.bin (little-endian, dependency-free for the Rust side).
head.bin: f32 temperature, dp u32, d u32, q.w[dp*d], q.b[dp], k.w[dp*d], k.b[dp]
golden.bin: u32 nq; per q: u32 n, ids[n], u32 decide, u32 k, opts[k], f32 probs[k], logits[k], h_decide[d], h_opts[k*d]"""
import json, struct, torch, numpy as np, sys
R = sys.argv[1]
h = torch.load(f"{R}/golden_head.pt"); qw = h["q.weight"].detach().float().numpy(); dp, d = qw.shape
with open(f"{R}/head.bin", "wb") as f:
    f.write(struct.pack("<fII", float(h["temperature"]), dp, d))
    for k in ["q.weight", "q.bias", "k.weight", "k.bias"]: f.write(h[k].detach().float().numpy().astype("<f4").tobytes())
g = json.load(open(f"{R}/golden.json")); qs = [q for r in g["records"] for q in r["questions"]]
with open(f"{R}/golden.bin", "wb") as f:
    f.write(struct.pack("<I", len(qs)))
    for q in qs:
        f.write(struct.pack("<I", len(q["ids"]))); f.write(np.array(q["ids"], "<u4").tobytes())
        f.write(struct.pack("<II", q["decide"], len(q["opts"]))); f.write(np.array(q["opts"], "<u4").tobytes())
        for k in ["probs", "logits", "h_decide", "h_opts"]: f.write(np.array(q[k], "<f4").ravel().tobytes())
print("packed", len(qs))
