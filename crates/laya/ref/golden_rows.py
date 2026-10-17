"""Per-row (unpadded) Laya golden from golden.json's captured batches: re-runs encoder + head per row in fp32 and writes
golden.bin: u32 nrows; per row: u32 L, ids[L], u32 qtype, u32 K, marker_pos[K], f32 enc[L*1024] (encoder last hidden),
f32 head_out[L*1024] (after type_emb + head layers), f32 logits[K] (raw, before temperature), f32 act[2]."""
import sys, json, struct, numpy as np, torch
import laya
from laya.common import DecisionModel
M, R = sys.argv[1], sys.argv[2]
agent = laya.load(M, device="cpu"); dm = next(v for v in vars(agent).values() if isinstance(v, DecisionModel)); dm.float().eval()
g = json.load(open(f"{R}/golden.json")); rows = []
with torch.no_grad():
    for c in g["forward"]:
        for b in range(len(c["input_ids"])):
            L = int(sum(c["attention_mask"][b])); ids = torch.tensor([c["input_ids"][b][:L]]); att = torch.ones_like(ids)
            mm = torch.tensor([c["marker_mask"][b]]); mp = torch.tensor([c["marker_pos"][b]]); qt = torch.tensor([c["qtype"][b]])
            K = int(mm.sum()); mp, mm = mp[:, :K], mm[:, :K]
            enc = dm.encoder(input_ids=ids, attention_mask=att).last_hidden_state
            h = enc + dm.type_emb(qt)[:, None, :]
            for layer in dm.head.layers: h = layer(h, src_key_padding_mask=~att.bool())
            lg, act = dm(ids, att, mp, mm, qt)
            ref = torch.tensor(c["logits"][b][:K])
            print(f"row L={L} K={K} qtype={int(qt)} |logits-batched| {float((lg[0]-ref).abs().max()):.2e}")
            rows.append((ids[0].tolist(), int(qt), mp[0].tolist(), enc[0].numpy(), h[0].numpy(), lg[0].numpy(), act[0].numpy()))
with open(f"{R}/golden.bin", "wb") as f:
    f.write(struct.pack("<I", len(rows)))
    for ids, qt, mp, enc, h, lg, act in rows:
        f.write(struct.pack("<I", len(ids))); f.write(np.array(ids, "<u4").tobytes()); f.write(struct.pack("<II", qt, len(mp))); f.write(np.array(mp, "<u4").tobytes())
        for a in (enc, h, lg, act): f.write(np.asarray(a, "<f4").ravel().tobytes())
print("rows", len(rows))
