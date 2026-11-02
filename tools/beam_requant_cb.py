#!/usr/bin/env python3
"""Per-layer best-of {act-aware q3, codebook q3} for Beam routed experts (layers L0..L1).
Usage: beam_requant_cb.py L0 L1 OUTA OUTB LUTFILE [--device cuda]
Writes OUTA/Lxx.bin = act-aware uniform q3 (same format as beam_requant.py), OUTB/Lxx.bin = winner (codebook codes, or symlink to OUTA),
LUTFILE lines "layer v0..v7" for codebook layers (value = lut[q] * scale; engine: BEAM_CBLUT). Codebook = one Lloyd 8-level codebook per layer
on absmax-normalized group-32 weights, codes chosen per weight, scale per group by clip search + weighted LS refit."""
import sys, os, numpy as np, torch
l0, l1, OA, OB, LF = int(sys.argv[1]), int(sys.argv[2]), *[os.path.expanduser(a) for a in sys.argv[3:6]]
rest = sys.argv[6:]
sys.argv = [sys.argv[0], "3", OA, "--act", "~/strix-rt/logs/beam/act"] + rest
src = open(os.path.expanduser("~/strix-rt/tools/beam_requant.py")).read().split("WP = 13107200")[0]
exec(src)
os.makedirs(OB, exist_ok=True); torch.manual_seed(0)
def pack(q):
    n, g, _ = q.shape; qi = q.long()
    flat = ((qi.unsqueeze(-1) >> torch.arange(3, device=q.device)) & 1).reshape(n, g, 96)
    wd = flat.view(n, g, 3, 32); pw = (1 << torch.arange(32, device=q.device, dtype=torch.int64))
    return (((wd.to(torch.int64) * pw).sum(-1) & 0xffffffff)).cpu().numpy().astype(np.uint32)
def fit_cb(l):
    xs = []
    for e in range(0, 256, 32):
        for proj in ("down", "gate", "up"):
            G = deq(l, e, proj).float().view(-1, 32); xs.append((G / G.abs().amax(-1, keepdim=True).clamp_min(1e-12)).flatten()[::7])
    x = torch.cat(xs); xs = x[torch.randperm(x.numel(), device=x.device)[:2000000]]
    c = torch.quantile(xs[:500000], torch.linspace(1 / 16, 15 / 16, 8, device=x.device))
    for _ in range(30):
        a = (xs[:, None] - c[None]).abs().argmin(1); c = torch.stack([xs[a == j].mean() if (a == j).any() else c[j] for j in range(8)])
    return c.sort().values
def qcb(W, wk, cb):
    n, k = W.shape; G = W.view(n, k // 32, 32); wg = wk.view(1, k // 32, 32); s0 = G.abs().amax(-1, keepdim=True).clamp_min(1e-12); best = None
    for clip in (1.1, 1.0, .95, .9, .85, .8, .75):
        s = s0 * clip; a = ((G / s).unsqueeze(-1) - cb).abs().argmin(-1); v = cb[a]
        s2 = ((wg * G * v).sum(-1, keepdim=True) / (wg * v * v).sum(-1, keepdim=True).clamp_min(1e-20)); s2 = torch.where(torch.isfinite(s2) & (s2 > 0), s2, s)
        s2 = s2.half().float(); e = (wg * (G - v * s2) ** 2).sum(-1, keepdim=True)
        if best is None: best = (e, s2, a)
        else: m = e < best[0]; best = (torch.where(m, e, best[0]), torch.where(m, s2, best[1]), torch.where(m, a, best[2]))
    e, s, a = best; rec = (cb[a] * s).view(n, k)
    return pack(a), s.view(n, k // 32).cpu().numpy().astype(np.float16), rec
WP = 13107200 * 3 // 8; SP = 13107200 // 32 * 2; REC = 3 * WP + 3 * SP
lut = open(LF, "a")
for l in range(l0, l1 + 1):
    cb = fit_cb(l); fa, fc = f"{OA}/L{l:02d}.bin", f"{OB}/L{l:02d}.cb.tmp"
    A = np.memmap(fa, dtype=np.uint8, mode="w+", shape=(256 * REC,)); Bm = np.memmap(fc, dtype=np.uint8, mode="w+", shape=(256 * REC,)); ea = eb = den = 0.0
    for e in range(256):
        ca, sa, cc, sc = [], [], [], []
        for proj in ("down", "gate", "up"):
            W = deq(l, e, proj).float(); wk = ACT[l][1] if proj == "down" else ACT[l][0]
            c, s, _ = quant(W, wk); ca.append(c.tobytes()); sa.append(s.tobytes())
            rq = ((torch.from_numpy(s.astype(np.float32)).to(W.device).repeat_interleave(32, 1)) * 0)  # placeholder (uniform recon computed below)
            n, k = W.shape; qi = torch.from_numpy(c.astype(np.int64)).to(W.device)  # [n,g,3] words
            bitsv = ((qi.unsqueeze(-1) >> torch.arange(32, device=W.device)) & 1).view(n, k // 32, 3, 32)
            q = (bitsv * (1 << torch.arange(3, device=W.device)).view(1, 1, 3, 1)).sum(2).float()
            ru = ((q - 3.5) * torch.from_numpy(s.astype(np.float32)).to(W.device).unsqueeze(-1)).view(n, k)
            c2, s2, rc = qcb(W, wk, cb); cc.append(c2.tobytes()); sc.append(s2.tobytes())
            wv = wk[None]; ea += float((wv * (W - ru) ** 2).sum()); eb += float((wv * (W - rc) ** 2).sum()); den += float((wv * W ** 2).sum())
        A[e * REC:(e + 1) * REC] = np.frombuffer(b"".join(ca) + b"".join(sa), dtype=np.uint8); Bm[e * REC:(e + 1) * REC] = np.frombuffer(b"".join(cc) + b"".join(sc), dtype=np.uint8)
    A.flush(); Bm.flush(); del A, Bm
    win = "cb" if eb < ea else "q3"; fb = f"{OB}/L{l:02d}.bin"
    if os.path.lexists(fb): os.remove(fb)
    if win == "cb": os.rename(fc, fb); lut.write(f"{l} " + " ".join(f"{v:.7g}" for v in cb.tolist()) + "\n"); lut.flush()
    else: os.remove(fc); os.symlink(fa, fb)
    print(f"layer {l} act-err q3 {ea/den:.4f} cb {eb/den:.4f} -> {win}  cb={[round(v,3) for v in cb.tolist()]}", flush=True)
print("EXIT 0")
