#!/usr/bin/env python3
"""Cactus-style (rotation + codebook) 3-bit probe on Beam experts vs our q3: act-weighted relative reconstruction error.
Usage: cq_probe.py [--bits 3]  (CPU; samples layers x experts)."""
import sys, os, numpy as np, torch, glob
sys.argv = [sys.argv[0], "3", "/tmp/cqp", "--device", "cpu", "--act", "~/strix-rt/logs/beam/act"] + sys.argv[1:]
src = open(os.path.expanduser("~/strix-rt/tools/beam_requant.py")).read().split("C = (2 ** bits")[0]
exec(src)
torch.manual_seed(0); B = 128
def hadamard(n):
    H = torch.ones(1, 1)
    while H.shape[0] < n: H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / n ** .5
H = hadamard(B)
def rot(W, sg):  # rotate rows along K in 128 blocks: W' = W D H  (x' = H^T D x, so W'x' = Wx)
    n, k = W.shape; return ((W * sg).view(n, k // B, B) @ H).view(n, k)
def unrot(Wr, sg):
    n, k = Wr.shape; return ((Wr.view(n, k // B, B) @ H.T).view(n, k)) * sg
L = 2 ** bits
def q_uniform(W, wk):
    n, k = W.shape; G = W.view(n, k // 32, 32); amax = G.abs().amax(-1, keepdim=True).clamp_min(1e-12); C = (L - 1) / 2
    best = None
    for clip in (1.0, .95, .9, .85, .8, .75, .7):
        s = amax * clip / C; r = (torch.clamp(torch.round(G / s + C), 0, L - 1) - C) * s; e = (wk * (G - r) ** 2).sum(-1, keepdim=True)
        best = (e, r) if best is None else (torch.minimum(best[0], e), torch.where(e < best[0], r, best[1]))
    return best[1].reshape(n, k)
def lloyd(x, iters=25):
    c = torch.quantile(x[torch.randperm(x.numel())[:200000]], torch.linspace(.5 / L, 1 - .5 / L, L))
    xs = x[torch.randperm(x.numel())[:400000]]
    for _ in range(iters):
        a = (xs[:, None] - c[None]).abs().argmin(1)
        c = torch.stack([xs[a == j].mean() if (a == j).any() else c[j] for j in range(L)])
    return c
def q_codebook(W, wk):
    n, k = W.shape; G = W.view(n, k // 32, 32); s = G.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    cb = lloyd((G / s).flatten())  # shared per-tensor codebook on absmax-normalized weights
    best = None
    for clip in (1.0, .95, .9, .85, .8):
        ss = s * clip; a = ((G / ss).unsqueeze(-1) - cb).abs().argmin(-1); r = cb[a] * ss
        # 1 LS refit of scale per group (weighted)
        num = (wk * G * cb[a]).sum(-1, keepdim=True); den = (wk * cb[a] ** 2).sum(-1, keepdim=True).clamp_min(1e-20); r2 = cb[a] * (num / den)
        for rr in (r, r2):
            e = (wk * (G - rr) ** 2).sum(-1, keepdim=True)
            best = (e, rr) if best is None else (torch.minimum(best[0], e), torch.where(e < best[0], rr, best[1]))
    return best[1].reshape(n, k)
res = {m: [] for m in ("q3flat", "q3", "rot+q3", "cb", "rot+cb")}
for l in (5, 20, 35, 50):
    wx, wh = ACT[l]
    for e in (0, 250):
        for proj in ("gate", "up", "down"):
            W = deq(l, e, proj).float(); n, k = W.shape; wv = (wx if k == 5120 else wh)[None]
            sg = (torch.randint(0, 2, (k,)) * 2 - 1).float()
            Wr = rot(W, sg); wr = wv.mean().expand(1, k)  # after rotation the diag weights are ~flat: use mean
            err = lambda Wh: float((wv * (W - Wh) ** 2).sum() / (wv * W ** 2).sum())
            res["q3"].append(err(q_uniform(W, wv.view(1, k // 32, 32).expand(n, -1, -1))))
            res["q3flat"].append(err(q_uniform(W, 1.0)))
            res["cb"].append(err(q_codebook(W, wv.view(1, k // 32, 32).expand(n, -1, -1))))
            res["rot+q3"].append(err(unrot(q_uniform(Wr, 1.0), sg)))
            res["rot+cb"].append(err(unrot(q_codebook(Wr, 1.0), sg)))
        print(l, e, {m: round(float(np.mean(v[-3:])), 4) for m, v in res.items()}, flush=True)
print("MEAN act-weighted rel err", {m: round(float(np.mean(v)), 4) for m, v in res.items()})
