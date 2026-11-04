#!/usr/bin/env python3
"""Per-layer best-of {act-aware q3, codebook q3} for Beam routed experts.
Usage: beam_requant_cb.py OUTA OUTB LUTFILE [--layers L0-L1] [--device cuda]      (default layers 20-29)
       beam_requant_cb.py L0 L1 OUTA OUTB LUTFILE [--device cuda]                 (old positional form)
Writes OUTA/Lxx.bin = act-aware uniform q3 (same format as beam_requant.py); OUTB/Lxx.bin = winner (codebook codes, or a real copy of OUTA's
file -- never a symlink). LUTFILE (rewritten, not appended) lines "layer v0..v7" for codebook layers (value = lut[q] * scale; engine: BEAM_CBLUT).
Codebook = one Lloyd 8-level codebook per layer on absmax-normalized group-32 weights, codes per weight, scale per group by clip search + weighted LS refit.
Safety: every output is written to a temp file in the target directory and atomically renamed into place; refuses if an output path or output
directory is a symlink, or if an output dir resolves to the source q3/nvfp4 dirs. Nothing is ever written through a symlink."""
import sys, os, shutil, numpy as np, torch
argv = sys.argv[1:]; dev_args = []
if "--device" in argv: i = argv.index("--device"); dev_args = argv[i:i + 2]; del argv[i:i + 2]
lay = None
if "--layers" in argv: i = argv.index("--layers"); lay = argv[i + 1]; del argv[i:i + 2]
if len(argv) == 5 and argv[0].isdigit() and argv[1].isdigit(): L0, L1, argv = int(argv[0]), int(argv[1]), argv[2:]
elif len(argv) == 3: L0, L1 = (int(x) for x in (lay or "20-29").split("-"))
else: sys.exit(__doc__)
OA, OB, LF = [os.path.expanduser(a) for a in argv]
assert 3 <= L0 <= L1 <= 51, (L0, L1)
PROT = {os.path.realpath(os.path.expanduser(p)) for p in ("~/models/new/beam/q3", "~/models/new/beam/nvfp4")}
def chkdir(d):
    if os.path.islink(d): sys.exit(f"refusing: output dir {d} is a symlink")
    os.makedirs(d, exist_ok=True)
    if os.path.realpath(d) in PROT: sys.exit(f"refusing: output dir {d} is a protected source dir")
chkdir(OA); chkdir(OB)
if os.path.islink(LF): sys.exit(f"refusing: {LF} is a symlink")
def target(p):  # validate final path; return temp path in same dir
    if os.path.islink(p): sys.exit(f"refusing: {p} is a symlink (would write through it)")
    t = os.path.join(os.path.dirname(p), f".{os.path.basename(p)}.tmp.{os.getpid()}")
    if os.path.lexists(t): os.remove(t)
    return t
# borrow deq/quant/ACT from beam_requant.py (its arg parsing would reset l0/l1 to 3..51 -- the old bug -- so we never use its l0/l1)
sys.argv = [sys.argv[0], "3", OA, "--act", "~/strix-rt/logs/beam/act"] + dev_args
exec(open(os.path.expanduser("~/strix-rt/tools/beam_requant.py")).read().split("WP = 13107200")[0])
print(f"layers {L0}-{L1} -> {OA} {OB} {LF}", flush=True)
torch.manual_seed(0)
def pack(q):  # codes [n,g,32] -> LE bitstream words [n,g,3] (code i at bit 3*i)
    n, g, _ = q.shape; qi = q.long()
    flat = ((qi.unsqueeze(-1) >> torch.arange(3, device=q.device)) & 1).reshape(n, g, 96)
    wd = flat.view(n, g, 3, 32); pw = (1 << torch.arange(32, device=q.device, dtype=torch.int64))
    return (((wd.to(torch.int64) * pw).sum(-1) & 0xffffffff)).cpu().numpy().astype(np.uint32)
def unpack(words, dev):  # inverse of pack / engine code_at: words [n,g,3] uint32 -> codes [n,g,32]
    w = torch.from_numpy(words.astype(np.int64)).to(dev); n, g, _ = w.shape
    bits = ((w.unsqueeze(-1) >> torch.arange(32, device=dev)) & 1).reshape(n, g, 96)   # stream bit b = word b//32, bit b%32
    return (bits.view(n, g, 32, 3) * torch.tensor([1, 2, 4], device=dev)).sum(-1)
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
lines = []; lft = target(LF)
for l in range(L0, L1 + 1):
    cb = fit_cb(l); fa, fb = f"{OA}/L{l:02d}.bin", f"{OB}/L{l:02d}.bin"; ta, tb = target(fa), target(fb)
    A = np.memmap(ta, dtype=np.uint8, mode="w+", shape=(256 * REC,)); Bm = np.memmap(tb, dtype=np.uint8, mode="w+", shape=(256 * REC,)); ea = eb = den = 0.0
    for e in range(256):
        ca, sa, cc, sc = [], [], [], []
        for proj in ("down", "gate", "up"):
            W = deq(l, e, proj).float(); wk = ACT[l][1] if proj == "down" else ACT[l][0]; n, k = W.shape
            c, s, rel = quant(W, wk); ca.append(c.tobytes()); sa.append(s.tobytes())
            ru = ((unpack(c, W.device).float() - 3.5) * torch.from_numpy(s.astype(np.float32)).to(W.device).unsqueeze(-1)).view(n, k)
            if e == 0:  # decode check: unweighted rel-MSE of decoded recon must equal quant()'s own
                chk = float(((W - ru) ** 2).mean() / (W ** 2).mean()); assert abs(chk - rel) <= 1e-4 + 1e-3 * rel, (l, proj, chk, rel)
            c2, s2, rc = qcb(W, wk, cb); cc.append(c2.tobytes()); sc.append(s2.tobytes())
            wv = wk[None]; ea += float((wv * (W - ru) ** 2).sum()); eb += float((wv * (W - rc) ** 2).sum()); den += float((wv * W ** 2).sum())
        A[e * REC:(e + 1) * REC] = np.frombuffer(b"".join(ca) + b"".join(sa), dtype=np.uint8); Bm[e * REC:(e + 1) * REC] = np.frombuffer(b"".join(cc) + b"".join(sc), dtype=np.uint8)
    A.flush(); Bm.flush(); del A, Bm
    win = "cb" if eb < ea else "q3"
    if win == "q3": os.remove(tb); shutil.copyfile(ta, tb)
    else: lines.append(f"{l} " + " ".join(f"{v:.7g}" for v in cb.tolist()) + "\n")
    for p in (fa, fb):
        if os.path.islink(p): sys.exit(f"refusing: {p} became a symlink")
    os.replace(ta, fa); os.replace(tb, fb)
    with open(lft, "w") as f: f.writelines(lines)
    os.replace(lft, LF); lft = target(LF)
    print(f"layer {l} act-err q3 {ea/den:.4f} cb {eb/den:.4f} -> {win}  cb={[round(v,3) for v in cb.tolist()]}", flush=True)
print("EXIT 0")
