#!/usr/bin/env python3
"""Re-quantize Beam NVFP4 routed experts to B-bit (2/3) symmetric group-32 quant with fp16 scales (data-free, per-group clip + 1 LS refinement).
Usage: beam_requant.py BITS OUTDIR [first_layer last_layer] [--device cpu|cuda] [--test]
Record per expert in L{LL}.bin: down codes | gate codes | up codes | down scales | gate scales | up scales
  codes: [N][K/32][BITS] uint32 little-endian bitstream (code i of a group at bit BITS*i); scales: [N][K/32] fp16. All three projections have 13,107,200 weights."""
import sys, os, numpy as np, torch
bits = int(sys.argv[1]); out = os.path.expanduser(sys.argv[2]); args = sys.argv[3:]
dev = "cuda" if "--device" not in args else args[args.index("--device") + 1]
test = "--test" in args
actdir = args[args.index("--act") + 1] if "--act" in args else None
ACT = {}  # layer -> (wx[5120], wh[2560]) mean-square activations (diag-Hessian weights), shrunk toward their mean
if actdir:
    import glob
    fs = sorted(glob.glob(os.path.expanduser(actdir) + "/*.f32")); A = np.mean([np.fromfile(f, dtype=np.float32).reshape(52, 5120 + 2560) for f in fs], 0)
    for l in range(52):
        wx, wh = torch.from_numpy(A[l, :5120]).to(dev), torch.from_numpy(A[l, 5120:]).to(dev)
        ACT[l] = tuple((w + 0.2 * w.mean()) / (1.2 * w.mean()).clamp_min(1e-20) for w in (wx, wh))
    print("act-aware weights from", len(fs), "files", flush=True)
nums = [a for a in args if a.isdigit()]
l0, l1 = (int(nums[0]), int(nums[1])) if len(nums) >= 2 else (3, 51)
D = os.path.expanduser("$MODELS/new/beam/nvfp4"); os.makedirs(out, exist_ok=True)
idx = {}
for line in open(f"{D}/beam.idx"):
    p = line.split(" ");
    if "experts" in p[0] or ".mlp.gate." in p[0]: idx[p[0]] = (p[2], int(p[3]), int(p[4]), p[1], p[5].strip())
LUT4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32, device=dev)
def e4m3(b):
    s = -1.0 if b & 128 else 1.0; e = (b >> 3) & 15; m = b & 7
    return s * (m / 512.0 if e == 0 else (1 + m / 8.0) * 2.0 ** (e - 7))
LUT8 = torch.tensor([e4m3(i) for i in range(256)], dtype=torch.float32, device=dev)
mm = {}
def raw(name):
    f, off, n, dt, shp = idx[name]
    if f not in mm: mm[f] = np.memmap(f"{D}/{f}", dtype=np.uint8, mode="r")
    return mm[f][off:off + n]
def deq(l, e, proj):
    q = lambda s: f"model.layers.{l}.mlp.experts.{e}.{proj}_proj.{s}"
    w = torch.from_numpy(np.array(raw(q("weight")))).to(dev); sc = torch.from_numpy(np.array(raw(q("weight_scale")))).to(dev)
    ws2 = float(np.frombuffer(raw(q("weight_scale_2")).tobytes(), dtype=np.float32)[0])
    n, kh = (int(x) for x in idx[q("weight")][4].split(","))
    w = w.view(n, kh).long(); v = torch.stack([LUT4[w & 15], LUT4[w >> 4]], -1).view(n, kh * 2)
    s = LUT8[sc.long()].view(n, -1).repeat_interleave(16, 1)
    return v * s * ws2
C = (2 ** bits - 1) / 2.0; QMAX = 2 ** bits - 1
def quant(W, wk=None):
    n, k = W.shape; G = W.view(n, k // 32, 32); wg = 1.0 if wk is None else wk.view(1, k // 32, 32); amax = G.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    best = None
    for clip in (1.1, 1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.5):
        s = amax * clip / C; q = torch.clamp(torch.round(G / s + C), 0, QMAX); r = (q - C) * s; err = (wg * (G - r) ** 2).sum(-1, keepdim=True)
        if best is None: best = (err, s, q)
        else: m = err < best[0]; best = (torch.where(m, err, best[0]), torch.where(m, s, best[1]), torch.where(m, q, best[2]))
    err, s, q = best
    d = q - C; s2 = ((wg * G * d).sum(-1, keepdim=True) / (wg * d * d).sum(-1, keepdim=True).clamp_min(1e-12)) # LS refinement of scale for fixed codes
    s2 = torch.where(torch.isfinite(s2) & (s2 > 0), s2, s); s16 = s2.half(); s = s16.float()
    q = torch.clamp(torch.round(G / s + C), 0, QMAX)
    rec = ((q - C) * s).view(n, k); mse = ((W - rec) ** 2).mean().item(); var = (W ** 2).mean().item()
    qi = q.long().view(n, k // 32, 32)
    bitsT = (qi.unsqueeze(-1) >> torch.arange(bits, device=W.device)) & 1          # [n,g,32,bits]
    bitsT = bitsT.reshape(n, k // 32, bits, 32)                                      # bit stream index = 32*? -> regroup below
    flat = ((qi.unsqueeze(-1) >> torch.arange(bits, device=W.device)) & 1).reshape(n, k // 32, 32 * bits)
    wd = flat.view(n, k // 32, bits, 32); pw = (1 << torch.arange(32, device=W.device, dtype=torch.int64))
    words = (wd.to(torch.int64) * pw).sum(-1) & 0xffffffff                           # [n,g,bits]
    return words.cpu().numpy().astype(np.uint32), s16.view(n, k // 32).cpu().numpy().astype(np.float16), mse / var
WP = 13107200 * bits // 8; SP = 13107200 // 32 * 2; REC = 3 * WP + 3 * SP
print(f"bits {bits} record {REC} bytes ({REC/22118400:.3f} of NVFP4 22118400)", flush=True)
for l in range(l0, l1 + 1):
    nexp = 4 if test else 256
    fn = f"{out}/L{l:02d}.bin"; tmp = f"{out}/.L{l:02d}.bin.tmp.{os.getpid()}"
    if os.path.islink(fn) or os.path.islink(out): sys.exit(f"refusing: {fn} or its dir is a symlink (would write through it)")
    arr = np.memmap(tmp, dtype=np.uint8, mode="w+", shape=(nexp * REC,)); errs = []
    for e in range(nexp):
        cod, sca = [], []
        for proj in ("down", "gate", "up"):
            c, s, r = quant(deq(l, e, proj), (ACT[l][1] if proj == "down" else ACT[l][0]) if ACT else None); cod.append(c.tobytes()); sca.append(s.tobytes()); errs.append(r)
        rec = b"".join(cod) + b"".join(sca); assert len(rec) == REC, (len(rec), REC)
        arr[e * REC:(e + 1) * REC] = np.frombuffer(rec, dtype=np.uint8)
    arr.flush(); del arr; os.replace(tmp, fn)  # atomic: never write in place / through links
    print(f"layer {l} done rel-MSE mean {np.mean(errs):.4f}", flush=True)
