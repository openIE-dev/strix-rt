import numpy as np, os
d = "~/k6_ref/aud"
for n in ["conv_in", "mid_b1", "mid_attn", "mid_b2", "up2", "up1", "up0", "conv_pre"] + [f"ups{i}" for i in range(6)]:
    if not os.path.exists(f"{d}/a_{n}.npy"): print(n, "missing"); continue
    a = np.load(f"{d}/a_{n}.npy").ravel(); b = np.load(f"{d}/r_{n}.npy").ravel()
    if a.size != b.size: print(n, "size", a.size, b.size); continue
    print(f"{n:9} cos {a@b/np.linalg.norm(a)/np.linalg.norm(b):.6f} rel {np.linalg.norm(a-b)/np.linalg.norm(b):.3e}")
