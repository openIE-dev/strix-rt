import numpy as np, sys
d = sys.argv[1]
for n in ["conv_in", "mid", "up0", "up1", "up2", "up3"]:
    try: a = np.load(f"{d}/vae_{n}.npy").ravel(); b = np.load(f"{d}/vref_{n}.npy").ravel()
    except Exception as e: print(n, e); continue
    if a.size != b.size: print(n, "size", a.size, b.size); continue
    print(f"{n:8} cos {a@b/np.linalg.norm(a)/np.linalg.norm(b):.6f} rel {np.linalg.norm(a-b)/np.linalg.norm(b):.3e}")
a = np.load(f"{d}/ours_video.npy").ravel(); b = np.load(f"{d}/vref_video.npy").ravel()
print(f"video    cos {a@b/np.linalg.norm(a)/np.linalg.norm(b):.6f} rel {np.linalg.norm(a-b)/np.linalg.norm(b):.3e} max {np.abs(a-b).max():.4f}")
