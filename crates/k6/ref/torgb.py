# video npy [3,T,H,W] in ~[-1,1] -> raw rgb24 frames (T*H*W*3) for ffmpeg; prints geometry
import sys, numpy as np
v = np.load(sys.argv[1]).astype(np.float32)
u = ((np.clip(v, -1, 1) + 1) * 127.5 + 0.5).astype(np.uint8).transpose(1, 2, 3, 0)
u.tofile(sys.argv[2]); print(u.shape)
