# Reference HunyuanVideo VAE decode timing on the GPU (bf16) for a latent crop: warmup + reps, median.
import sys, time, numpy as np, torch
from diffusers import AutoencoderKLHunyuanVideo
lat = np.load(sys.argv[1]).astype(np.float32); reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
vae = AutoencoderKLHunyuanVideo.from_pretrained("$MODELS/k6-pro-distill/vae", torch_dtype=torch.bfloat16).to("cuda").eval()
import os
if os.environ.get("K6_TILE") == "1": vae.enable_tiling()
z = torch.from_numpy(lat).to("cuda", torch.bfloat16) / vae.config.scaling_factor; ts = []
with torch.no_grad():
    for i in range(reps + 1):
        torch.cuda.synchronize(); t0 = time.time(); v = vae.decode(z, return_dict=False)[0]; torch.cuda.synchronize(); ts.append(time.time() - t0)
print("shape", tuple(v.shape), "times", [round(t, 3) for t in ts], "median(after warmup)", round(sorted(ts[1:])[len(ts[1:]) // 2], 3))
