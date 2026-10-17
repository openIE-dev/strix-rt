# Wan2.2 VAE reference decode of a normalized Cosmos latent with stage capture. usage: vae_ref.py LAT.f32 T H W OUTDIR [cpu|cuda] [f32|bf16]
import sys, os, time, torch, numpy as np
from diffusers import AutoencoderKLWan
lat_path, T, H, W, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
dev = sys.argv[6] if len(sys.argv) > 6 else "cpu"; dt = torch.bfloat16 if (len(sys.argv) > 7 and sys.argv[7] == "bf16") else torch.float32
os.makedirs(out, exist_ok=True)
M = os.path.expanduser("$MODELS/new/cosmos3/Cosmos3-Nano")
vae = AutoencoderKLWan.from_pretrained(M, subfolder="vae", torch_dtype=dt).to(dev).eval()
lat = torch.from_numpy(np.fromfile(lat_path, dtype=np.float32).reshape(1, 48, T, H, W))
mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1); std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
z = (lat * std + mean).to(dev, dt)   # pipeline: latents / inv_std + mean, inv_std = 1/std
vo = {"mid": [], "up0": [], "up1": [], "up2": [], "dec": []}
d = vae.decoder
d.mid_block.register_forward_hook(lambda m, a, o: vo["mid"].append(o.float().cpu()))
for i in range(3): d.up_blocks[i].register_forward_hook((lambda k: lambda m, a, o: vo[k].append(o.float().cpu()))(f"up{i}"))
d.register_forward_hook(lambda m, a, o: vo["dec"].append(o.float().cpu()))
for rep in range(int(os.environ.get("REPS", "1"))):
    if rep: [x.clear() for x in vo.values()]
    t0 = time.time()
    with torch.no_grad(): v = vae.decode(z).sample
    if dev != "cpu": torch.cuda.synchronize()
    t1 = time.time(); print(f"PHASE vae{rep} {t0:.3f} {t1:.3f}", flush=True); print("vae decode", t1 - t0, "s", tuple(v.shape), flush=True)
for k, x in vo.items(): np.save(f"{out}/{k}.npy", torch.cat(x, 2)[0].numpy())
np.save(f"{out}/video.npy", v[0].float().cpu().numpy())
