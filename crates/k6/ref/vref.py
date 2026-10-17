# Reference HunyuanVideo VAE decode on a latent crop, fp32 on CPU, with stage dumps (channel-last [T,H,W,C]).
import sys, numpy as np, torch
from diffusers import AutoencoderKLHunyuanVideo
out = sys.argv[1]; T, H, W = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
lat = np.load("~/k6_ref/out/latents.npy").astype(np.float32)[:, :, :T, :H, :W].copy()
np.save(f"{out}/vcrop_lat.npy", lat)
vae = AutoencoderKLHunyuanVideo.from_pretrained("$MODELS/k6-pro-distill/vae", torch_dtype=torch.float32).eval()
d = vae.decoder
def hook(name):
    def f(m, i, o): np.save(f"{out}/vref_{name}.npy", o[0].permute(1, 2, 3, 0).float().numpy())
    return f
d.conv_in.register_forward_hook(hook("conv_in")); d.mid_block.register_forward_hook(hook("mid"))
for b, ub in enumerate(d.up_blocks): ub.register_forward_hook(hook(f"up{b}"))
torch.set_num_threads(16)
with torch.no_grad():
    z = torch.from_numpy(lat) / vae.config.scaling_factor
    v = vae.decode(z, return_dict=False)[0]
np.save(f"{out}/vref_video.npy", v[0].float().numpy()); print("video", tuple(v.shape))
