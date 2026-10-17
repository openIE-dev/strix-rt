# Timed diffusers Cosmos3-Nano reference: denoise (prompt + 2x CFG transformer + UniPC) and VAE decode, PHASE lines in epoch s.
# usage: ref_bench.py [H W F STEPS] [vae_cpu]
import sys, os, time, torch, numpy as np
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
H, W, F, S = [int(x) for x in (sys.argv[1:5] if len(sys.argv) > 4 else (256, 256, 9, 8))]
M = os.path.expanduser("$MODELS/new/cosmos3/Cosmos3-Nano")
pipe = Cosmos3OmniPipeline.from_pretrained(M, torch_dtype=torch.bfloat16, enable_safety_checker=False).to("cuda")
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=10.0)
prompt = open(f"{M}/assets/example_t2v_prompt_short.txt").read().strip()
torch.cuda.synchronize(); a = time.time()
lat = pipe(prompt=prompt, negative_prompt="", num_frames=F, height=H, width=W, num_inference_steps=S, guidance_scale=6.0,
           generator=torch.Generator(device="cuda").manual_seed(42), output_type="latent").video
torch.cuda.synchronize(); b = time.time(); print(f"PHASE denoise {a:.3f} {b:.3f}", flush=True)
vcpu = "vae_cpu" in sys.argv
vae = pipe.vae.to("cpu", torch.float32) if vcpu else pipe.vae
dt = torch.float32 if vcpu else vae.dtype
mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1); std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
z = (lat.float().cpu() * std + mean).to("cpu" if vcpu else "cuda", dt)
c = time.time()
with torch.no_grad(): v = vae.decode(z).sample
if not vcpu: torch.cuda.synchronize()
d = time.time(); print(f"PHASE vae {c:.3f} {d:.3f}", flush=True)
print("denoise", b - a, "vae", d - c, "vae_dev", "cpu" if vcpu else "cuda", tuple(v.shape))
