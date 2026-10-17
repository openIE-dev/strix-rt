# Decode latents.npy (B,C,T,H,W) + audio_latents.npy (B,C,Na) with the reference K6 decoders -> mp4 (+ timings).
import sys, os, time, json, numpy as np, torch
from diffusers import Kandinsky6TI2VAPipeline
from diffusers.utils import encode_video
M = os.path.expanduser("$MODELS/k6-pro-distill"); d = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else f"{d}/k6.mp4"
dev, dt = "cuda", torch.bfloat16
p = Kandinsky6TI2VAPipeline.from_pretrained(M, transformer=None, text_encoder=None, text_encoder_2=None, torch_dtype=dt)
for n in ("vae", "audio_vae", "vocoder"): getattr(p, n).to(dev)
if os.environ.get("K6_TILE", "1") == "1": p.vae.enable_tiling()
lat = torch.from_numpy(np.load(f"{d}/latents.npy")).to(dev, dt); alat = torch.from_numpy(np.load(f"{d}/audio_latents.npy")).to(dev, dt)
tm = {}
with torch.no_grad():
    torch.cuda.synchronize(); t0 = time.time()
    vid = p.vae.decode(lat / p.vae.config.scaling_factor, return_dict=False)[0]; torch.cuda.synchronize(); tm["vae"] = time.time() - t0
    t0 = time.time(); mel = p.audio_vae.decode(alat / p.audio_vae.config.scaling_factor, return_dict=False)[0]
    aud = p.vocoder(mel.to(p.vocoder.dtype), return_dict=False)[0][:, 0].float(); torch.cuda.synchronize(); tm["audio_dec"] = time.time() - t0
np.save(f"{d}/video.npy", vid.float().cpu().numpy()); np.save(f"{d}/audio.npy", aud.cpu().numpy())
frames = p.video_processor.postprocess_video(vid, output_type="np")
try: encode_video(frames[0], fps=24, output_path=out, audio=aud[0][None].cpu(), audio_sample_rate=p.audio_sample_rate)
except Exception as e: print("encode_video failed:", e)
print(json.dumps(tm), out)
