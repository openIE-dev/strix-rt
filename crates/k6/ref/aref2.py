# Reference audio stage dumps (fp32 CPU), channel-last [L, C], matching k6 K6_ADUMP names
import numpy as np, torch
from diffusers.models.autoencoders.autoencoder_mmaudio import MMAudioVAE
from diffusers.models.autoencoders.mmaudio_vocoder import MMAudioVocoder
M = "$MODELS/k6-pro-distill"; out = "~/k6_ref/aud"; torch.set_num_threads(16)
vae = MMAudioVAE.from_pretrained(f"{M}/audio_vae", torch_dtype=torch.float32).eval()
voc = MMAudioVocoder.from_pretrained(f"{M}/vocoder", torch_dtype=torch.float32).eval()
def hook(n):
    def f(m, i, o): np.save(f"{out}/r_{n}.npy", o[0].permute(1, 0).numpy())
    return f
d = vae.vae.decoder
d.conv_in.register_forward_hook(hook("conv_in")); d.mid.block_1.register_forward_hook(hook("mid_b1")); d.mid.attn_1.register_forward_hook(hook("mid_attn")); d.mid.block_2.register_forward_hook(hook("mid_b2"))
for lv in range(3):
    m = d.up[lv].upsample if hasattr(d.up[lv], "upsample") else d.up[lv].block[-1]
    m.register_forward_hook(hook(f"up{lv}"))
voc.conv_pre.register_forward_hook(hook("conv_pre"))
for s in range(6): voc.ups[s][0].register_forward_hook(hook(f"ups{s}"))
z = torch.from_numpy(np.load("~/k6_ref/out/audio_latents.npy").astype(np.float32))
with torch.no_grad():
    mel = vae.decode(z / vae.config.scaling_factor, return_dict=False)[0]
    mel_ref = torch.from_numpy(np.load(f"{out}/aref_mel.npy"))[None]
    voc(mel_ref, return_dict=False)
print("ok")
