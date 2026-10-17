# Reference audio decode (fp32 CPU): audio_latents (1,40,L) / 0.417 -> MMAudio VAE mel -> vocoder wav
import sys, numpy as np, torch
from diffusers.models.autoencoders.autoencoder_mmaudio import MMAudioVAE
from diffusers.models.autoencoders.mmaudio_vocoder import MMAudioVocoder
src, out = sys.argv[1], sys.argv[2]; M = "$MODELS/k6-pro-distill"
torch.set_num_threads(16)
vae = MMAudioVAE.from_pretrained(f"{M}/audio_vae", torch_dtype=torch.float32).eval()
voc = MMAudioVocoder.from_pretrained(f"{M}/vocoder", torch_dtype=torch.float32).eval()
z = torch.from_numpy(np.load(src).astype(np.float32))
with torch.no_grad():
    mel = vae.decode(z / vae.config.scaling_factor, return_dict=False)[0]
    wav = voc(mel, return_dict=False)[0][:, 0]
np.save(f"{out}/aref_mel.npy", mel[0].numpy()); np.save(f"{out}/aref_wav.npy", wav[0].numpy()); print("mel", tuple(mel.shape), "wav", tuple(wav.shape))
