# Kandinsky 6.0 Pro-distill fixed-seed reference with per-stage dumps (staged loading: TE -> DiT -> VAEs).
import os, sys, time, json, gc, numpy as np, torch
from diffusers import Kandinsky6TI2VAPipeline, Kandinsky6Transformer3DModel
M = os.path.expanduser("$MODELS/k6-pro-distill")
OUT = os.environ.get("K6_OUT", os.path.expanduser("~/k6_ref/out"))
H, W, F = int(os.environ.get("K6_H", 320)), int(os.environ.get("K6_W", 576)), int(os.environ.get("K6_F", 49))
STEPS, SEED = int(os.environ.get("K6_STEPS", 16)), int(os.environ.get("K6_SEED", 1234))
PROMPT = os.environ.get("K6_PROMPT", "A red fox trotting through fresh snow in a pine forest at sunrise, soft crunching footsteps, birds chirping.")
DUMP_BLOCKS = [int(x) for x in os.environ.get("K6_DUMP_BLOCKS", "0,1,29,59").split(",") if x]
dev = "cuda"; dt = torch.bfloat16
os.makedirs(OUT, exist_ok=True)
sv = lambda n, t: np.save(f"{OUT}/{n}.npy", t.detach().float().cpu().numpy())
tm = {}
def sync(): torch.cuda.synchronize()
# ---- stage 1: text encoders
t0 = time.time()
pipe = Kandinsky6TI2VAPipeline.from_pretrained(M, transformer=None, vae=None, audio_vae=None, vocoder=None, torch_dtype=dt)
pipe.text_encoder.to(dev); pipe.text_encoder_2.to(dev)
tm["te_load"] = time.time() - t0; sync(); t0 = time.time()
with torch.no_grad():
    pe, pp, pm, *_ = pipe.encode_prompt(prompt=[PROMPT], negative_prompt=None, do_classifier_free_guidance=False,
        num_videos_per_prompt=1, max_sequence_length=1024, device=dev, dtype=dt)
sync(); tm["te"] = time.time() - t0
sv("prompt_embeds", pe); sv("pooled", pp)
if pm is not None:
    sv("prompt_mask", pm); nv = int(pm[0].sum()); assert bool(pm[0, :nv].all()), "mask not a prefix"
    pe = pe[:, :nv]; print("trimmed prompt to", nv, "tokens")
sv("prompt_embeds_trim", pe); print("prompt tokens", pe.shape)
sched = pipe.scheduler; del pipe; gc.collect(); torch.cuda.empty_cache()
# ---- stage 2: DiT
t0 = time.time()
tr = Kandinsky6Transformer3DModel.from_pretrained(M, subfolder="transformer", torch_dtype=dt).to(dev).eval()
tm["dit_load"] = time.time() - t0
json.dump(dict(tr.config), open(f"{OUT}/transformer_config.json", "w"), default=str, indent=1)
calls = {"n": 0}
def blk_hook(i):
    def h(mod, args, out):
        if calls["n"] == 1:
            sv(f"s0_blk{i}_vis", out[0]);
            if out[1] is not None: sv(f"s0_blk{i}_aud", out[1])
    return h
for i in DUMP_BLOCKS: tr.visual_transformer_blocks[i].register_forward_hook(blk_hook(i))
def tb_hook(n):
    def h(mod, args, out):
        if calls["n"] == 1: sv(n, out)
    return h
tr.video_text_transformer_blocks[-1].register_forward_hook(tb_hook("s0_vtext"))
tr.audio_text_transformer_blocks[-1].register_forward_hook(tb_hook("s0_atext"))
orig = tr.forward
step_t = []
def fwd(*a, **k):
    calls["n"] += 1
    if calls["n"] == 1:
        sv("s0_in_vis", k["hidden_states"]); sv("s0_in_aud", k["audio_hidden_states"]); sv("s0_t", k["timestep"])
    sync(); t = time.time(); o = orig(*a, **k); sync(); step_t.append(time.time() - t)
    if calls["n"] == 1: sv("s0_v_vis", o[0]); sv("s0_v_aud", o[1])
    if os.environ.get("K6_ALLSTEPS"):
        i = calls["n"] - 1; sv(f"st{i}_in_vis", k["hidden_states"][..., :16]); sv(f"st{i}_in_aud", k["audio_hidden_states"]); sv(f"st{i}_t", k["timestep"]); sv(f"st{i}_v_vis", o[0]); sv(f"st{i}_v_aud", o[1])
    return o
tr.forward = fwd
_aux = Kandinsky6TI2VAPipeline.from_pretrained(M, transformer=None, vae=None, text_encoder=None, text_encoder_2=None, torch_dtype=dt)
pipe = Kandinsky6TI2VAPipeline(transformer=tr, vae=None, scheduler=sched, text_encoder=None, tokenizer=None,
    text_encoder_2=None, tokenizer_2=None, audio_vae=_aux.audio_vae, vocoder=_aux.vocoder)
g = torch.Generator(device="cpu").manual_seed(SEED)
steps_lat = []
def cb(p, i, t, kw):
    steps_lat.append(float(t)); return {}
t0 = time.time()
with torch.no_grad():
    lat, alat = pipe(prompt_embeds=pe, pooled_prompt_embeds=pp, height=H, width=W, num_frames=F, num_inference_steps=STEPS,
        guidance_scale=1.0, generator=g, output_type="latent", return_dict=False, callback_on_step_end=cb)
sync(); tm["dit_total"] = time.time() - t0; tm["dit_steps"] = step_t
sv("latents", lat); sv("audio_latents", alat)
json.dump(steps_lat, open(f"{OUT}/timesteps.json", "w"))
del pipe, tr; gc.collect(); torch.cuda.empty_cache()
if os.environ.get("K6_DITONLY"): json.dump(tm, open(f"{OUT}/timing_dit.json", "w"), indent=1); print(json.dumps(tm)); sys.exit(0)
# ---- stage 3: decoders
from diffusers import AutoencoderKLHunyuanVideo
from diffusers.utils import encode_video
full = Kandinsky6TI2VAPipeline.from_pretrained(M, transformer=None, text_encoder=None, text_encoder_2=None, torch_dtype=dt)
for n in ("vae", "audio_vae", "vocoder"): getattr(full, n).to(dev)
sync(); t0 = time.time()
with torch.no_grad():
    vid = full.vae.decode(lat.to(dev, full.vae.dtype) / full.vae.config.scaling_factor, return_dict=False)[0]
sync(); tm["vae"] = time.time() - t0; sv("video", vid)
t0 = time.time()
with torch.no_grad():
    mel = full.audio_vae.decode(alat.to(dev, full.audio_vae.dtype) / full.audio_vae.config.scaling_factor, return_dict=False)[0]
    sv("mel", mel)
    aud = full.vocoder(mel.to(full.vocoder.dtype), return_dict=False)[0][:, 0].float()
sync(); tm["audio_dec"] = time.time() - t0; sv("audio", aud)
frames = full.video_processor.postprocess_video(vid, output_type="np")
encode_video(frames[0], fps=24, output_path=f"{OUT}/ref.mp4", audio=aud[0][None].cpu(), audio_sample_rate=full.audio_sample_rate)
tm.update(H=H, W=W, F=F, steps=STEPS, seed=SEED, prompt=PROMPT, sr=full.audio_sample_rate)
json.dump(tm, open(f"{OUT}/timing.json", "w"), indent=1); print(json.dumps(tm))
