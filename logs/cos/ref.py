# Cosmos3-Nano diffusers reference with tensor capture. usage: ref.py OUTDIR [H W F STEPS] [cpu]
import sys, os, json, time, torch, numpy as np
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
H, W, F, S = [int(x) for x in (sys.argv[2:6] if len(sys.argv) > 5 else (256, 256, 9, 8))]
dev = "cpu" if "cpu" in sys.argv else "cuda"
M = os.path.expanduser("$MODELS/new/cosmos3/Cosmos3-Nano")
t0 = time.time()
pipe = Cosmos3OmniPipeline.from_pretrained(M, torch_dtype=torch.bfloat16, enable_safety_checker=False).to(dev)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=10.0)
print("load", time.time() - t0, flush=True)
tr = pipe.transformer
calls = []
def save(name, t): np.save(f"{out}/{name}.npy", t.detach().float().cpu().numpy())
def pre(m, a, kw):
    i = len(calls); calls.append(i)
    if i < 2:
        for k in ("input_ids", "position_ids", "vision_timesteps"):
            save(f"c{i}_{k}", kw[k]) 
        save(f"c{i}_vis_in", kw["vision_tokens"][0]); json.dump({"und_len": kw["und_len"], "seq": kw["sequence_length"]}, open(f"{out}/c{i}_meta.json", "w"))
    else:
        save(f"c{i}_vis_in", kw["vision_tokens"][0]); save(f"c{i}_ts", kw["vision_timesteps"][:1])
def post(m, a, kw, o):
    save(f"c{len(calls)-1}_vel", o[0][0] if isinstance(o, tuple) else o.sample[0])
tr.register_forward_pre_hook(pre, with_kwargs=True); tr.register_forward_hook(post, with_kwargs=True)
def lh(idx):
    def f(m, a, o):
        if len(calls) == 1: save(f"c0_L{idx}_und", o[0]); save(f"c0_L{idx}_gen", o[1])
    return f
for idx in (0, 1, 17, 35): tr.layers[idx].register_forward_hook(lh(idx))
def eh(m, a, o):
    if len(calls) == 1: save("c0_emb_gen_in", a[1])
tr.layers[0].register_forward_hook(lambda m,a,o: save("c0_L0_in_gen", a[1]) if len(calls)==1 else None)
vo = {"dec": [], "mid": [], "up0": [], "up1": [], "up2": []}
pipe.vae.post_quant_conv.register_forward_pre_hook(lambda m, a: save("vae_z", a[0]))
pipe.vae.decoder.register_forward_hook(lambda m, a, o: vo["dec"].append(o.detach().float().cpu()))
pipe.vae.decoder.mid_block.register_forward_hook(lambda m, a, o: vo["mid"].append(o.detach().float().cpu()))
for _i in range(3): pipe.vae.decoder.up_blocks[_i].register_forward_hook((lambda k: lambda m, a, o: vo[k].append(o.detach().float().cpu()))(f"up{_i}"))
prompt = open(f"{M}/assets/example_t2v_prompt_short.txt").read().strip()
t1 = time.time()
r = pipe(prompt=prompt, negative_prompt="", num_frames=F, height=H, width=W, num_inference_steps=S, guidance_scale=6.0,
         generator=torch.Generator(device=dev).manual_seed(42), output_type="np")
if dev == "cuda": torch.cuda.synchronize()
el = time.time() - t1
[np.save(f"{out}/vae_{k}.npy", torch.cat(v, 2).numpy()) for k, v in vo.items() if v]
np.save(f"{out}/video.npy", np.asarray(r.video))
np.save(f"{out}/sigmas.npy", pipe.scheduler.sigmas.cpu().numpy()); np.save(f"{out}/timesteps.npy", pipe.scheduler.timesteps.cpu().numpy())
json.dump({"H": H, "W": W, "F": F, "steps": S, "gen_s": el, "dev": dev, "calls": len(calls), "prompt": prompt}, open(f"{out}/run.json", "w"))
print("gen", el, "calls", len(calls), flush=True)
