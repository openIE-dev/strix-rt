# Fixed-seed YuE2-3B reference (CPU torch). Saves artifacts + per-stage parity dumps.
import sys, time, json, os, numpy as np, torch
from pathlib import Path
torch.set_num_threads(int(os.environ.get("NT", "32")))
from yue2 import YuE2Pipeline
from yue2.nar import song_chunks, CachedNAR
out = Path(sys.argv[1] if len(sys.argv) > 1 else "out_ref"); out.mkdir(exist_ok=True)
STYLE = "Upbeat synth pop, bright female vocal, punchy drums, warm bass, catchy chorus"
LYRICS = "[Verse]\nNeon lights are calling out my name\nEvery street is humming in the rain\n\n[Chorus]\nWe are running, running through the night\nHearts on fire, everything is bright\n"
SEED = int(os.environ.get("SEED", "1234")); COT = os.environ.get("COT", "full")
MAXSEM = int(os.environ.get("MAXSEM", "750"))
t0 = time.perf_counter()
pipe = YuE2Pipeline.from_pretrained("m-a-p/YuE2-3B", device="cpu", progress=False, backend="torch-eager", verify_hashes=False)
T = {}
s = time.perf_counter(); plan = pipe.plan(style=STYLE, lyrics=LYRICS, cot=COT, seed=SEED); T["abc"] = time.perf_counter() - s
s = time.perf_counter(); sem = pipe.generate_semantic(plan, sampling={"max_tokens": MAXSEM, "min_tokens": min(200, MAXSEM)}); T["semantic"] = time.perf_counter() - s
s = time.perf_counter(); lat = pipe.synthesize(sem); T["nar"] = time.perf_counter() - s
s = time.perf_counter(); audio = pipe.decode(lat); T["vae"] = time.perf_counter() - s
import soundfile as sf
sf.write(out / "ref.wav", audio, 48000, subtype="FLOAT")
np.save(out / "prefix.npy", np.asarray(plan.prefix, np.int32)); np.save(out / "abc.npy", np.asarray(plan.abc_ids, np.int32))
np.save(out / "semantic.npy", np.asarray(sem.tokens, np.int32)); np.save(out / "latent.npy", lat.astype(np.float32))
(out / "score.abc").write_text(plan.abc or "")
T["audio_s"] = len(audio) / 48000; T["abc_tokens"] = len(plan.abc_ids); T["sem_tokens"] = len(sem.tokens); T["prefix_len"] = len(plan.prefix)
T["abc_timing"] = plan.timing; T["sem_timing"] = sem.timing
# ---- per-stage parity dumps (teacher forced) ----
model = pipe._load_model(for_nar=True)
with torch.inference_mode():
    from yue2.protocol import CODEC_OFFSET, MUSIC_END
    full = plan.prefix + [t + CODEC_OFFSET for t in sem.tokens]
    o = model(torch.tensor([full]), use_cache=False)
    lg = o.logits[0].float()
    pos = list(range(len(plan.prefix) - 1, len(full), 25))
    np.save(out / "ar_logit_pos.npy", np.asarray(pos, np.int32))
    np.save(out / "ar_logits.npy", lg[pos].numpy())
    ch = song_chunks(plan.prefix, sem.tokens, SEED)[0]
    np.save(out / "noise.npy", ch.noise.numpy())
    eng = CachedNAR(model, ch)
    raw = torch.logit(torch.tensor(1.0, dtype=torch.float64)).clamp(-20, 20).item()
    np.save(out / "v0.npy", eng.velocity(ch.noise.to(torch.bfloat16), raw).float().numpy())
    eng.close()
json.dump(T, open(out / "timing.json", "w"), indent=1, default=str)
print(json.dumps(T, default=str))
