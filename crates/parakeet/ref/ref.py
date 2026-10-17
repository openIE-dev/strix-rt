# Parakeet Redux reference: unpack thrush-ternary-v2 weights -> HF ParakeetForTDT (fp32, CPU), greedy TDT decode.
# Writes golden transcripts + intermediates (mel, encoder out) for the strix-rt port.
import json, sys, time, os, numpy as np, torch, soundfile as sf
from safetensors.torch import load_file
from transformers import ParakeetForTDT, ParakeetTDTConfig, AutoProcessor
M = os.path.expanduser("$MODELS/parakeet-redux")
def load_model():
    sd = load_file(f"{M}/model.safetensors"); cfg = json.load(open(f"{M}/config.json"))
    out = {}
    for k, v in sd.items():
        if k.endswith(".qweight"):
            base = k[:-8]; sc = sd[base + ".scales"].float(); n, nb = v.shape
            q = v.to(torch.int64); digs = []
            for d in range(5): digs.append(q % 3); q = q // 3
            codes = torch.stack(digs, -1).reshape(n, nb * 5)
            kin = sc.shape[1] * 128
            codes = codes[:, :kin].float() - 1.0
            w = codes * sc.repeat_interleave(128, 1)[:, :kin]
            out[base + ".weight"] = w
        elif k.endswith(".scales"): continue
        elif v.dtype == torch.float16: out[k] = v.float()
        else: out[k] = v
    conf = ParakeetTDTConfig(**{k: v for k, v in cfg.items() if k not in ("ternary_modules", "ternary_group_size")})
    model = ParakeetForTDT(conf).eval()
    # conv1d-shaped ternary modules
    msd = model.state_dict()
    for k in list(out):
        if k in msd and out[k].shape != msd[k].shape and out[k].numel() == msd[k].numel(): out[k] = out[k].reshape(msd[k].shape)
    miss, unexp = model.load_state_dict(out, strict=False)
    print("missing", [m for m in miss if "num_batches" not in m][:20], "unexpected", [u for u in unexp if not u.startswith("vad_head")][:20], file=sys.stderr)
    return model
if __name__ == "__main__":
    model = load_model()
    from transformers import GenerationConfig
    model.generation_config = GenerationConfig.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
    proc = AutoProcessor.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
    torch.set_num_threads(int(os.environ.get("THREADS", "16")))
    res = {}
    os.makedirs("golden", exist_ok=True)
    for path in sys.argv[1:]:
        a, sr = sf.read(path, dtype="float32")
        if a.ndim > 1: a = a.mean(1)
        if sr != 16000:
            import librosa; a = librosa.resample(a, orig_sr=sr, target_sr=16000); sr = 16000
        name = os.path.splitext(os.path.basename(path))[0]
        inp = proc(a, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            t0 = time.time(); out = model.generate(**inp); dt = time.time() - t0
            enc = model.encoder(input_features=inp["input_features"], attention_mask=inp.get("attention_mask")).last_hidden_state
        ids = out.sequences[0].tolist() if hasattr(out, "sequences") else out[0].tolist()
        text = proc.batch_decode([ids], skip_special_tokens=True)[0]
        np.save(f"golden/{name}_mel.npy", inp["input_features"][0].numpy()); np.save(f"golden/{name}_enc.npy", enc[0].numpy())
        res[name] = {"text": text, "ids": ids, "sec": len(a) / sr, "ref_s": dt}
        print(f"{name}: {len(a)/sr:.2f}s audio, {dt:.3f}s -> RTF {dt/(len(a)/sr):.4f}: {text}", flush=True)
    json.dump(res, open("golden/hf.json", "w"), indent=1)
