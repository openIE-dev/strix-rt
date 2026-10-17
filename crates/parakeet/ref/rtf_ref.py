# CPU reference timings (median of REPS runs per clip, after a warm-up): HF ParakeetForTDT (fp32) and Moondream Photon.
# usage: python rtf_ref.py out.json clip16k.wav ...
import sys, json, time, os, statistics, soundfile as sf, torch
sys.path.insert(0, os.path.dirname(__file__))
from ref import load_model
from transformers import AutoProcessor
import moondream as md
REPS = int(os.environ.get("REPS", "3")); torch.set_num_threads(int(os.environ.get("THREADS", "16")))
clips = sys.argv[2:]; res = {os.path.splitext(os.path.basename(p))[0]: {} for p in clips}
from transformers import GenerationConfig
model = load_model(); model.generation_config = GenerationConfig.from_pretrained("nvidia/parakeet-tdt-0.6b-v3"); proc = AutoProcessor.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
def hf(p):
    a, sr = sf.read(p, dtype="float32"); inp = proc(a, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad(): out = model.generate(**inp)
    return proc.batch_decode([out.sequences[0].tolist() if hasattr(out, "sequences") else out[0].tolist()], skip_special_tokens=True)[0]
with md.photon("moondream/parakeet-redux", device="cpu") as sp:
    for name, fn in [("hf", hf), ("photon", lambda p: sp.transcribe(audio=p)["text"])]:
        fn(clips[0])
        for p in clips:
            n = os.path.splitext(os.path.basename(p))[0]; ts = []
            for _ in range(REPS): t0 = time.time(); txt = fn(p); ts.append(time.time() - t0)
            res[n]["sec"] = len(sf.read(p)[0]) / 16000; res[n][name + "_s"] = statistics.median(ts); res[n][name + "_text"] = txt
            print(name, n, f"{res[n][name + '_s']:.4f}s RTF {res[n][name + '_s'] / res[n]['sec']:.4f}", txt, flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
