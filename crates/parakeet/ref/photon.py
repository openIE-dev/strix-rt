# Second golden: Moondream Photon (the model's own runtime), CPU.
import sys, json, time, os, soundfile as sf
import moondream as md
res = {}
with md.photon("moondream/parakeet-redux", device="cpu") as sp:
    sp.transcribe(audio=sys.argv[1])  # warm-up
    for p in sys.argv[1:]:
        a, sr = sf.read(p); sec = len(a) / sr
        t0 = time.time(); r = sp.transcribe(audio=p); dt = time.time() - t0
        n = os.path.splitext(os.path.basename(p))[0]; res[n] = {"text": r["text"], "sec": sec, "photon_s": dt}
        print(f"{n}: {sec:.2f}s, {dt:.3f}s RTF {dt/sec:.4f}: {r['text']}", flush=True)
json.dump(res, open("golden/photon.json", "w"), indent=1)
