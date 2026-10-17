from datasets import load_dataset, Audio
import soundfile as sf, os, io
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation").cast_column("audio", Audio(decode=False))
os.makedirs("clips", exist_ok=True)
with open("clips/refs.tsv", "w") as f:
    for i in range(8):
        r = ds[i]; au = r["audio"]
        a, sr = sf.read(io.BytesIO(au["bytes"])) if au.get("bytes") else sf.read(au["path"])
        sf.write(f"clips/ls{i}.wav", a, sr); f.write("ls%d\t%s\n" % (i, r["text"]))
print(open("clips/refs.tsv").read())
