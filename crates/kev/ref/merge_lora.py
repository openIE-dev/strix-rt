"""Fold a Kev LoRA adapter into its Qwen3.5 base in fp32 and write an HF dir for convert_hf_to_gguf (any shard layout)."""
import sys, shutil, json, glob, os, torch
from safetensors.torch import load_file, save_file
base, ad, out = sys.argv[1:4]
shutil.copytree(base, out, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.safetensors*"))
A = load_file(f"{ad}/adapter_model.safetensors")
cfg = json.load(open(f"{ad}/adapter_config.json")); sc = cfg["lora_alpha"] / cfg["r"]
pairs = {k.split("layers.", 1)[1].replace(".lora_A.weight", ".weight"): k for k in A if ".lora_A." in k}
n = 0
for f in sorted(glob.glob(f"{base}/*.safetensors")):
    W = load_file(f)
    for t in list(W):
        if "visual" in t or "mtp" in t or "layers." not in t: continue
        k = pairs.get(t.split("layers.", 1)[1])
        if k is None: continue
        a = A[k].float(); b = A[k.replace("lora_A", "lora_B")].float()
        W[t] = (W[t].float() + sc * (b @ a)).to(W[t].dtype); n += 1
    save_file(W, f"{out}/{os.path.basename(f)}", metadata={"format": "pt"})
for f in glob.glob(f"{base}/*.safetensors.index.json"): shutil.copy(f, out)
print("merged", n, "of", len(pairs)); assert n == len(pairs)
