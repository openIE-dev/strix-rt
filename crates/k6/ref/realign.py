# Copy a safetensors file with its JSON header space-padded so every tensor starts 16-byte aligned (GEMM requirement).
import json, struct, sys, os
src, dst = sys.argv[1], sys.argv[2]
f = open(src, "rb"); n = struct.unpack("<Q", f.read(8))[0]; hb = f.read(n); h = json.loads(hb)
bad = [k for k, v in h.items() if k != "__metadata__" and v["data_offsets"][0] % 16]
assert not bad, f"data offsets not 16-aligned: {bad[:3]}"
pad = (-(8 + n)) % 16; hb2 = hb + b" " * pad
assert (8 + len(hb2)) % 16 == 0
o = open(dst + ".tmp", "wb"); o.write(struct.pack("<Q", len(hb2))); o.write(hb2)
while True:
    b = f.read(256 << 20)
    if not b: break
    o.write(b)
o.close(); os.rename(dst + ".tmp", dst); print("ok pad", pad, "size", os.path.getsize(dst))
