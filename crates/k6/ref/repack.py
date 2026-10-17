# Rewrite a safetensors shard with every tensor 16-byte aligned (offsets rewritten), optionally dropping a key prefix.
import json, struct, sys, os
src, dst = sys.argv[1], sys.argv[2]; drop = sys.argv[3] if len(sys.argv) > 3 else None
f = open(src, "rb"); n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n)); base = 8 + n
meta = h.pop("__metadata__", None)
keys = sorted((k for k in h if not (drop and k.startswith(drop))), key=lambda k: h[k]["data_offsets"][0])
nh, off = {}, 0
for k in keys:
    a, b = h[k]["data_offsets"]; nh[k] = dict(dtype=h[k]["dtype"], shape=h[k]["shape"], data_offsets=[off, off + b - a]); off += b - a; off += (-off) % 16
if meta: nh["__metadata__"] = meta
hb = json.dumps(nh).encode(); hb += b" " * ((-(8 + len(hb))) % 16)
o = open(dst + ".tmp", "wb"); o.write(struct.pack("<Q", len(hb))); o.write(hb)
for k in keys:
    a, b = h[k]["data_offsets"]; f.seek(base + a); o.write(f.read(b - a)); o.write(b"\0" * ((-(b - a)) % 16))
o.close(); os.rename(dst + ".tmp", dst); print("ok", len(keys), "tensors", os.path.getsize(dst))
