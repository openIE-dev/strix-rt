# mmproj GGUF -> one safetensors file (GGUF tensor names, row-major [n][k] shapes, every tensor 256-byte aligned)
# so qimage::st can map the BF16 vision weights zero-copy for prefill's GEMM.
import sys, json, struct
sys.path.insert(0, '~/llama.cpp/gguf-py')
import gguf
src, dst = sys.argv[1], sys.argv[2]
r = gguf.GGUFReader(src)
DT = {'BF16': 'BF16', 'F32': 'F32', 'F16': 'F16'}
hdr, blobs, off = {}, [], 0
for t in r.tensors:
    ty = t.tensor_type.name
    raw = bytes(t.data.tobytes())
    shape = [int(x) for x in reversed([int(s) for s in t.shape])]
    if ty in ('BF16', 'F16') and len(shape) == 2 and shape[1] % 64:
        # GEMM K must be a multiple of 64: zero-pad each weight row (ViT ffn_down: 4304 -> 4352)
        k, kp = shape[1], (shape[1] + 63) // 64 * 64
        rows = [raw[i * 2 * k:(i + 1) * 2 * k] + b'\0' * (2 * (kp - k)) for i in range(shape[0])]
        raw = b''.join(rows); shape[1] = kp
    pad = (-off) % 256
    if pad: blobs.append(b'\0' * pad); off += pad
    hdr[t.name] = {'dtype': DT[ty], 'shape': shape, 'data_offsets': [off, off + len(raw)]}
    blobs.append(raw); off += len(raw)
h = json.dumps(hdr).encode()
h += b' ' * ((-(8 + len(h))) % 256)
with open(dst, 'wb') as f:
    f.write(struct.pack('<Q', len(h))); f.write(h)
    for b in blobs: f.write(b)
print('wrote', dst, len(hdr), 'tensors', off, 'bytes')
