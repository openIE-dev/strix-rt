#!/usr/bin/env python3
"""Build a flat tensor index for a safetensors directory: one line per tensor
   name dtype file abs_offset nbytes dims(comma)   -- abs_offset = 8 + header_len + data_offset.
Usage: beam_index.py DIR  -> DIR/beam.idx"""
import json, struct, sys, os, glob
d = sys.argv[1]
out = []
for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n))
    for k, v in h.items():
        if k == "__metadata__": continue
        a, b = v["data_offsets"]
        out.append(f"{k} {v['dtype']} {os.path.basename(f)} {8+n+a} {b-a} {','.join(map(str,v['shape']))}")
open(os.path.join(d, "beam.idx"), "w").write("\n".join(out) + "\n")
print(len(out), "tensors")
