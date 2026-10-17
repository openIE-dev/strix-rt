#!/usr/bin/env python3
# Physical contiguity of a file's page-cache pages: maps the file, touches it, reads PFNs from /proc/self/pagemap (root).
# Reports the share of the file covered by physically contiguous, 2MiB-aligned 2MiB runs (what the GPU can map as big fragments).
import mmap, os, struct, sys
path = sys.argv[1]; f = open(path, 'rb'); n = os.fstat(f.fileno()).st_size
m = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
npg = (n + 4095) // 4096
for i in range(0, npg): m[i * 4096]
import ctypes
addr = ctypes.addressof(ctypes.c_char.from_buffer_copy(b'\0'))  # dummy
# get mapping address via /proc/self/maps
base = None
for l in open('/proc/self/maps'):
    if path in l: base = int(l.split('-')[0], 16); break
pm = open('/proc/self/pagemap', 'rb'); pm.seek(base // 4096 * 8); raw = pm.read(npg * 8)
pfns = [struct.unpack_from('<Q', raw, i * 8)[0] & ((1 << 55) - 1) for i in range(npg)]
runs2m = 0; total2m = 0; contig_pages = 0
for i in range(0, npg - 511, 512):
    total2m += 1
    p0 = pfns[i]
    if p0 % 512 == 0 and all(pfns[i + j] == p0 + j for j in range(512)): runs2m += 1
# generic run-length histogram
hist = {}
i = 0
while i < npg:
    j = i + 1
    while j < npg and pfns[j] == pfns[j - 1] + 1: j += 1
    L = j - i; b = 1 << (L.bit_length() - 1); hist[b] = hist.get(b, 0) + L; i = j
print(f"{path}: {npg} pages; 2MiB-aligned contiguous chunks: {runs2m}/{total2m} ({100*runs2m/max(total2m,1):.1f}%)")
print("bytes by physically-contiguous run length (pages, pow2 bucket): " + ", ".join(f"{k}:{100*v/npg:.1f}%" for k, v in sorted(hist.items())))
