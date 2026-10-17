# stdlib-only PNG (8-bit RGB/RGBA, non-interlaced) -> binary PPM, for mcv_vis / mcv_run inputs
import sys, zlib, struct
d = open(sys.argv[1], 'rb').read(); assert d[:8] == b'\x89PNG\r\n\x1a\n'
i, idat = 8, b''
while i < len(d):
    n, t = struct.unpack('>I4s', d[i:i + 8]); c = d[i + 8:i + 8 + n]; i += 12 + n
    if t == b'IHDR': w, h, bd, ct, _, _, il = struct.unpack('>IIBBBBB', c)
    elif t == b'IDAT': idat += c
assert bd == 8 and ct in (2, 6) and il == 0
bpp = 3 if ct == 2 else 4; raw = zlib.decompress(idat); st = w * bpp
out, prev, p = bytearray(), bytearray(st), 0
for y in range(h):
    f = raw[p]; line = bytearray(raw[p + 1:p + 1 + st]); p += 1 + st
    for x in range(st):
        a = line[x - bpp] if x >= bpp else 0; b = prev[x]; cc = prev[x - bpp] if x >= bpp else 0
        if f == 1: line[x] = (line[x] + a) & 255
        elif f == 2: line[x] = (line[x] + b) & 255
        elif f == 3: line[x] = (line[x] + (a + b) // 2) & 255
        elif f == 4:
            pp = a + b - cc; pa, pb, pc = abs(pp - a), abs(pp - b), abs(pp - cc)
            line[x] = (line[x] + (a if pa <= pb and pa <= pc else (b if pb <= pc else cc))) & 255
    prev = line
    for x in range(w): out += line[x * bpp:x * bpp + 3]
open(sys.argv[2], 'wb').write(b'P6\n%d %d\n255\n' % (w, h) + bytes(out))
print(w, h)
