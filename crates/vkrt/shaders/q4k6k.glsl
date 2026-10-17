// Shared Q4_K / Q6_K dot-product items. Requires: buffers w (uint) and w16 (uint16_t) at the weight binding, and vec4 XV(uint).
uint byte_of(uvec3 s, uint i) { uint u = i < 4u ? s.x : (i < 8u ? s.y : s.z); return (u >> (8u * (i & 3u))) & 0xFFu; }
void q4k_scales(uvec3 s, uint j, out uint sc0, out uint m0, out uint sc1, out uint m1) {
    uint i0 = 2u * j, i1 = i0 + 1u;
    if (j < 2u) {
        sc0 = byte_of(s, i0) & 63u; m0 = byte_of(s, i0 + 4u) & 63u;
        sc1 = byte_of(s, i1) & 63u; m1 = byte_of(s, i1 + 4u) & 63u;
    } else {
        sc0 = (byte_of(s, i0 + 4u) & 0xFu) | ((byte_of(s, i0 - 4u) >> 6) << 4); m0 = (byte_of(s, i0 + 4u) >> 4) | ((byte_of(s, i0) >> 6) << 4);
        sc1 = (byte_of(s, i1 + 4u) & 0xFu) | ((byte_of(s, i1 - 4u) >> 6) << 4); m1 = (byte_of(s, i1 + 4u) >> 4) | ((byte_of(s, i1) >> 6) << 4);
    }
}
// woff in uint units; item = (block, 64-value pair j). Q4_K blocks are 144 B = 9 x 16 B, so header and qs are uvec4 loads.
float item4k(uint woff, uint row, uint nb, uint it) {
    uint b = it >> 2u, j = it & 3u;
    uint base = woff + (row * nb + b) * 36u;
#ifdef VEC4_LOADS
    uvec4 hdr = w4[base >> 2];
    uvec4 qa = w4[(base >> 2) + 1u + 2u * j], qc = w4[(base >> 2) + 2u + 2u * j];
    uint qq[8] = uint[8](qa.x, qa.y, qa.z, qa.w, qc.x, qc.y, qc.z, qc.w);
#else
    uvec4 hdr = uvec4(w[base], w[base + 1u], w[base + 2u], w[base + 3u]);
    uint qq[8]; for (uint k = 0u; k < 8u; k++) qq[k] = w[base + 4u + j * 8u + k];
#endif
    vec2 dm = unpackHalf2x16(hdr.x);
    uint sc0, m0, sc1, m1; q4k_scales(hdr.yzw, j, sc0, m0, sc1, m1);
    uint xb = (b * 256u + j * 64u) >> 2u;
    float s0 = 0.0, s1 = 0.0, sx0 = 0.0, sx1 = 0.0;
    [[unroll]] for (uint k = 0u; k < 8u; k++) {
        uint q = qq[k];
        vec4 lo = vec4(uvec4(q, q >> 8, q >> 16, q >> 24) & 0xFu);
        vec4 hi = vec4(uvec4(q >> 4, q >> 12, q >> 20, q >> 28) & 0xFu);
        vec4 xa = XV(xb + k), xh = XV(xb + 8u + k);
        s0 += dot(lo, xa); sx0 += dot(xa, vec4(1.0)); s1 += dot(hi, xh); sx1 += dot(xh, vec4(1.0));
    }
    return dm.x * (float(sc0) * s0 + float(sc1) * s1) - dm.y * (float(m0) * sx0 + float(m1) * sx1);
}
// woff in uint16 units; item = (block, half n, group of 8 g)
float item6k(uint woff, uint row, uint nb, uint it) {
    uint b = it >> 3u, n = (it >> 2u) & 1u, g = it & 3u;
    uint base = woff + (row * nb + b) * 105u;
    float d = unpackHalf2x16(uint(w16[base + 104u])).x;
    int sh = int((g >> 1u) * 8u); uint sb = base + 96u + n * 4u;
    float sc0 = float(bitfieldExtract(int(uint(w16[sb])), sh, 8)), sc1 = float(bitfieldExtract(int(uint(w16[sb + 1u])), sh, 8));
    float sc2 = float(bitfieldExtract(int(uint(w16[sb + 2u])), sh, 8)), sc3 = float(bitfieldExtract(int(uint(w16[sb + 3u])), sh, 8));
    uint qlb = base + n * 32u + g * 4u, qhb = base + 64u + n * 16u + g * 4u, xb = (b * 256u + n * 128u + g * 8u) >> 2u;
    float s1 = 0.0, s2 = 0.0, s3 = 0.0, s4 = 0.0;
    [[unroll]] for (uint v = 0u; v < 2u; v++) {
        uint ua = uint(w16[qlb + 2u * v]) | (uint(w16[qlb + 2u * v + 1u]) << 16);
        uint uc = uint(w16[qlb + 16u + 2u * v]) | (uint(w16[qlb + 17u + 2u * v]) << 16);
        uint uh = uint(w16[qhb + 2u * v]) | (uint(w16[qhb + 2u * v + 1u]) << 16);
        uvec4 A = uvec4(ua, ua >> 8, ua >> 16, ua >> 24), C = uvec4(uc, uc >> 8, uc >> 16, uc >> 24), H = uvec4(uh, uh >> 8, uh >> 16, uh >> 24);
        vec4 q1 = vec4(ivec4((A & 0xFu) | ((H & 3u) << 4)) - 32), q2 = vec4(ivec4((C & 0xFu) | (((H >> 2) & 3u) << 4)) - 32);
        vec4 q3 = vec4(ivec4(((A >> 4) & 0xFu) | (((H >> 4) & 3u) << 4)) - 32), q4 = vec4(ivec4(((C >> 4) & 0xFu) | (((H >> 6) & 3u) << 4)) - 32);
        s1 += dot(q1, XV(xb + v)); s2 += dot(q2, XV(xb + 8u + v)); s3 += dot(q3, XV(xb + 16u + v)); s4 += dot(q4, XV(xb + 24u + v));
    }
    return d * (sc0 * s1 + sc1 * s2 + sc2 * s3 + sc3 * s4);
}
