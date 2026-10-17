// Dequantize one 32-weight sub-block `it` of a row starting at byte rb into v[8] (vec4 = 4 consecutive weights).
// Same formats/type codes as vkrt q_any.glsl (rtcore::qtype): 0 Q4_K, 1 Q6_K, 2 Q5_K, 3 Q8_0, 4 IQ4_NL, 5 IQ4_XS, 6 Q3_K, 7 IQ3_S,
// 8 PQ2_0 (ternary, 128-blocks), 9 PTQ1_0 (5 trits/byte, 128-blocks, 28 B), 10 BF16, 11 F16, 15 Q5_0, 16 Q4_0.
// Needs bindings w (uint), w2 (uvec2), w16 (uint16_t), w4 (uvec4), f32w (codebooks at [0,528)).
shared uint IQ3S_GRID[512];
shared float KV_IQ4NL[16];
void deq_init(uint ty) {
    if (!(ty == 4u || ty == 5u || ty == 7u)) return;
    for (uint i = gl_LocalInvocationID.x; i < 528u; i += gl_WorkGroupSize.x) { if (i < 512u) IQ3S_GRID[i] = floatBitsToUint(f32w[i]); else KV_IQ4NL[i - 512u] = f32w[i]; }
    barrier();
}
uint rdb(uint bo) { return (w[bo >> 2] >> (8u * (bo & 3u))) & 0xFFu; }
uint rd16(uint bo) { return (bo & 3u) == 3u ? (rdb(bo) | (rdb(bo + 1u) << 8)) : ((w[bo >> 2] >> (8u * (bo & 3u))) & 0xFFFFu); }
uint rd32(uint bo) { uint h = bo >> 1; return uint(w16[h]) | (uint(w16[h + 1u]) << 16); }
uint sbyte(uvec4 v, uint i) { uint u = i < 4u ? v.x : i < 8u ? v.y : i < 12u ? v.z : v.w; return (u >> (8u * (i & 3u))) & 0xFFu; }
float f16at(uint bo) { return unpackHalf2x16(rd16(bo)).x; }
uint blk_elems(uint ty) { return (ty == 3u || ty == 4u || ty >= 10u) ? 32u : (ty == 8u || ty == 9u) ? 128u : 256u; }
uint blk_bytes(uint ty) { return ty == 0u ? 144u : ty == 1u ? 210u : ty == 2u ? 176u : ty == 3u ? 34u : ty == 4u ? 18u : ty == 5u ? 136u : ty == 8u ? 34u : ty == 9u ? 28u : ty == 15u ? 22u : ty == 16u ? 18u : ty >= 10u ? 64u : 110u; }
uint row_bytes_of(uint ty, uint K) { return ty == 12u ? 0u : ty == 13u ? K * 2u + 256u : K / blk_elems(ty) * blk_bytes(ty); }
// T8 (code 12): weights repacked at load time into 128-row x 32-k chunks: [128 x 32 int8][128 x f16 scale] = 4352 B,
// chunk (row/128, kb) at base + ((row/128) * (K/32) + kb) * 4352. Each lane reads its row's 32 contiguous bytes, so a
// 128-row tile's k-slab is one contiguous 4 KB read.
void deq_t8(uint base, uint row, uint K, uint kb, out vec4 v[8]) {
    uint c = base + ((row >> 7) * (K >> 5) + kb) * 4352u, r = row & 127u;
    uvec4 q0 = w4[(c + r * 32u) >> 4], q1 = w4[(c + r * 32u + 16u) >> 4];
    float d = unpackHalf2x16(uint(w16[(c + 4096u + r * 2u) >> 1])).x;
    uint qa[8] = uint[8](q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w);
    [[unroll]] for (uint k = 0u; k < 8u; k++) { uint a = qa[k]; v[k] = vec4(ivec4(int(a << 24) >> 24, int(a << 16) >> 24, int(a << 8) >> 24, int(a) >> 24)) * d; }
}
void deq32(uint ty, uint rb, uint it, out vec4 v[8]) {
    if (ty == 9u) {                              // PTQ1_0: qs[16] (k = p*16 + j), qs[16..24] (k = 80 + p*8 + j), qh[2] (k = 120 + p*2 + j), f16 d
        uint b = rb + (it >> 2) * 28u, s = it & 3u;
        uint wb = b >> 2;                        // 28-byte blocks of a 4-aligned row are 4-aligned
        uint q7[7]; [[unroll]] for (uint i = 0u; i < 7u; i++) q7[i] = w[wb + i];
        float d = unpackHalf2x16(q7[6] >> 16).x;
        [[unroll]] for (uint kk = 0u; kk < 8u; kk++) { vec4 o;
            [[unroll]] for (uint c = 0u; c < 4u; c++) {
                uint k = s * 32u + kk * 4u + c, by, p;
                if (k < 80u) { by = k & 15u; p = k >> 4; } else if (k < 120u) { by = 16u + ((k - 80u) & 7u); p = (k - 80u) >> 3; } else { by = 24u + ((k - 120u) & 1u); p = (k - 120u) >> 1; }
                uint q = (q7[by >> 2] >> (8u * (by & 3u))) & 255u;
                uint pw = p == 0u ? 1u : p == 1u ? 3u : p == 2u ? 9u : p == 3u ? 27u : 81u;   // trit p = ((q * 3^p) mod 256) * 3 >> 8
                o[c] = float(int(((q * pw) & 255u) * 3u >> 8) - 1) * d;
            }
            v[kk] = o; }
    } else if (ty == 8u) {                              // PQ2_0: d f16, 32 bytes of 2-bit codes (0=-1, 1=0, 2=+1), 128 per block
        uint b = rb + (it >> 2) * 34u, s = it & 3u;
        float d = f16at(b);
        [[unroll]] for (uint i = 0u; i < 4u; i++) {
            uint u = uint(w16[(b + 2u + 8u * s + 2u * i) >> 1]);
            v[2u * i] = (vec4((uvec4(u) >> uvec4(0u, 2u, 4u, 6u)) & 3u) - 1.0) * d;
            v[2u * i + 1u] = (vec4((uvec4(u) >> uvec4(8u, 10u, 12u, 14u)) & 3u) - 1.0) * d;
        }
    } else if (ty == 10u || ty == 11u || ty == 13u) {   // BF16 / F16 (13 = BF16 with padded rows): 32 weights = 64 bytes
        // 16-byte loads only when this 64-byte block starts 16-byte aligned (always for 13 / padded rows; safetensors and
        // GGUF tensors may be only 2-8 byte aligned). Otherwise 2-byte loads. Both read exactly the block's 64 bytes.
        uint bo = rb + it * 64u; uvec4 u[4];
        if ((bo & 15u) == 0u) { [[unroll]] for (uint q = 0u; q < 4u; q++) u[q] = w4[(bo >> 4) + q]; }
        else { uint h = bo >> 1; [[unroll]] for (uint q = 0u; q < 4u; q++) { uint b = h + 8u * q;
            u[q] = uvec4(uint(w16[b]) | (uint(w16[b + 1u]) << 16), uint(w16[b + 2u]) | (uint(w16[b + 3u]) << 16),
                         uint(w16[b + 4u]) | (uint(w16[b + 5u]) << 16), uint(w16[b + 6u]) | (uint(w16[b + 7u]) << 16)); } }
        [[unroll]] for (uint q = 0u; q < 4u; q++) {
            if (ty == 11u) { v[2u * q] = vec4(unpackHalf2x16(u[q].x), unpackHalf2x16(u[q].y)); v[2u * q + 1u] = vec4(unpackHalf2x16(u[q].z), unpackHalf2x16(u[q].w)); }
            else { v[2u * q] = uintBitsToFloat(uvec4(u[q].x << 16, u[q].x & 0xFFFF0000u, u[q].y << 16, u[q].y & 0xFFFF0000u)); v[2u * q + 1u] = uintBitsToFloat(uvec4(u[q].z << 16, u[q].z & 0xFFFF0000u, u[q].w << 16, u[q].w & 0xFFFF0000u)); }
        }
    } else if (ty == 0u || ty == 2u) {
        uint bs = ty == 0u ? 144u : 176u, b = rb + (it >> 3) * bs, s = it & 7u, jj = s >> 1, sh = (s & 1u) * 4u;
        uvec4 hd = w4[b >> 4];
        vec2 dm = unpackHalf2x16(hd.x);
        uint sc_, mn_;
        if (s < 4u) { sc_ = sbyte(hd, 4u + s) & 63u; mn_ = sbyte(hd, 8u + s) & 63u; }
        else { sc_ = (sbyte(hd, 8u + s) & 0xFu) | ((sbyte(hd, s) >> 6) << 4); mn_ = (sbyte(hd, 8u + s) >> 4) | ((sbyte(hd, 4u + s) >> 6) << 4); }
        float d = dm.x * float(sc_), m = dm.y * float(mn_);
        uint qb = (b + (ty == 0u ? 16u : 48u) + jj * 32u) >> 4;
        uvec4 qa = w4[qb], qc = w4[qb + 1u];
        uint qq[8] = uint[8](qa.x, qa.y, qa.z, qa.w, qc.x, qc.y, qc.z, qc.w);
        uint hh[8];
        if (ty == 2u) { uvec4 ha = w4[(b + 16u) >> 4], hc = w4[(b + 32u) >> 4]; hh = uint[8](ha.x, ha.y, ha.z, ha.w, hc.x, hc.y, hc.z, hc.w); }
        [[unroll]] for (uint k = 0u; k < 8u; k++) {
            uint q = qq[k]; uvec4 n4 = (uvec4(q, q >> 8, q >> 16, q >> 24) >> sh) & 0xFu;
            if (ty == 2u) { uint h = hh[k]; n4 |= ((uvec4(h, h >> 8, h >> 16, h >> 24) >> s) & 1u) << 4; }
            v[k] = vec4(n4) * d - m;
        }
    } else if (ty == 1u) {
        uint b = rb + (it >> 3) * 210u, s = it & 7u, n = s >> 2, q = s & 3u;
        float d = f16at(b + 208u);
        uint qlb = b + n * 64u + (q & 1u) * 32u, qhb = b + 128u + n * 32u, nsh = (q >> 1) * 4u, hsh = q * 2u;
        float c0 = d * float(int(rdb(b + 192u + n * 8u + 2u * q) << 24) >> 24), c1 = d * float(int(rdb(b + 193u + n * 8u + 2u * q) << 24) >> 24);
        [[unroll]] for (uint k = 0u; k < 8u; k++) {
            uint a = rd32(qlb + 4u * k), h = rd32(qhb + 4u * k);
            v[k] = vec4(ivec4((((uvec4(a, a >> 8, a >> 16, a >> 24) >> nsh) & 0xFu) | (((uvec4(h, h >> 8, h >> 16, h >> 24) >> hsh) & 3u) << 4))) - 32) * (k < 4u ? c0 : c1);
        }
    } else if (ty == 3u) {
        uint b = rb + it * 34u; float d = f16at(b);
        [[unroll]] for (uint k = 0u; k < 8u; k++) { uint a = rd32(b + 2u + 4u * k); v[k] = vec4(ivec4(int(a << 24) >> 24, int(a << 16) >> 24, int(a << 8) >> 24, int(a) >> 24)) * d; }
    } else if (ty == 16u) {                    // Q4_0 (rtcore qtype 16): d, qs[16]; low nibbles = weights 0..15, high = 16..31, value q - 8
        uint b = rb + it * 18u; float d = f16at(b);
        [[unroll]] for (uint k = 0u; k < 4u; k++) {
            uint a = rd32(b + 2u + 4u * k); uvec4 by = uvec4(a, a >> 8, a >> 16, a >> 24);
            v[k] = vec4(ivec4(by & 0xFu) - 8) * d;
            v[k + 4u] = vec4(ivec4((by >> 4) & 0xFu) - 8) * d;
        }
    } else if (ty == 15u) {                    // Q5_0: d, qh (u32), qs[16]; low nibbles = weights 0..15, high = 16..31
        uint b = rb + it * 22u; float d = f16at(b); uint qh = rd32(b + 2u);
        [[unroll]] for (uint k = 0u; k < 4u; k++) {
            uint a = rd32(b + 6u + 4u * k); uvec4 j = uvec4(4u * k) + uvec4(0u, 1u, 2u, 3u);
            v[k] = vec4(ivec4((uvec4(a, a >> 8, a >> 16, a >> 24) & 0xFu) | (((uvec4(qh) >> j) & 1u) << 4)) - 16) * d;
            v[k + 4u] = vec4(ivec4(((uvec4(a, a >> 8, a >> 16, a >> 24) >> 4) & 0xFu) | (((uvec4(qh) >> (j + 16u)) & 1u) << 4)) - 16) * d;
        }
    } else if (ty == 4u || ty == 5u) {
        float dl; uint qa4[4];
        if (ty == 4u) { uint b = rb + it * 18u, wi = b >> 2;   // 18-byte blocks are 2-aligned: 5 dword loads + funnel shifts
            uint u0 = w[wi], u1 = w[wi + 1u], u2 = w[wi + 2u], u3 = w[wi + 3u], u4 = w[wi + 4u];
            if ((b & 2u) == 0u) { dl = unpackHalf2x16(u0).x; qa4 = uint[4]((u0 >> 16) | (u1 << 16), (u1 >> 16) | (u2 << 16), (u2 >> 16) | (u3 << 16), (u3 >> 16) | (u4 << 16)); }
            else { dl = unpackHalf2x16(u0 >> 16).x; qa4 = uint[4](u1, u2, u3, u4); } }
        else { uint b = rb + (it >> 3) * 136u, ib = it & 7u;
            uvec2 hd = w2[b >> 3];
            uint ls = ((hd.y >> (8u * (ib >> 1) + 4u * (ib & 1u))) & 0xFu) | ((((hd.x >> 16) >> (2u * ib)) & 3u) << 4);
            dl = unpackHalf2x16(hd.x).x * (float(ls) - 32.0);
            uint qb = (b + 8u + 16u * ib) >> 3; uvec2 a0 = w2[qb], a1 = w2[qb + 1u]; qa4 = uint[4](a0.x, a0.y, a1.x, a1.y); }
        [[unroll]] for (uint k = 0u; k < 4u; k++) {
            uint a = qa4[k];
            v[k] = vec4(KV_IQ4NL[a & 15u], KV_IQ4NL[(a >> 8) & 15u], KV_IQ4NL[(a >> 16) & 15u], KV_IQ4NL[(a >> 24) & 15u]) * dl;
            v[k + 4u] = vec4(KV_IQ4NL[(a >> 4) & 15u], KV_IQ4NL[(a >> 12) & 15u], KV_IQ4NL[(a >> 20) & 15u], KV_IQ4NL[(a >> 28) & 15u]) * dl;
        }
    } else if (ty == 6u) {
        uint b = rb + (it >> 3) * 110u, s = it & 7u, n = s >> 2, j = s & 3u;
        float d = f16at(b + 108u);
        [[unroll]] for (uint hf = 0u; hf < 2u; hf++) {
            uint k = 8u * n + 2u * j + hf;
            uint lo = rdb(b + 96u + (k & 7u)); lo = k < 8u ? (lo & 0xFu) : (lo >> 4);
            uint hi = (rdb(b + 96u + 8u + (k & 3u)) >> (2u * (k >> 2))) & 3u;
            float sc = d * float(int(lo | (hi << 4)) - 32);
            [[unroll]] for (uint kk = 0u; kk < 4u; kk++) {
                uint q = rd32(b + 32u + n * 32u + hf * 16u + 4u * kk), hm = rd32(b + hf * 16u + 4u * kk);
                uvec4 qv = (uvec4(q, q >> 8, q >> 16, q >> 24) >> (2u * j)) & 3u;
                uvec4 hv = (uvec4(hm, hm >> 8, hm >> 16, hm >> 24) >> (4u * n + j)) & 1u;
                v[hf * 4u + kk] = vec4(ivec4(qv) - 4 * (1 - ivec4(hv))) * sc;
            }
        }
    } else {
        uint b = rb + (it >> 3) * 110u, ib = it & 7u;
        float db = f16at(b) * float(1u + 2u * ((rdb(b + 106u + ib / 2u) >> (4u * (ib & 1u))) & 0xFu));
        uint qh = rdb(b + 66u + ib);
        [[unroll]] for (uint l = 0u; l < 4u; l++) {
            uint g1 = IQ3S_GRID[rdb(b + 2u + 8u * ib + 2u * l) | ((qh << (8u - 2u * l)) & 256u)];
            uint g2 = IQ3S_GRID[rdb(b + 3u + 8u * ib + 2u * l) | ((qh << (7u - 2u * l)) & 256u)];
            uint sg = rdb(b + 74u + 4u * ib + l);
            vec4 m1 = vec4(uvec4(g1, g1 >> 8, g1 >> 16, g1 >> 24) & 0xFFu), m2 = vec4(uvec4(g2, g2 >> 8, g2 >> 16, g2 >> 24) & 0xFFu);
            vec4 s1 = vec4(1.0) - 2.0 * vec4((uvec4(sg) >> uvec4(0u, 1u, 2u, 3u)) & 1u), s2 = vec4(1.0) - 2.0 * vec4((uvec4(sg) >> uvec4(4u, 5u, 6u, 7u)) & 1u);
            v[2u * l] = m1 * s1 * db; v[2u * l + 1u] = m2 * s2 * db;
        }
    }
}
