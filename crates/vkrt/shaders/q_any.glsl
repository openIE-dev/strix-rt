// Pluggable weight formats: dot product of one 32-weight sub-block of a row with x (vec4 XV(i) supplied by the includer).
// Weights are byte-addressed through `uint w[]` (binding of the includer). Type codes (rtcore::qtype):
// 0 Q4_K, 1 Q6_K, 2 Q5_K, 3 Q8_0, 4 IQ4_NL, 5 IQ4_XS, 6 Q3_K, 7 IQ3_S, 8 PQ2_0, 9 PTQ1_0 (Prism ternary), 10 BF16, 11 F16, 15 Q5_0.
// Items are 32 weights except PTQ1_0 (one 128-weight block per item, see item_elems). Transcribed from ggml's dequantize_row_*.
// Codebooks live in f32w[0..528) (rtcore::TBL_FLOATS: iq3s_grid bits, then kvalues_iq4nl) and are staged in shared
// memory by q_any_init(): a const array indexed dynamically gets lowered to scratch (256 VGPRs + 128 KiB scratch, ~1000x slower).
shared uint IQ3S_GRID[512];
shared float KV_IQ4NL[16];
void q_any_init(uint types) {
    bool need = false;
    for (uint s = 0u; s < 5u; s++) { uint t = (types >> (4u * s)) & 0xFu; need = need || t == 4u || t == 5u || t == 7u; }
    if (!need) return;
    for (uint i = gl_LocalInvocationID.x; i < 528u; i += gl_WorkGroupSize.x) {
        if (i < 512u) IQ3S_GRID[i] = floatBitsToUint(f32w[i]); else KV_IQ4NL[i - 512u] = f32w[i];
    }
    barrier();
}
uint rdb(uint bo) { return (w[bo >> 2] >> (8u * (bo & 3u))) & 0xFFu; }
uint rd16(uint bo) { return (bo & 3u) == 3u ? (rdb(bo) | (rdb(bo + 1u) << 8)) : ((w[bo >> 2] >> (8u * (bo & 3u))) & 0xFFFFu); }
// Rows start 4-byte aligned (rtcore::GQ::segs checks), so every block field read as 32 bits is at least 2-aligned.
uint rd32(uint bo) { uint h = bo >> 1; return uint(w16[h]) | (uint(w16[h + 1u]) << 16); }
uint rd32a(uint bo) { return w[bo >> 2]; }
uint sbyte(uvec4 v, uint i) { uint u = i < 4u ? v.x : i < 8u ? v.y : i < 12u ? v.z : v.w; return (u >> (8u * (i & 3u))) & 0xFFu; }   // 4-aligned (Q4_K/Q5_K/IQ4_XS blocks are 144/176/136 bytes)
float f16at(uint bo) { return unpackHalf2x16(rd16(bo)).x; }
uint blk_elems(uint ty) { return ty == 12u ? 256u : (ty == 3u || ty == 4u || ty >= 10u) ? 32u : (ty == 8u || ty == 9u) ? 128u : 256u; }
uint item_elems(uint ty) { return (ty == 8u || ty == 9u || ty == 12u) ? 128u : 32u; }
uint blk_bytes(uint ty) { return ty == 0u ? 144u : ty == 1u ? 210u : ty == 2u ? 176u : ty == 3u ? 34u : ty == 4u ? 18u : ty == 5u ? 136u : ty == 8u ? 34u : ty == 9u ? 28u : ty == 12u ? 68u : ty == 15u ? 22u : ty >= 10u ? 64u : 110u; }
uint row_bytes_of(uint ty, uint K) { return K / blk_elems(ty) * blk_bytes(ty); }
void kscale(uint sb, uint j, out float sc, out float mn) { // Q4_K/Q5_K 6-bit scale/min j (0..7), scales at byte sb
    if (j < 4u) { sc = float(rdb(sb + j) & 63u); mn = float(rdb(sb + j + 4u) & 63u); }
    else { sc = float((rdb(sb + j + 4u) & 0xFu) | ((rdb(sb + j - 4u) >> 6) << 4)); mn = float((rdb(sb + j + 4u) >> 4) | ((rdb(sb + j) >> 6) << 4)); }
}
// dot of sub-block `it` (32 weights) of the row starting at byte rb, with x starting at vec4 index xb4
float dot32x(uint ty, uint rb, uint it, uint xb4, vec4 xa[8]) {   // xa = XV(xb4 .. xb4+7) (unused by PTQ1_0)
    float acc = 0.0;
    if (ty == 0u || ty == 2u) {               // Q4_K / Q5_K: d, dmin, scales[12], (Q5: qh[32]), qs[128]; blocks 16-aligned -> uvec4 loads
        uint bs = ty == 0u ? 144u : 176u, b = rb + (it >> 3) * bs, s = it & 7u, jj = s >> 1, sh = (s & 1u) * 4u;
        uvec4 hd = w4[b >> 4];
        vec2 dm = unpackHalf2x16(hd.x);
        uint sc_, mn_;
        if (s < 4u) { sc_ = sbyte(hd, 4u + s) & 63u; mn_ = sbyte(hd, 8u + s) & 63u; }
        else { sc_ = (sbyte(hd, 8u + s) & 0xFu) | ((sbyte(hd, s) >> 6) << 4); mn_ = (sbyte(hd, 8u + s) >> 4) | ((sbyte(hd, 4u + s) >> 6) << 4); }
        uint qb = (b + (ty == 0u ? 16u : 48u) + jj * 32u) >> 4;
        uvec4 qa = w4[qb], qc = w4[qb + 1u];
        uint qq[8] = uint[8](qa.x, qa.y, qa.z, qa.w, qc.x, qc.y, qc.z, qc.w);
        uint hh[8];
        if (ty == 2u) { uvec4 ha = w4[(b + 16u) >> 4], hc = w4[(b + 32u) >> 4]; hh = uint[8](ha.x, ha.y, ha.z, ha.w, hc.x, hc.y, hc.z, hc.w); }
        float sq = 0.0, sx = 0.0;
        [[unroll]] for (uint k = 0u; k < 8u; k++) {
            uint q = qq[k]; uvec4 n4 = (uvec4(q, q >> 8, q >> 16, q >> 24) >> sh) & 0xFu;
            if (ty == 2u) { uint h = hh[k]; n4 |= ((uvec4(h, h >> 8, h >> 16, h >> 24) >> s) & 1u) << 4; }
            sq += dot(vec4(n4), xa[k]); sx += dot(xa[k], vec4(1.0));
        }
        acc = dm.x * float(sc_) * sq - dm.y * float(mn_) * sx;
    } else if (ty == 1u) {                     // Q6_K: ql[128], qh[64], scales[16] (int8), d
        uint b = rb + (it >> 3) * 210u, s = it & 7u, n = s >> 2, q = s & 3u;
        float d = f16at(b + 208u);
        uint qlb = b + n * 64u + (q & 1u) * 32u, qhb = b + 128u + n * 32u, nsh = (q >> 1) * 4u, hsh = q * 2u;
        float s0 = 0.0, s1 = 0.0;
        [[unroll]] for (uint k = 0u; k < 8u; k++) {
            uint a = rd32(qlb + 4u * k), h = rd32(qhb + 4u * k);
            vec4 v = vec4(ivec4((((uvec4(a, a >> 8, a >> 16, a >> 24) >> nsh) & 0xFu) | (((uvec4(h, h >> 8, h >> 16, h >> 24) >> hsh) & 3u) << 4))) - 32);
            if (k < 4u) s0 += dot(v, xa[k]); else s1 += dot(v, xa[k]);
        }
        float c0 = float(int(rdb(b + 192u + n * 8u + 2u * q) << 24) >> 24), c1 = float(int(rdb(b + 193u + n * 8u + 2u * q) << 24) >> 24);
        acc = d * (c0 * s0 + c1 * s1);
    } else if (ty == 3u) {                     // Q8_0: d, qs[32] int8
        uint b = rb + it * 34u; float d = f16at(b);
        [[unroll]] for (uint k = 0u; k < 8u; k++) { uint a = rd32(b + 2u + 4u * k); acc += dot(vec4(ivec4(int(a << 24) >> 24, int(a << 16) >> 24, int(a << 8) >> 24, int(a) >> 24)), xa[k]); }
        acc *= d;
    } else if (ty == 15u) {                    // Q5_0: d, qh (u32), qs[16]; low nibbles = weights 0..15, high = 16..31; blocks 2-aligned
        uint b = rb + it * 22u; float d = f16at(b); uint qh = rd32(b + 2u);
        float s = 0.0;
        [[unroll]] for (uint k = 0u; k < 4u; k++) {
            uint a = rd32(b + 6u + 4u * k); uvec4 j = uvec4(4u * k) + uvec4(0u, 1u, 2u, 3u);
            uvec4 lo = (uvec4(a, a >> 8, a >> 16, a >> 24) & 0xFu) | (((uvec4(qh) >> j) & 1u) << 4);
            uvec4 hi = ((uvec4(a, a >> 8, a >> 16, a >> 24) >> 4) & 0xFu) | (((uvec4(qh) >> (j + 16u)) & 1u) << 4);
            s += dot(vec4(ivec4(lo) - 16), xa[k]) + dot(vec4(ivec4(hi) - 16), xa[k + 4u]);
        }
        acc = d * s;
    } else if (ty == 4u || ty == 5u) {         // IQ4_NL (d, qs[16]) / IQ4_XS (d, scales_h, scales_l[4], qs[128])
        float dl; uint qa4[4];
        if (ty == 4u) { uint b = rb + it * 18u; dl = f16at(b); [[unroll]] for (uint k = 0u; k < 4u; k++) qa4[k] = rd32(b + 2u + 4u * k); }
        else { uint b = rb + (it >> 3) * 136u, ib = it & 7u;  // IQ4_XS blocks are 8-aligned -> uvec2 loads
            uvec2 hd = w2[b >> 3];
            uint ls = ((hd.y >> (8u * (ib >> 1) + 4u * (ib & 1u))) & 0xFu) | ((((hd.x >> 16) >> (2u * ib)) & 3u) << 4);
            dl = unpackHalf2x16(hd.x).x * (float(ls) - 32.0);
            uint qb = (b + 8u + 16u * ib) >> 3; uvec2 a0 = w2[qb], a1 = w2[qb + 1u]; qa4 = uint[4](a0.x, a0.y, a1.x, a1.y); }
        float s = 0.0;
        [[unroll]] for (uint k = 0u; k < 4u; k++) {
            uint a = qa4[k];
            vec4 lo = vec4(KV_IQ4NL[a & 15u], KV_IQ4NL[(a >> 8) & 15u], KV_IQ4NL[(a >> 16) & 15u], KV_IQ4NL[(a >> 24) & 15u]);
            vec4 hi = vec4(KV_IQ4NL[(a >> 4) & 15u], KV_IQ4NL[(a >> 12) & 15u], KV_IQ4NL[(a >> 20) & 15u], KV_IQ4NL[(a >> 28) & 15u]);
            s += dot(lo, xa[k]) + dot(hi, xa[k + 4u]);
        }
        acc = dl * s;
    } else if (ty == 6u) {                     // Q3_K: hmask[32], qs[64], scales[12], d
        uint b = rb + (it >> 3) * 110u, s = it & 7u, n = s >> 2, j = s & 3u;
        float d = f16at(b + 108u);
        float part[2];
        [[unroll]] for (uint hf = 0u; hf < 2u; hf++) {
            uint k = 8u * n + 2u * j + hf;   // scale index
            uint lo = rdb(b + 96u + (k & 7u)); lo = k < 8u ? (lo & 0xFu) : (lo >> 4);
            uint hi = (rdb(b + 96u + 8u + (k & 3u)) >> (2u * (k >> 2))) & 3u;
            float sc = float(int(lo | (hi << 4)) - 32);
            float sm = 0.0;
            [[unroll]] for (uint kk = 0u; kk < 4u; kk++) {
                uint q = rd32(b + 32u + n * 32u + hf * 16u + 4u * kk), hm = rd32(b + hf * 16u + 4u * kk);
                uvec4 qv = (uvec4(q, q >> 8, q >> 16, q >> 24) >> (2u * j)) & 3u;
                uvec4 hv = (uvec4(hm, hm >> 8, hm >> 16, hm >> 24) >> (4u * n + j)) & 1u;
                sm += dot(vec4(ivec4(qv) - 4 * (1 - ivec4(hv))), xa[hf * 4u + kk]);
            }
            part[hf] = sc * sm;
        }
        acc = d * (part[0] + part[1]);
    } else if (ty == 8u) {                     // PQ2_0: d, qs[32] (2 bits/weight: 00=-1, 01=0, 10=+1); item = whole 34-byte block,
        uint bo = rb + it * 34u, bw = bo >> 2;     // read as 9 aligned dwords (blocks are 2-aligned)
        bool odd = (bo & 2u) != 0u;
        uint wv[9]; [[unroll]] for (uint k = 0u; k < 9u; k++) wv[k] = w[bw + k];
        float d = unpackHalf2x16(odd ? (wv[0] >> 16) : wv[0]).x, s = 0.0;
        [[unroll]] for (uint k = 0u; k < 8u; k++) {
            uint v = odd ? wv[k + 1u] : ((wv[k] >> 16) | (wv[k + 1u] << 16));
            s += dot(vec4(uvec4(v, v >> 2, v >> 4, v >> 6) & 3u) - 1.0, XV(xb4 + 4u * k))
               + dot(vec4(uvec4(v >> 8, v >> 10, v >> 12, v >> 14) & 3u) - 1.0, XV(xb4 + 4u * k + 1u))
               + dot(vec4(uvec4(v >> 16, v >> 18, v >> 20, v >> 22) & 3u) - 1.0, XV(xb4 + 4u * k + 2u))
               + dot(vec4(uvec4(v >> 24, v >> 26, v >> 28, v >> 30) & 3u) - 1.0, XV(xb4 + 4u * k + 3u));
        }
        acc = d * s;
    } else if (ty == 9u) {                     // PTQ1_0: qs[24], qh[2], d; base-3 fixed-point trits; item = whole 128-weight block
        uint wb = (rb + it * 28u) >> 2;
        float s = 0.0;
        uvec4 q0 = (uvec4(w[wb]) >> uvec4(0, 8, 16, 24)) & 255u, q1 = (uvec4(w[wb + 1u]) >> uvec4(0, 8, 16, 24)) & 255u;
        uvec4 q2 = (uvec4(w[wb + 2u]) >> uvec4(0, 8, 16, 24)) & 255u, q3 = (uvec4(w[wb + 3u]) >> uvec4(0, 8, 16, 24)) & 255u;
        [[unroll]] for (uint n = 0u; n < 5u; n++) {
            q0 *= 3u; q1 *= 3u; q2 *= 3u; q3 *= 3u;
            s += dot(vec4(q0 >> 8) - 1.0, XV(xb4 + n * 4u)) + dot(vec4(q1 >> 8) - 1.0, XV(xb4 + n * 4u + 1u))
               + dot(vec4(q2 >> 8) - 1.0, XV(xb4 + n * 4u + 2u)) + dot(vec4(q3 >> 8) - 1.0, XV(xb4 + n * 4u + 3u));
            q0 &= 255u; q1 &= 255u; q2 &= 255u; q3 &= 255u;
        }
        uvec4 q4 = (uvec4(w[wb + 4u]) >> uvec4(0, 8, 16, 24)) & 255u, q5 = (uvec4(w[wb + 5u]) >> uvec4(0, 8, 16, 24)) & 255u;
        [[unroll]] for (uint n = 0u; n < 5u; n++) {
            q4 *= 3u; q5 *= 3u;
            s += dot(vec4(q4 >> 8) - 1.0, XV(xb4 + 20u + n * 2u)) + dot(vec4(q5 >> 8) - 1.0, XV(xb4 + 21u + n * 2u));
            q4 &= 255u; q5 &= 255u;
        }
        uint w6 = w[wb + 6u]; uvec2 qh = uvec2(w6, w6 >> 8) & 255u;
        vec4 xa_ = XV(xb4 + 30u), xc_ = XV(xb4 + 31u);
        [[unroll]] for (uint n = 0u; n < 4u; n++) {
            qh *= 3u; vec2 t = vec2(qh >> 8) - 1.0;
            s += dot(t, n == 0u ? xa_.xy : (n == 1u ? xa_.zw : (n == 2u ? xc_.xy : xc_.zw)));
            qh &= 255u;
        }
        acc = unpackHalf2x16(w6 >> 16).x * s;
    } else if (ty == 12u) {                    // TQ2_1 (kitten fork): 256 weights = qs[64] + d[2]; item = one 128-weight half
        uint b = rb + (it >> 1) * 68u, g = it & 1u, qb = (b + g * 32u) >> 2;
        float s = 0.0;
        [[unroll]] for (uint k = 0u; k < 8u; k++) { uint v = w[qb + k];
            [[unroll]] for (uint l = 0u; l < 4u; l++) s += dot(vec4((uvec4(v) >> uvec4(2u * l, 2u * l + 8u, 2u * l + 16u, 2u * l + 24u)) & 3u) - 1.0, XV(xb4 + 8u * l + k)); }
        uint dw = w[(b + 64u) >> 2];
        acc = unpackHalf2x16(g == 0u ? dw : (dw >> 16)).x * s;
    } else if (ty == 10u || ty == 11u) {       // BF16 / F16: 32 weights = 64 bytes = 4 uvec4
        uint b4 = (rb + it * 64u) >> 4;
        [[unroll]] for (uint k = 0u; k < 4u; k++) {
            uvec4 u = w4[b4 + k];
            vec4 lo, hi;
            if (ty == 10u) { lo = uintBitsToFloat(uvec4(u.x << 16, u.x & 0xFFFF0000u, u.y << 16, u.y & 0xFFFF0000u)); hi = uintBitsToFloat(uvec4(u.z << 16, u.z & 0xFFFF0000u, u.w << 16, u.w & 0xFFFF0000u)); }
            else { lo = vec4(unpackHalf2x16(u.x), unpackHalf2x16(u.y)); hi = vec4(unpackHalf2x16(u.z), unpackHalf2x16(u.w)); }
            acc += dot(lo, xa[2u * k]) + dot(hi, xa[2u * k + 1u]);
        }
    } else {                                   // IQ3_S: d, qs[64], qh[8], signs[32], scales[4]
        uint b = rb + (it >> 3) * 110u, ib = it & 7u;
        float db = f16at(b) * float(1u + 2u * ((rdb(b + 106u + ib / 2u) >> (4u * (ib & 1u))) & 0xFu));
        uint qh = rdb(b + 66u + ib);
        [[unroll]] for (uint l = 0u; l < 4u; l++) {
            uint g1 = IQ3S_GRID[rdb(b + 2u + 8u * ib + 2u * l) | ((qh << (8u - 2u * l)) & 256u)];
            uint g2 = IQ3S_GRID[rdb(b + 3u + 8u * ib + 2u * l) | ((qh << (7u - 2u * l)) & 256u)];
            uint sg = rdb(b + 74u + 4u * ib + l);
            vec4 m1 = vec4(uvec4(g1, g1 >> 8, g1 >> 16, g1 >> 24) & 0xFFu), m2 = vec4(uvec4(g2, g2 >> 8, g2 >> 16, g2 >> 24) & 0xFFu);
            vec4 s1 = vec4(1.0) - 2.0 * vec4((uvec4(sg) >> uvec4(0u, 1u, 2u, 3u)) & 1u), s2 = vec4(1.0) - 2.0 * vec4((uvec4(sg) >> uvec4(4u, 5u, 6u, 7u)) & 1u);
            acc += dot(m1 * s1, xa[2u * l]) + dot(m2 * s2, xa[2u * l + 1u]);
        }
        acc *= db;
    }
    return acc;
}

float dot32(uint ty, uint rb, uint it, uint xb4) {
    vec4 xa[8]; [[unroll]] for (uint k = 0u; k < 8u; k++) xa[k] = XV(xb4 + k);
    return dot32x(ty, rb, it, xb4, xa);
}
