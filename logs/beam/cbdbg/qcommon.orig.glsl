// B-bit symmetric group-32 quant: group g of a row = BITS uint32 words (code i at bit BITS*i, LE bitstream), scale fp16 per group. value = (q - C) * s, C = (2^BITS-1)/2.
// CB (spec constant 3, BEAM_CBLUT=file): per-layer 8-entry codebook in push constants, value = lut[q] * s (MAC path only).
// ABM (spec constant 2, BEAM_ABM): 0 = multiply-accumulate per weight (default);
//   1 = add-before-multiply, bit-plane form (T-MAC style): sum q*x = sum_k 2^k * (sum of x where bit k of q is set); value = that - C*sum(x); BITS adds-only passes + BITS+1 multiplies per group;
//   2 = add-before-multiply, code-bin form (LUT-GEMM / arXiv 2610.31913 style): bin[q] += x, then sum_b (b - C) * bin[b] (2^BITS multiplies per group).
uint code_at(uint wbase, uint i, uint BITS) {
    uint bit = BITS * i, w = bit >> 5, sh = bit & 31u; uint v = cw[wbase + w] >> sh;
    if (sh + BITS > 32u) v |= cw[wbase + w + 1u] << (32u - sh);
    return v & ((1u << BITS) - 1u);
}
float qdot32(uint wbase, uint sc16, uint xo, uint BITS) {
    const uint MASK = (1u << BITS) - 1u; float CEN = float(MASK) * 0.5; float acc = 0.0;
    if (ABM == 1u) {
        float sx = 0.0, p0 = 0.0, p1 = 0.0, p2 = 0.0, p3 = 0.0;
        for (uint i = 0u; i < 32u; i++) {
            uint q = code_at(wbase, i, BITS); float xv = x[xo + i]; sx += xv;
            p0 += (q & 1u) != 0u ? xv : 0.0; p1 += (q & 2u) != 0u ? xv : 0.0; p2 += (q & 4u) != 0u ? xv : 0.0; p3 += (q & 8u) != 0u ? xv : 0.0;
        }
        acc = p0 + 2.0 * p1 + 4.0 * p2 + 8.0 * p3 - CEN * sx;
    } else if (ABM == 2u) {
        float b[16]; for (uint k = 0u; k < 16u; k++) b[k] = 0.0;
        for (uint i = 0u; i < 32u; i++) b[code_at(wbase, i, BITS)] += x[xo + i];
        for (uint k = 0u; k <= MASK; k++) acc += (float(k) - CEN) * b[k];
    } else {
        if (CB != 0u) { for (uint i = 0u; i < 32u; i++) acc += pc.lut[code_at(wbase, i, BITS)] * x[xo + i]; }
        else { for (uint i = 0u; i < 32u; i++) acc += (float(code_at(wbase, i, BITS)) - CEN) * x[xo + i]; }
    }
    return acc * unpackHalf2x16(sc16).x;
}
uint scale16(uint byteoff) { return (ss[byteoff >> 2] >> ((byteoff & 2u) * 8u)) & 0xffffu; }
