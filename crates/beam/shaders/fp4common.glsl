// NVFP4 (modelopt): weight U8 [N,K/2] two e2m1 per byte (low nibble = even k), scale F8_E4M3 per 16 elements, F32 global scale per tensor.
shared float lut4[16]; shared float lut8[256];
void init_luts() {
    uint t = gl_LocalInvocationID.x;
    { uint e = t; uint s = e >> 7, ex = (e >> 3) & 15u, m = e & 7u; float v;
      if (ex == 0u) v = float(m) * (1.0 / 512.0); else v = uintBitsToFloat(((ex + 120u) << 23) | (m << 20));
      lut8[t] = (s != 0u) ? -v : v; }
    if (t < 16u) { uint m = t & 7u; float mag = (m < 4u) ? float(m) * 0.5 : (m == 4u ? 2.0 : (m == 5u ? 3.0 : (m == 6u ? 4.0 : 6.0)));
      lut4[t] = (t & 8u) != 0u ? -mag : mag; }
    barrier();
}
// dot of 32 consecutive elements (one uvec4 of weights, two fp8 scales packed in sc16) with x[xo..xo+32)
float dot32(uvec4 w, uint sc16, uint xo) {
    float tot = 0.0;
    for (uint q = 0u; q < 4u; q++) {
        uint u = (q == 0u) ? w.x : (q == 1u) ? w.y : (q == 2u) ? w.z : w.w;
        float p = 0.0; uint xb = xo + q * 8u;
        for (uint b = 0u; b < 4u; b++) { uint by = (u >> (8u * b)) & 255u;
            p += lut4[by & 15u] * x[xb + 2u * b] + lut4[by >> 4] * x[xb + 2u * b + 1u]; }
        float s = lut8[(q < 2u) ? (sc16 & 255u) : ((sc16 >> 8) & 255u)];
        tot += p * s;
    }
    return tot;
}
