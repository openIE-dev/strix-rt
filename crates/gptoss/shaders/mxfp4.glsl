// MXFP4 (GGML type 39): 17-byte blocks (e8m0 scale, 16 bytes of nibbles; element j = low nibble of byte j, j+16 = high).
// Value = kvalues_mxfp4[nibble] * e8m0_to_fp32_half(e), kvalues = {0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12}.
uint rdb(uint bo) { return (w[bo >> 2] >> (8u * (bo & 3u))) & 0xFFu; }
uint rdu32(uint bo) { uint s = bo & 3u, a = w[bo >> 2]; return s == 0u ? a : ((a >> (8u * s)) | (w[(bo >> 2) + 1u] << (32u - 8u * s))); }
float e8h(uint e) { return uintBitsToFloat(e < 2u ? (0x00200000u << e) : ((e - 1u) << 23)); }
vec4 kv4(uvec4 n) { uvec4 m = n & 7u; vec4 a = vec4(mix(mix(2u * m - 4u, uvec4(12u), equal(m, uvec4(7u))), m, lessThan(m, uvec4(5u)))); return mix(a, -a, notEqual(n & 8u, uvec4(0u))); }
// dot of 32-weight block at byte bo with x[xi .. xi+32): 5 aligned dword loads + funnel shifts (blocks are 1-aligned)
float mx_dot(uint bo, uint xi) {
    uint a = bo >> 2, sh = 8u * (bo & 3u);
    uint v0 = w[a], v1 = w[a + 1u], v2 = w[a + 2u], v3 = w[a + 3u], v4 = w[a + 4u];
    uint u[5] = uint[5](v0, v1, v2, v3, v4);
    uint b[5]; [[unroll]] for (uint k = 0u; k < 4u; k++) b[k] = sh == 0u ? u[k] : ((u[k] >> sh) | (u[k + 1u] << (32u - sh)));
    b[4] = u[4] >> sh;
    float d = e8h(b[0] & 0xFFu), s = 0.0;
    [[unroll]] for (uint k = 0u; k < 4u; k++) {
        uint q = (b[k] >> 8) | (b[k + 1u] << 24);
        uvec4 bb = uvec4(q, q >> 8, q >> 16, q >> 24) & 0xFFu;
        s += dot(kv4(bb & 15u), XV4(xi + 4u * k)) + dot(kv4(bb >> 4), XV4(xi + 16u + 4u * k));
    }
    return d * s;
}
