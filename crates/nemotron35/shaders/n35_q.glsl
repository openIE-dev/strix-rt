// Nemotron 3.5 weight dot: q_any.glsl formats plus Q4_0 (code 16; rtcore::qtype maps GGML type 2 -> 16).
// Q4_0 cannot use the 4-bit packed `types` field of gemvq, so only these full-u32 `ty` kernels accept it.
#include "q_any.glsl"
float dotq(uint ty, uint rb, uint it, uint xb4) {
    if (ty == 16u) {                           // Q4_0: d (f16), qs[16]; low nibbles = weights 0..15, high = 16..31, value q - 8
        uint b = rb + it * 18u; float d = f16at(b), s = 0.0;
        [[unroll]] for (uint k = 0u; k < 4u; k++) {
            uint a = rd32(b + 2u + 4u * k); uvec4 v = uvec4(a, a >> 8, a >> 16, a >> 24);
            s += dot(vec4(ivec4(v & 0xFu) - 8), XV(xb4 + k)) + dot(vec4(ivec4((v >> 4) & 0xFu) - 8), XV(xb4 + k + 4u));
        }
        return d * s;
    }
    return dot32(ty, rb, it, xb4);
}
uint rowb(uint ty, uint k) { return ty == 16u ? k / 32u * 18u : row_bytes_of(ty, k); }
