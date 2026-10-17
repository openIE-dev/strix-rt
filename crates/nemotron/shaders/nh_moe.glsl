// Nemotron-H MoE expert GEMVs (decode), any rtcore::qtype via q_any. LPR lanes per row, 256/LPR rows per workgroup.
// DOWN == 0 (up):   h[s*ff + r] = relu(W_up[e_s][r] . x)^2              grid (ff/RPW, top_k)
// DOWN == 1 (down): y[r] += sum_s w_s * (W_down[e_s][r] . h[s*ff ..])    grid (dim/RPW)
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_clustered : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
layout(local_size_x = 256) in;
layout(constant_id = 0) const uint DOWN = 0u;
#ifndef LPR
#define LPR 32u
#endif
const uint RPW = 256u / LPR;
layout(std430, binding = 0) readonly buffer W { uint w[]; };
layout(std430, binding = 0) readonly buffer W2 { uvec2 w2[]; };
layout(std430, binding = 0) readonly buffer W4 { uvec4 w4[]; };
layout(std430, binding = 0) readonly buffer W16 { uint16_t w16[]; };
layout(std430, binding = 1) readonly buffer X { vec4 x4[]; };
layout(std430, binding = 2) buffer Y { float y[]; };
layout(std430, binding = 3) readonly buffer F { float f32w[]; };
layout(std430, binding = 4) readonly buffer AU { uint au[]; };
layout(push_constant) uniform PC { uint w_off; uint stride; uint n; uint k; uint ty; uint top_k; } pc;
uint g_xo = 0u;
vec4 XV(uint i) { return x4[g_xo + i]; }
#include "q_any.glsl"
void main() {
    q_any_init(pc.ty);
    uint lid = gl_LocalInvocationID.x, r = gl_WorkGroupID.x * RPW + lid / LPR, ln = lid % LPR;
    uint rbytes = row_bytes_of(pc.ty, pc.k), nit = pc.k / 32u;
    if (DOWN == 0u) {
        uint s = gl_WorkGroupID.y, e = au[168u + s];
        float a = 0.0;
        if (r < pc.n) { uint rb = pc.w_off + e * pc.stride + r * rbytes; for (uint it = ln; it < nit; it += LPR) a += dot32(pc.ty, rb, it, it * 8u); }
        a = subgroupClusteredAdd(a, LPR);
        if (ln == 0u && r < pc.n) { float v = max(a, 0.0); y[s * pc.n + r] = v * v; }
    } else {
        float tot = 0.0;
        for (uint s = 0u; s < pc.top_k; s++) {
            uint e = au[168u + s]; g_xo = s * pc.k / 4u;
            float a = 0.0;
            if (r < pc.n) { uint rb = pc.w_off + e * pc.stride + r * rbytes; for (uint it = ln; it < nit; it += LPR) a += dot32(pc.ty, rb, it, it * 8u); }
            a = subgroupClusteredAdd(a, LPR);
            tot += uintBitsToFloat(au[176u + s]) * a;
        }
        if (ln == 0u && r < pc.n) y[r] += tot;
    }
}
