#version 450
// Fused decode GEMV: up to 3 row segments (e.g. q|k|v, each Q4_K or Q6_K, with bias) in one dispatch,
// optional fused RMSNorm on the input (scale from a sum-of-squares slot), optional residual accumulate that also
// accumulates the new residual's sum of squares for the next fused norm. SWIGLU: silu(gate.x)*(up.x), Q4_K.
#extension GL_GOOGLE_include_directive : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_clustered : require
#ifndef LPR
#define LPR 32u
#endif
// LPR = lanes per row: 8 divides the item count of every Q4_K/Q6_K shape here exactly (56/112/296/592 items).
#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_atomic_float : require
layout(local_size_x_id = 0) in;
layout(std430, binding = 0) readonly buffer W { uint w[]; };
layout(std430, binding = 0) readonly buffer W16 { uint16_t w16[]; };
layout(std430, binding = 0) readonly buffer W4 { uvec4 w4[]; };
#define VEC4_LOADS
layout(std430, binding = 1) readonly buffer X { vec4 x[]; };
layout(std430, binding = 2) buffer Y { float y[]; };
layout(std430, binding = 3) readonly buffer F { float f32w[]; };
layout(std430, binding = 3) readonly buffer F4 { vec4 f32w4[]; };
layout(std430, binding = 4) buffer A { float sq[]; };
layout(std430, binding = 4) buffer AUI { uint au[]; };
// MoE (wide aux layout): selected expert ids at au[168+s], routing weights at sq[172+s]
#define AUX_MOE_ID 168u
#define AUX_MOE_W 172u
layout(std430, binding = 5) writeonly buffer XW { float xw[]; };
// flag 128: deterministic norm sum (partials per workgroup at aux words D3_PART.., summed in workgroup order by the last
// workgroup; counter at D3_CNT) instead of a float atomic, so greedy decoding is run-to-run reproducible at near-ties
layout(std430, binding = 4) coherent buffer AC3 { uint cnt3[]; };
layout(std430, binding = 4) coherent buffer AS3 { float sq3[]; };
const uint D3_CNT = 1020u, D3_PART = 1024u;
shared uint s_last3; shared float s_red3[32];
layout(push_constant) uniform PC { uint N; uint K; uint flags; uint y_off; uint norm_off; uint sq_in; uint sq_out; uint r1; uint r2; uint types;
    uint w0; uint w1; uint w2; uint b0; uint b1; uint b2; uint wup; float eps; uint bsel; uint misc; } pc;
shared float red[32];
// Fused RMSNorm without touching the inner loop: producers write xw = x * norm_w (flag 16, norm_off = next norm weight),
// consumers read xw and scale the dot product by 1/rms (flag 4, from the sum-of-squares slot sq_in).
uint g_xo = 0u;
vec4 XV(uint i) { return x[g_xo + i]; }
#include "q4k6k.glsl"
// MoE top-k from router probs sq[176+e] (+ selection bias f32w[bsel+e]); weights = probs[sel] normalised to sum 1.
// Every subgroup computes it redundantly (64 values, a few subgroup ops). n_exp <= 2 * subgroup size, top_k <= 8.
uint mids[8]; float mws[8];
void moe_select() {
    uint ne = pc.misc & 0xFFFFu, tk = pc.misc >> 16, l = gl_SubgroupInvocationID, S = gl_SubgroupSize;
    bool h0 = l < ne, h1 = l + S < ne;
    float p0 = h0 ? sq[176u + l] : 0.0, p1 = h1 ? sq[176u + l + S] : 0.0;
    float v0 = h0 ? p0 + f32w[pc.bsel + l] : -3.4e38, v1 = h1 ? p1 + f32w[pc.bsel + l + S] : -3.4e38;
    float tot = 0.0;
    for (uint j = 0u; j < tk; j++) {
        float m = subgroupMax(max(v0, v1));
        uint idx = subgroupMin(v0 == m ? l : (v1 == m ? l + S : 0xFFFFu));
        float w = subgroupMax((idx == l) ? p0 : ((idx == l + S) ? p1 : -1.0));
        mids[j] = idx; mws[j] = w; tot += w;
        if (idx == l) v0 = -3.4e38; if (idx == l + S) v1 = -3.4e38;
    }
    tot = max(tot, 6.103515625e-5);
    for (uint j = 0u; j < tk; j++) mws[j] /= tot;
}
void main() {
    uint wg = gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x;
    uint rps = gl_SubgroupSize / LPR, lane = gl_SubgroupInvocationID % LPR;
    uint row = (wg * gl_NumSubgroups + gl_SubgroupID) * rps + gl_SubgroupInvocationID / LPR;
    bool valid = row < pc.N;
    if ((pc.flags & 96u) != 0u) moe_select();
    uint nb = pc.K / 256u;
    float acc = 0.0, acc2 = 0.0, nv = 0.0;
    uint seg = 0u, lrow = row, woff = pc.w0, boff = 0xFFFFFFFFu, wupo = pc.wup;
    if ((pc.flags & 32u) != 0u) {
        // MoE rows: row = slot * r1 + r; expert e = au[AUX_MOE_ID+slot]; per-expert stride r2 (w0 units), b1 (wup units)
        uint s = row / pc.r1; lrow = row - s * pc.r1;
        uint e = valid ? mids[s] : 0u;
        woff = pc.w0 + e * pc.r2; wupo = pc.wup + e * pc.b1;
    } else if ((pc.flags & 64u) == 0u) {
        seg = row < pc.r1 ? 0u : (row < pc.r2 ? 1u : 2u);
        lrow = row - (seg == 0u ? 0u : (seg == 1u ? pc.r1 : pc.r2));
        woff = seg == 0u ? pc.w0 : (seg == 1u ? pc.w1 : pc.w2);
        boff = seg == 0u ? pc.b0 : (seg == 1u ? pc.b1 : pc.b2);
    }
    if (valid) {
        if ((pc.flags & 64u) != 0u) {
            // MoE down: y[row] (+)= sum_s w_s * W_{e_s}[row] . x[s*K ..], b0 = number of slots, r2 = expert stride
            for (uint s = 0u; s < pc.b0; s++) {
                uint wo = pc.w0 + mids[s] * pc.r2; g_xo = s * (pc.K >> 2u);
                float t = 0.0;
                if ((pc.types & 1u) != 0u) { for (uint it = lane; it < nb * 8u; it += LPR) t += item6k(wo, lrow, nb, it); }
                else { for (uint it = lane; it < nb * 4u; it += LPR) t += item4k(wo, lrow, nb, it); }
                acc += mws[s] * t;
            }
        } else if (((pc.types >> seg) & 1u) != 0u) {
            for (uint it = lane; it < nb * 8u; it += LPR) acc += item6k(woff, lrow, nb, it);
        } else {
            for (uint it = lane; it < nb * 4u; it += LPR) {
                acc += item4k(woff, lrow, nb, it);
#ifdef SWIGLU
                acc2 += item4k(wupo, lrow, nb, it);
#endif
            }
        }
    }
    acc = subgroupClusteredAdd(acc, LPR);
    float scale = ((pc.flags & 4u) != 0u) ? inversesqrt(sq[pc.sq_in] / float(pc.K) + pc.eps) : 1.0;
    acc *= scale;
#ifdef SWIGLU
    acc2 = subgroupClusteredAdd(acc2, LPR) * scale;
    acc = acc / (1.0 + exp(-acc)) * acc2;
#endif
    if (valid && lane == 0u) {
        if (boff != 0xFFFFFFFFu) acc += f32w[boff + lrow];
        uint o = pc.y_off + row;
        nv = ((pc.flags & 1u) != 0u) ? y[o] + acc : acc;
        y[o] = nv;
        if ((pc.flags & 16u) != 0u) xw[o] = nv * f32w[pc.norm_off + row];
    }
    if ((pc.flags & 8u) != 0u) {
        float t2 = subgroupAdd((valid && lane == 0u) ? nv * nv : 0.0);
        if (subgroupElect()) red[gl_SubgroupID] = t2;
        barrier();
        if ((pc.flags & 128u) == 0u) {
            if (gl_LocalInvocationID.x == 0u) { float t = 0.0; for (uint i = 0u; i < gl_NumSubgroups; i++) t += red[i]; atomicAdd(sq[pc.sq_out], t); }
        } else {
            uint lid = gl_LocalInvocationID.x, nwg = gl_NumWorkGroups.x * gl_NumWorkGroups.y, wg = gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x;
            if (lid == 0u) { float t = 0.0; for (uint i = 0u; i < gl_NumSubgroups; i++) t += red[i]; sq3[D3_PART + wg] = t; memoryBarrierBuffer();
                s_last3 = atomicAdd(cnt3[D3_CNT], 1u) == nwg - 1u ? 1u : 0u; }
            barrier();
            if (s_last3 == 1u) {
                memoryBarrierBuffer();
                float a = 0.0; for (uint i = lid; i < nwg; i += gl_WorkGroupSize.x) a += sq3[D3_PART + i];
                a = subgroupAdd(a); if (subgroupElect()) s_red3[gl_SubgroupID] = a;
                barrier();
                if (lid == 0u) { float tot = 0.0; for (uint i = 0u; i < gl_NumSubgroups; i++) tot += s_red3[i]; sq3[pc.sq_out] += tot; cnt3[D3_CNT] = 0u; }
            }
        }
    }
}
