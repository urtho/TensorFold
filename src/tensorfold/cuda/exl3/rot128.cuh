// rot_in_kernel's input rotation (x * suh, H / sqrt(128), linear.cu) for producers that already hold the values
// (TF_DSV41_ROT_FUSE: the attention merge writes wo_a's rotated rows, wo_a's epilogue wo_b's): every operation an
// explicit intrinsic, so no compiler contracts or reorders it and the bits equal rot_in's wherever it runs. rot_in
// itself is plain C (v *= s, then the butterflies), so nvcc chooses how its first butterfly contracts; ``mode`` says
// which form (linear.py rot_mode() finds it against rot_in on this build): a = a_form(mode / 3), b = b_form(mode % 3)
// of a = x0 s0 + x1 s1, b = x0 s0 - x1 s1 (the same for x2, x3): 0: fma(x0, s0, +-x1 s1), 1: fma(+-x1, s1, x0 s0),
// 2: both products rounded, then added (sm_121 CUDA 13.1 -O3 SASS: FMUL x1 s1, FFMA x0 s0 +- p, mode 0). The later
// stages are adds only (fwht128's order), then the multiply by 1 / sqrt(128). (Explicit-fma form after peer
// bertholomus/TensorFold bd0024d's _rot128, Apache-2.0; no code copied.)

#pragma once

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace tf_rot {

constexpr float ROT_SCALE = 0.08838834764831845f;   // 1 / sqrt(128): tf_exl3::HAD_SCALE

__device__ __forceinline__ void pair(float x0, float s0, float x1, float s1, int mode, float& a, float& b) {
    const float p0 = __fmul_rn(x0, s0), p1 = __fmul_rn(x1, s1);
    const int fa = mode / 3, fb = mode % 3;          // (uniform: a kernel argument)
    a = fa == 0 ? __fmaf_rn(x0, s0, p1) : fa == 1 ? __fmaf_rn(x1, s1, p0) : __fadd_rn(p0, p1);
    b = fb == 0 ? __fmaf_rn(x0, s0, -p1) : fb == 1 ? __fmaf_rn(-x1, s1, p0) : __fsub_rn(p0, p1);
}

// v: a lane's 4 values of a warp's 128 (lane L: 4L .. 4L + 3), s their suh; every lane of the warp calls it
__device__ __forceinline__ void rot128(float (&v)[4], const float (&s)[4], int lane, int mode) {
    float a, b, c, d;
    pair(v[0], s[0], v[1], s[1], mode, a, b);
    pair(v[2], s[2], v[3], s[3], mode, c, d);
    v[0] = __fadd_rn(a, c);
    v[1] = __fadd_rn(b, d);
    v[2] = __fsub_rn(a, c);
    v[3] = __fsub_rn(b, d);
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float o = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = (lane & m) ? __fsub_rn(o, v[j]) : __fadd_rn(v[j], o);
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = __fmul_rn(v[j], ROT_SCALE);
}

// rot128 of 4 values v as rot_in reads them (already rounded to the producer's stored dtype), their suh at su, the
// fp16 result to xh (rot_in's store; 8-byte aligned)
__device__ __forceinline__ void rot128_store(float (&v)[4], const half* su, half* xh, int lane, int mode) {
    float s[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) s[j] = __half2float(su[j]);
    rot128(v, s, lane, mode);
    const half2 lo = __floats2half2_rn(v[0], v[1]), hi = __floats2half2_rn(v[2], v[3]);
    uint2 u;
    u.x = *reinterpret_cast<const uint32_t*>(&lo);
    u.y = *reinterpret_cast<const uint32_t*>(&hi);
    *reinterpret_cast<uint2*>(xh) = u;
}

// rot128_store of 4 fp32 values their producer stores as bf16
__device__ __forceinline__ void rot128_bf16_store(const float (&o)[4], const half* su, half* xh, int lane, int mode) {
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = __bfloat162float(__float2bfloat16_rn(o[j]));
    rot128_store(v, su, xh, lane, mode);
}

}  // namespace tf_rot
