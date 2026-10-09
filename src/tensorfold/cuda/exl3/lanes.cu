// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jay Leaton. The DeepSeek-V4.1-Flash family of TensorFold (Apache-2.0): see THIRD_PARTY_NOTICES.md.
// Modified for TensorFold dsv41-cuda: adapted from patches/0002 families/deepseek_v41/cuda/dense3.cu (DENSE_V3: the
// lanes layout, to_lanes_kernel / to_strips_kernel, take_bits, pair_frags, load_group, the k loop, lanes_unpack_kernel)
// onto our linear.cu linear_kernel (column groups KG / GN with the XS row stride, bias, the XO rotated-output epilogue,
// every codebook, plain <<<>>> launches without PDL), loaded by lanes.py under TF_EXL3_LANES.
//
// EXL3 dense linear over the "lanes" layout: linear.cu's linear_kernel with only the k loop's loads changed, so every
// output has linear_kernel's bits.
//
// Why. linear_kernel's k step reads its 8 tiles as 8 * lane_words 4-byte __ldg a lane (24 at 5 bits; the next step's
// one step ahead): lane L needs the 3 words around its 16-bit windows, so each word is fetched by ~2.4 lanes, and the
// step's A fragment is loaded at the top of the step that uses it.
//
// The lanes layout (a bit permutation of each 128-column strip [K/16, 8, 4 K2], built at load by to_lanes_kernel;
// to_strips_kernel is its inverse). Tile (kt, j) is a ring of 128 K2 stream bits (bit b = bit 31 - b % 32 of word
// b / 32, as decode.cuh reads them); lane L decodes values 8L..8L+7, whose windows end in the lane's OWN bits
// [4 K2 L, 4 K2 (L + 1)) and start up to 16 bits before them (in lane L - 1's own bits; lane 0 wraps to lane 31,
// decode.cuh lane_start's ring). For a group of 2 k steps, lane L's own bits of the group's 16 tiles, in (step, tile)
// order, form its 2 K2 words; their 16-byte chunk c sits at word (c * 32 + L) * 4 of the group's block. A warp's chunk
// c is one 512-byte coalesced load and every byte of the strip is read once. A lane takes the 16 prefix bits from lane
// L - 1 by a shuffle (two tiles a shuffle). The next group is requested while this one decodes, and each step's A
// fragment one step early.
//
// Exactness. pair_frags' 8 states per lane and tile are decode.cuh lane_states' (tests/test_exl3_lanes_emu.py checks
// every lane, tile, step and width against an emulation of lane_states); decode2<CB> is decode.cuh's; each acc[j][h]
// takes one mma16816 per k step in ascending kt (only distinct accumulators j, j + 1 are interleaved); the warp's K
// range, the warp-order sums, Z partials, the last arriver and finish / store4 / store_rot are linear.cu's statements,
// copied verbatim. Widths: even K2 4..12 (take_bits needs bo + 4 K2 <= 64; K2 2's windows reach two lanes back; odd
// K2 has no whole-bit sh). Needs per_warp even (whole 2-step groups, and groups on even kt). No PDL: jayleaton saw
// DENSE_V3 + PDL not bit-exact.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>

#include "decode.cuh"
#include "rot128.cuh"

using namespace tf_exl3;

namespace {

enum DType : int { F16 = 0, BF16 = 1, F32 = 2 };

// -- linear.cu's helpers, verbatim (store4, finish, store_rot) ---------------------------------------------------
__device__ __forceinline__ void store4(void* p, int dtype, size_t i, const float (&v)[4]) {
    if (dtype == F32) {
        *reinterpret_cast<float4*>(static_cast<float*>(p) + i) = make_float4(v[0], v[1], v[2], v[3]);
    } else if (dtype == BF16) {
        __nv_bfloat162 a = __floats2bfloat162_rn(v[0], v[1]), b = __floats2bfloat162_rn(v[2], v[3]);
        uint2 u;
        u.x = *reinterpret_cast<uint32_t*>(&a);
        u.y = *reinterpret_cast<uint32_t*>(&b);
        *reinterpret_cast<uint2*>(static_cast<__nv_bfloat16*>(p) + i) = u;
    } else {
        half2 a = __floats2half2_rn(v[0], v[1]), b = __floats2half2_rn(v[2], v[3]);
        uint2 u;
        u.x = *reinterpret_cast<uint32_t*>(&a);
        u.y = *reinterpret_cast<uint32_t*>(&b);
        *reinterpret_cast<uint2*>(static_cast<half*>(p) + i) = u;
    }
}

// The finished outputs of one row's 128 columns from their fp32 sums (4 a lane): H / sqrt(128), * svh, + bias.
__device__ __forceinline__ void finish(float (&v)[4], int lane, const half* svh, const half* bias, int col) {
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[j] = v[j] * HAD_SCALE * __half2float(__ldg(svh + col + j));
        if (bias) v[j] += __half2float(__ldg(bias + col + j));
    }
}

__device__ __forceinline__ void store_rot(const float (&y)[4], int y_dtype, const half* SUHO, half* XO, size_t at,
                                          int col, int lane, int rmode) {
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
        v[j] = y_dtype == BF16 ? __bfloat162float(__float2bfloat16_rn(y[j]))
             : y_dtype == F16 ? __half2float(__float2half_rn(y[j])) : y[j];
    tf_rot::rot128_store(v, SUHO + col, XO + at, lane, rmode);
}

// -- the lanes layout's loads (dense3.cu) ------------------------------------------------------------------------
constexpr int G = 2;                                          // k steps a load group
template <int K2>
__host__ __device__ constexpr int gwords() { return G * K2; }   // a lane's words of a group (whole 16-byte chunks)

// Bits [p, p + n) (MSB first) of words R[p / 32], R[p / 32 + 1] as the low n bits of a 64-bit value (p % 32 + n <= 64
// for the own bits of K2 4..12; indices are compile-time constants after unrolling).
template <int GW>
__device__ __forceinline__ uint64_t take_bits(const uint32_t (&R)[GW], int p, int n) {
    const int wi = p >> 5, bo = p & 31;
    const uint64_t x = ((uint64_t)R[wi] << 32) | (wi + 1 < GW ? (uint64_t)R[wi + 1] : 0ull);
    return n == 64 ? x : (x << bo) >> (64 - n);
}

// Lane `lane`'s B fragments of tiles j and j + 1 of step g of its group words R: the own bits of each, the 16 prefix
// bits from lane - 1 (one shuffle for both tiles), the 8 states of decode.cuh's lane_states, decode2.
template <int K2, int CB>
__device__ __forceinline__ void pair_frags(const uint32_t (&R)[gwords<K2>()], int g, int j, int lane,
                                           uint32_t (&b)[2][2][2]) {
    constexpr int n = 4 * K2;
    const uint64_t o0 = take_bits(R, (g * 8 + j) * n, n), o1 = take_bits(R, (g * 8 + j + 1) * n, n);
    const uint32_t tails = (uint32_t)(o0 & 0xffffu) | ((uint32_t)(o1 & 0xffffu) << 16);
    const uint32_t pre = __shfl_sync(0xffffffffu, tails, (lane + 31) & 31);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const uint64_t o = h ? o1 : o0;
        const uint64_t p = h ? (pre >> 16) : (pre & 0xffffu);
        uint32_t s[8];
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const int sh = (7 - v) * K2 / 2;                  // the window of value 8 lane + v: bits [sh, sh + 16)
            uint64_t w = o >> sh;
            if (n - sh < 16) w |= p << (n - sh);
            s[v] = (uint32_t)(w & 0xffffu);
        }
        b[h][0][0] = decode2<CB>(s[0], s[1]);
        b[h][0][1] = decode2<CB>(s[2], s[3]);
        b[h][1][0] = decode2<CB>(s[4], s[5]);
        b[h][1][1] = decode2<CB>(s[6], s[7]);
    }
}

// a lane's chunks of one group: GW / 4 coalesced 16-byte loads (warp: 512 contiguous bytes each)
template <int GW>
__device__ __forceinline__ void load_group(const uint32_t* lane_base, uint32_t (&R)[GW]) {
#pragma unroll
    for (int c = 0; c < GW / 4; ++c) {
        const uint4 u = __ldg(reinterpret_cast<const uint4*>(lane_base + c * 128));
        R[4 * c] = u.x; R[4 * c + 1] = u.y; R[4 * c + 2] = u.z; R[4 * c + 3] = u.w;
    }
}

// linear_kernel's A fragment statements of k step kt (its non-ROT branch)
__device__ __forceinline__ void load_a(const half* x0, const half* x1, int kt, int t, uint32_t (&a)[4]) {
    a[0] = __ldg(reinterpret_cast<const uint32_t*>(x0 + kt * 16 + 2 * t));
    a[1] = __ldg(reinterpret_cast<const uint32_t*>(x1 + kt * 16 + 2 * t));
    a[2] = __ldg(reinterpret_cast<const uint32_t*>(x0 + kt * 16 + 2 * t + 8));
    a[3] = __ldg(reinterpret_cast<const uint32_t*>(x1 + kt * 16 + 2 * t + 8));
}

// linear_kernel<K2, CB, WK, false> over lanes words (T: the strips' pointer and strides; the same K ranges); the
// statements after the k loop are linear.cu's verbatim. Needs per_warp % 2 == 0 (lanes.py lanes_ok, the launcher).
// Blocks an SM: 3 at 4 warps (<= 170 registers; ptxas -v: 151-166), 1 at 8 warps (ptxas's own choice there capped
// K2 4 / 6 at 128 registers and spilled at K2 4; 181-243 now, no spills).
template <int K2, int CB, int WK>
__global__ void __launch_bounds__(WK * 32, WK == 8 ? 1 : 3) lanes_linear_kernel(
    const half* __restrict__ xh, const uint32_t* __restrict__ T, long long stride_k, long long stride_nb,
    const half* __restrict__ svh, const half* __restrict__ bias, void* __restrict__ y, int y_dtype,
    float* __restrict__ Z, int* __restrict__ counters, int M, int K, int N, int SK, int XS, int GN,
    half* __restrict__ XO, const half* __restrict__ SUHO, int rmode) {
    constexpr int GW = gwords<K2>();
    extern __shared__ __align__(16) float red[];              // WK * RH * 128 floats
    __shared__ int last;
    const int RH = min(M, 8);                                 // rows of red a warp

    const int nb = blockIdx.x, split = blockIdx.y, NB = gridDim.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int per_warp = (K >> 4) / SK / WK;
    const int kt0 = split * (per_warp * WK) + warp * per_warp;
    const int col0 = nb * 128;
    const int groups = per_warp / G;
    const uint32_t* base = T + nb * stride_nb + (size_t)kt0 * stride_k + 4 * lane;
    const size_t gstride = (size_t)G * stride_k;

    for (int m0 = 0, pass = 0; m0 < M; m0 += 16, ++pass) {
        const int R = min(16, M - m0);

        float acc[8][2][4];
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

        // the walk's rows for the mma, clamped so rows past the pass read inside the buffer (their outputs are dropped)
        const int r0 = m0 + (g < R ? g : R - 1), r1 = m0 + (g + 8 < R ? g + 8 : R - 1);
        // column group col0 / GN reads its own K inputs of a row (XS apart): GN = N, XS = K is one ordinary layer
        const half* xg = xh + (size_t)(col0 / GN) * K;
        const half* x0 = xg + (size_t)r0 * XS;
        const half* x1 = xg + (size_t)r1 * XS;
        uint32_t cur[GW], nxt[GW];
        if (groups > 0) load_group(base, cur);
        uint32_t a[4], an[4];
        if (per_warp > 0) load_a(x0, x1, kt0, t, a);
#pragma unroll 1
        for (int gi = 0; gi < groups; ++gi) {
            if (gi + 1 < groups) load_group(base + (size_t)(gi + 1) * gstride, nxt);
#pragma unroll
            for (int st = 0; st < G; ++st) {
                const int i = gi * G + st;
                if (i + 1 < per_warp) load_a(x0, x1, kt0 + i + 1, t, an);   // the next step's input, one step early
#pragma unroll
                for (int j = 0; j < 8; j += 2) {
                    uint32_t b[2][2][2];
                    pair_frags<K2, CB>(cur, st, j, lane, b);
                    mma16816(acc[j][0], a, b[0][0]);
                    mma16816(acc[j][1], a, b[0][1]);
                    mma16816(acc[j + 1][0], a, b[1][0]);
                    mma16816(acc[j + 1][1], a, b[1][1]);
                }
#pragma unroll
                for (int c = 0; c < 4; ++c) a[c] = an[c];
            }
#pragma unroll
            for (int c = 0; c < GW; ++c) cur[c] = nxt[c];
        }

        // -- linear.cu's statements from here on, verbatim --
        // the warps' sums, added in warp order, rows 0-7 of the pass and then rows 8-15
        for (int rlo = 0; rlo < R; rlo += 8) {
            const int rn = min(R - rlo, 8);
            __syncthreads();                         // red is reused by every half and pass
            if (g < RH) {
#pragma unroll
                for (int i = 0; i < 8; ++i)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int col = i * 16 + h * 8 + 2 * t;
                        *reinterpret_cast<float2*>(red + (warp * RH + g) * 128 + col) =
                            rlo ? make_float2(acc[i][h][2], acc[i][h][3]) : make_float2(acc[i][h][0], acc[i][h][1]);
                    }
            }
            __syncthreads();

            if (SK == 1) {
                for (int r = warp; r < rn; r += WK) {
                    float v[4];
                    const float4 u = *reinterpret_cast<const float4*>(red + r * 128 + 4 * lane);
                    v[0] = u.x; v[1] = u.y; v[2] = u.z; v[3] = u.w;
#pragma unroll
                    for (int w = 1; w < WK; ++w) {
                        const float4 q = *reinterpret_cast<const float4*>(red + (w * RH + r) * 128 + 4 * lane);
                        v[0] += q.x; v[1] += q.y; v[2] += q.z; v[3] += q.w;
                    }
                    finish(v, lane, svh, bias, col0 + 4 * lane);
                    store4(y, y_dtype, (size_t)(m0 + rlo + r) * N + col0 + 4 * lane, v);
                    if (XO != nullptr)                          // (a kernel argument; the warp's loop: whole warps)
                        store_rot(v, y_dtype, SUHO, XO, (size_t)(m0 + rlo + r) * N + col0 + 4 * lane, col0 + 4 * lane,
                                  lane, rmode);
                }
            } else {
                for (int idx = threadIdx.x; idx < rn * 32; idx += WK * 32) {
                    const int r = idx >> 5, c = 4 * (idx & 31);
                    float4 s = *reinterpret_cast<const float4*>(red + r * 128 + c);
#pragma unroll
                    for (int w = 1; w < WK; ++w) {
                        const float4 q = *reinterpret_cast<const float4*>(red + (w * RH + r) * 128 + c);
                        s.x += q.x; s.y += q.y; s.z += q.z; s.w += q.w;
                    }
                    *reinterpret_cast<float4*>(Z + ((size_t)split * M + m0 + rlo + r) * N + col0 + c) = s;
                }
            }
        }
        if (SK > 1) {
            __threadfence();
            __syncthreads();
            if (threadIdx.x == 0) last = atomicAdd(counters + pass * NB + nb, 1) == SK - 1;
            __syncthreads();
            if (last) {
                __threadfence();
                for (int r = warp; r < R; r += WK) {
                    const size_t at = ((size_t)m0 + r) * N + col0 + 4 * lane;
                    float4 s = __ldcg(reinterpret_cast<const float4*>(Z + at));
                    for (int q = 1; q < SK; ++q) {
                        const float4 u = __ldcg(reinterpret_cast<const float4*>(Z + (size_t)q * M * N + at));
                        s.x += u.x; s.y += u.y; s.z += u.z; s.w += u.w;
                    }
                    float v[4] = {s.x, s.y, s.z, s.w};
                    finish(v, lane, svh, bias, col0 + 4 * lane);
                    store4(y, y_dtype, (size_t)(m0 + r) * N + col0 + 4 * lane, v);
                    if (XO != nullptr)
                        store_rot(v, y_dtype, SUHO, XO, (size_t)(m0 + r) * N + col0 + 4 * lane, col0 + 4 * lane, lane,
                                  rmode);
                }
                if (threadIdx.x == 0) counters[pass * NB + nb] = 0;   // every program of the block has arrived
            }
        }
        __syncthreads();                             // red is reused in the next pass
    }
}

// -- the load-time transform and its inverse (dense3.cu; one thread a destination word, not on the decode path) ----
// A strip has KT k steps of 32 K2 words; in the lanes layout, group gi = kt / 2 holds lane L's word w (0 .. 2 K2 - 1)
// at gi * 2 * 32 K2 + ((w / 4) * 32 + L) * 4 + w % 4; bit lb of lane L's group stream is tile (step, j) =
// divmod(lb / (4 K2), 8), own bit ob = lb % (4 K2), i.e. stream bit 4 K2 L + ob of that tile.
__device__ __forceinline__ uint32_t bit_of(const uint32_t* tile, int sb) { return (tile[sb >> 5] >> (31 - (sb & 31))) & 1u; }

__global__ void to_lanes_kernel(const uint32_t* __restrict__ src, uint32_t* __restrict__ dst, long long total, int K2,
                                int KT) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int n = 4 * K2, TW = 4 * K2;
    const long long strip_words = (long long)KT * 32 * K2, gw = (long long)G * 32 * K2;
    const long long strip = i / strip_words, r = i % strip_words, gi = r / gw;
    const int qd = (int)(r % gw), c = qd / 128, L = (qd % 128) / 4, w = 4 * c + qd % 4;
    const uint32_t* s0 = src + strip * strip_words;
    uint32_t out = 0;
    for (int b = 0; b < 32; ++b) {
        const int lb = 32 * w + b, tt = lb / n, ob = lb % n;
        const long long kt = gi * G + tt / 8;
        out |= bit_of(s0 + kt * 32 * K2 + (tt % 8) * TW, n * L + ob) << (31 - b);
    }
    dst[i] = out;
}

__global__ void to_strips_kernel(const uint32_t* __restrict__ src, uint32_t* __restrict__ dst, long long total, int K2,
                                 int KT) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int n = 4 * K2, TW = 4 * K2;
    const long long strip_words = (long long)KT * 32 * K2;
    const long long strip = i / strip_words, r = i % strip_words, kt = r / (32 * K2);
    const int tw = (int)(r % (32 * K2)), j = tw / TW, wd = tw % TW;
    const long long gi = kt / G;
    const int tt = (int)(kt % G) * 8 + j;
    const uint32_t* s0 = src + strip * strip_words + gi * (long long)G * 32 * K2;
    uint32_t out = 0;
    for (int b = 0; b < 32; ++b) {
        const int sb = 32 * wd + b, L = sb / n, lb = tt * n + sb % n, w = lb >> 5;
        out |= ((s0[((w >> 2) * 32 + L) * 4 + (w & 3)] >> (31 - (lb & 31))) & 1u) << (31 - b);
    }
    dst[i] = out;
}

// W_q [K, N] fp16 from the lanes layout (the prompt GEMM's decode-once W; linear.cu unpack_kernel's values): one warp
// a (16-column tile, k step), the main kernel's fragment code.
template <int K2, int CB>
__global__ void __launch_bounds__(32) lanes_unpack_kernel(const uint32_t* __restrict__ T, half* __restrict__ W, int N,
                                                          int64_t stride_k, int64_t stride_nb) {
    constexpr int GW = gwords<K2>();
    const int kt = blockIdx.y, nt = blockIdx.x, lane = threadIdx.x, j = nt & 6;
    uint32_t R[GW];
    load_group(T + (nt >> 3) * stride_nb + (size_t)(kt / G) * G * stride_k + 4 * lane, R);
    uint32_t b[2][2][2];
    pair_frags<K2, CB>(R, kt % G, j, lane, b);
    const int h = nt & 1;
#pragma unroll
    for (int v = 0; v < 8; ++v) {
        const uint32_t x = b[h][v >> 2][(v >> 1) & 1];
        const unsigned short u = (v & 1) ? (unsigned short)(x >> 16) : (unsigned short)(x & 0xffffu);
        W[(size_t)(kt * 16 + value_row(lane, v)) * N + nt * 16 + value_col(lane, v)] = __ushort_as_half(u);
    }
}

int dtype_of(const at::Tensor& t) {
    return t.scalar_type() == at::kFloat ? F32 : t.scalar_type() == at::kBFloat16 ? BF16 : F16;
}

}  // namespace

#define TF_LANES_WIDTHS(X, CB) X(4, CB) X(6, CB) X(8, CB) X(10, CB) X(12, CB)
#define TF_LANES_ALL(X) TF_LANES_WIDTHS(X, 0) TF_LANES_WIDTHS(X, 1) TF_LANES_WIDTHS(X, 2)

void exl3_lanes_linear_cuda(const at::Tensor& xh, const at::Tensor& T, int64_t stride_k, int64_t stride_nb,
                            const at::Tensor& svh, const c10::optional<at::Tensor>& bias, at::Tensor& y,
                            const c10::optional<at::Tensor>& Z, at::Tensor& counters, int64_t K2, int64_t cb,
                            int64_t SK, int64_t WK, int64_t KG, int64_t GN, const c10::optional<at::Tensor>& xo,
                            const c10::optional<at::Tensor>& suho, int64_t rmode) {
    const int M = (int)xh.size(0), XS = (int)xh.size(1), N = (int)y.size(1);
    const int K = KG > 0 ? (int)KG : XS, Gn = GN > 0 ? (int)GN : N;
    TORCH_CHECK(WK == 4 || WK == 8, "lanes: WK must be 4 or 8");
    TORCH_CHECK((K / 16) % (SK * WK) == 0, "K / 16 must split evenly over SK * WK warps");
    TORCH_CHECK(((K / 16) / (SK * WK)) % G == 0, "lanes: a warp's k steps must be whole 2-step load groups");
    dim3 grid((unsigned)(N / 128), (unsigned)SK);
    auto stream = at::cuda::getCurrentCUDAStream();
    const half* bptr = bias ? reinterpret_cast<const half*>(bias->data_ptr()) : nullptr;
    float* zptr = Z ? Z->data_ptr<float>() : nullptr;
    TORCH_CHECK(SK == 1 || zptr, "Z is needed with more than one split");
#define TF_LAUNCH(K2_, CB_)                                                                                        \
    if (K2 == K2_ && cb == CB_) {                                                                               \
        auto kernel = WK == 4 ? lanes_linear_kernel<K2_, CB_, 4> : lanes_linear_kernel<K2_, CB_, 8>;            \
        const int smem = (int)(WK * std::min(M, 8) * 128 * sizeof(float));                                      \
        if (smem > 48 * 1024) cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); \
        kernel<<<grid, (unsigned)(WK * 32), smem, stream>>>(                                                    \
            reinterpret_cast<const half*>(xh.data_ptr()), reinterpret_cast<const uint32_t*>(T.data_ptr()),      \
            stride_k, stride_nb, reinterpret_cast<const half*>(svh.data_ptr()), bptr, y.data_ptr(), dtype_of(y),\
            zptr, counters.data_ptr<int>(), M, K, N, (int)SK, XS, Gn,                                           \
            xo ? reinterpret_cast<half*>(xo->data_ptr()) : nullptr,                                             \
            xo ? reinterpret_cast<const half*>(suho->data_ptr()) : nullptr, (int)rmode);                        \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                         \
        return;                                                                                                 \
    }
    TF_LANES_ALL(TF_LAUNCH)
#undef TF_LAUNCH
    TORCH_CHECK(false, "lanes: unsupported EXL3 width/codebook: K2=", K2, " codebook=", cb);
}

// src -> dst (int32 words of whole strips [N/128, K/16, 8, 4 K2]); to lanes, or back to strips
void exl3_lanes_relayout_cuda(const at::Tensor& src, at::Tensor& dst, int64_t K2, int64_t K, bool to_lanes) {
    const long long total = src.numel();
    const unsigned blocks = (unsigned)((total + 255) / 256);
    auto stream = at::cuda::getCurrentCUDAStream();
    const auto* s = reinterpret_cast<const uint32_t*>(src.data_ptr());
    auto* d = reinterpret_cast<uint32_t*>(dst.data_ptr());
    if (to_lanes)
        to_lanes_kernel<<<blocks, 256, 0, stream>>>(s, d, total, (int)K2, (int)(K / 16));
    else
        to_strips_kernel<<<blocks, 256, 0, stream>>>(s, d, total, (int)K2, (int)(K / 16));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exl3_lanes_unpack_cuda(const at::Tensor& T, at::Tensor& W, int64_t stride_k, int64_t stride_nb, int64_t K2,
                            int64_t cb) {
    const int K = (int)W.size(0), N = (int)W.size(1);
    dim3 grid((unsigned)(N / 16), (unsigned)(K / 16));
    auto stream = at::cuda::getCurrentCUDAStream();
#define TF_LAUNCH(K2_, CB_)                                                                                         \
    if (K2 == K2_ && cb == CB_) {                                                                                \
        lanes_unpack_kernel<K2_, CB_><<<grid, 32, 0, stream>>>(reinterpret_cast<const uint32_t*>(T.data_ptr()),   \
                                                              reinterpret_cast<half*>(W.data_ptr()), N, stride_k, \
                                                              stride_nb);                                         \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                          \
        return;                                                                                                  \
    }
    TF_LANES_ALL(TF_LAUNCH)
#undef TF_LAUNCH
    TORCH_CHECK(false, "lanes: unsupported EXL3 width/codebook: K2=", K2, " codebook=", cb);
}
