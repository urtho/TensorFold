// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jay Leaton. The DeepSeek-V4.1-Flash family of TensorFold (Apache-2.0): see THIRD_PARTY_NOTICES.md.
// Modified for TensorFold dsv41-cuda: the relayout / unpack checks of patches/0002 families/deepseek_v41/cuda/dense3.cpp
// beside linear.cpp's linear / linear_rot_out checks, for lanes.cu (TF_EXL3_LANES, lanes.py).
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void exl3_lanes_linear_cuda(const at::Tensor&, const at::Tensor&, int64_t, int64_t, const at::Tensor&,
                            const c10::optional<at::Tensor>&, at::Tensor&, const c10::optional<at::Tensor>&,
                            at::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                            const c10::optional<at::Tensor>&, const c10::optional<at::Tensor>&, int64_t);
void exl3_lanes_relayout_cuda(const at::Tensor&, at::Tensor&, int64_t, int64_t, bool);
void exl3_lanes_unpack_cuda(const at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t, int64_t);

static bool width_ok(int64_t K2) { return K2 == 4 || K2 == 6 || K2 == 8 || K2 == 10 || K2 == 12; }

static void check(const at::Tensor& x, at::ScalarType t, const char* name) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == t && x.is_contiguous(), name,
                ": expected a contiguous CUDA tensor of the right dtype");
}

static void check_io(const at::Tensor& x, const char* name) {
    const auto t = x.scalar_type();
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 2 &&
                    (t == at::kHalf || t == at::kBFloat16 || t == at::kFloat),
                name, ": expected a contiguous 2-d fp16, bf16 or fp32 CUDA tensor");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, name, ": must be 16-byte aligned");
}

// linear.cpp linear_any's checks, then the lanes kernel
static void linear_any(const at::Tensor& xh, const at::Tensor& T, int64_t stride_k, int64_t stride_nb,
                       const at::Tensor& svh, const c10::optional<at::Tensor>& bias, at::Tensor y,
                       const c10::optional<at::Tensor>& Z, at::Tensor counters, int64_t K2, int64_t cb, int64_t SK,
                       int64_t WK, int64_t KG, int64_t GN, const c10::optional<at::Tensor>& xo,
                       const c10::optional<at::Tensor>& suho, int64_t rmode) {
    check(xh, at::kHalf, "xh");
    check_io(y, "y");
    check(svh, at::kHalf, "svh");
    check(T, at::kInt, "T");
    check(counters, at::kInt, "counters");
    TORCH_CHECK(width_ok(K2), "lanes: K2 4, 6, 8, 10 or 12");
    const int64_t M = xh.size(0), K = KG > 0 ? KG : xh.size(1), N = y.size(1);
    TORCH_CHECK(xh.dim() == 2 && y.size(0) == M && M >= 1 && M <= 128, "xh and y must have the same 1 to 128 rows");
    TORCH_CHECK(KG <= 0 || (GN > 0 && GN % 128 == 0 && N % GN == 0 && xh.size(1) == KG * (N / GN)),
                "groups: xh must hold N / GN groups of KG inputs");
    TORCH_CHECK(K % 128 == 0 && N % 128 == 0, "K and N must be multiples of 128");
    TORCH_CHECK(svh.numel() == N, "svh must have N elements");
    TORCH_CHECK(T.numel() == K * N * K2 / 64, "T must hold K * N * bits / 32 words");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(T.data_ptr()) % 16 == 0, "T must be 16-byte aligned");
    TORCH_CHECK(stride_k == 32 * K2 && stride_nb == (K / 16) * 32 * K2, "lanes: strip strides");
    TORCH_CHECK(counters.numel() >= 8 * (N / 128), "counters must hold 8 * N / 128 ints");
    if (bias) check(*bias, at::kHalf, "bias");
    if (SK > 1) {
        TORCH_CHECK(Z.has_value(), "Z is needed with more than one split");
        check(*Z, at::kFloat, "Z");
        TORCH_CHECK(Z->numel() >= SK * M * N, "Z too small");
    }
    if (xo) {
        check(*xo, at::kHalf, "xo");
        check(*suho, at::kHalf, "suho");
        TORCH_CHECK(xo->dim() == 2 && xo->size(0) == M && xo->size(1) == N && suho->numel() == N,
                    "xo must be [M, N] and suho [N]");
        TORCH_CHECK(rmode >= 0 && rmode < 9, "rmode: 0..8");
    }
    c10::cuda::CUDAGuard guard(xh.device());
    exl3_lanes_linear_cuda(xh, T, stride_k, stride_nb, svh, bias, y, Z, counters, K2, cb, SK, WK, KG, GN, xo, suho,
                           rmode);
}

void linear(const at::Tensor& xh, const at::Tensor& T, int64_t stride_k, int64_t stride_nb, const at::Tensor& svh,
            const c10::optional<at::Tensor>& bias, at::Tensor y, const c10::optional<at::Tensor>& Z,
            at::Tensor counters, int64_t K2, int64_t cb, int64_t SK, int64_t WK, int64_t KG, int64_t GN) {
    linear_any(xh, T, stride_k, stride_nb, svh, bias, y, Z, counters, K2, cb, SK, WK, KG, GN, c10::nullopt,
               c10::nullopt, 0);
}

void linear_rot_out(const at::Tensor& xh, const at::Tensor& T, int64_t stride_k, int64_t stride_nb,
                    const at::Tensor& svh, const c10::optional<at::Tensor>& bias, at::Tensor y,
                    const c10::optional<at::Tensor>& Z, at::Tensor counters, int64_t K2, int64_t cb, int64_t SK,
                    int64_t WK, int64_t KG, int64_t GN, at::Tensor xo, at::Tensor suho, int64_t rmode) {
    linear_any(xh, T, stride_k, stride_nb, svh, bias, y, Z, counters, K2, cb, SK, WK, KG, GN, xo, suho, rmode);
}

// dst = src relaid (whole strips of K / 16 k steps): to the lanes layout, or back to strips
void relayout(const at::Tensor& src, at::Tensor dst, int64_t K2, int64_t K, bool to_lanes) {
    check(src, at::kInt, "src");
    check(dst, at::kInt, "dst");
    TORCH_CHECK(width_ok(K2) && K % 128 == 0 && src.numel() == dst.numel() && src.data_ptr() != dst.data_ptr(),
                "relayout: K2 4-12, K a multiple of 128, two distinct buffers of one size");
    TORCH_CHECK(src.numel() % ((K / 16) * 32 * K2) == 0, "relayout: whole strips");
    c10::cuda::CUDAGuard guard(src.device());
    exl3_lanes_relayout_cuda(src, dst, K2, K, to_lanes);
}

// W [K, N] fp16 = W_q from lanes words (linear.cpp unpack's values); strip strides
void unpack(const at::Tensor& T, at::Tensor W, int64_t stride_k, int64_t stride_nb, int64_t K2, int64_t cb) {
    check(T, at::kInt, "T");
    check(W, at::kHalf, "W");
    TORCH_CHECK(width_ok(K2), "lanes: K2 4, 6, 8, 10 or 12");
    TORCH_CHECK(W.dim() == 2 && W.size(0) % 128 == 0 && W.size(1) % 128 == 0, "W must be [K, N], multiples of 128");
    TORCH_CHECK(T.numel() == W.numel() * K2 / 64, "T must hold K * N * bits / 32 words");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(T.data_ptr()) % 16 == 0, "T must be 16-byte aligned");
    TORCH_CHECK(stride_k == 32 * K2 && stride_nb == (W.size(0) / 16) * 32 * K2, "lanes: strip strides");
    c10::cuda::CUDAGuard guard(T.device());
    exl3_lanes_unpack_cuda(T, W, stride_k, stride_nb, K2, cb);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linear", &linear);
    m.def("linear_rot_out", &linear_rot_out);
    m.def("relayout", &relayout);
    m.def("unpack", &unpack);
}
