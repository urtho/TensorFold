#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void exl3x_grouped_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                        const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&,
                        int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                        int64_t, int64_t);
void exl3x_dequant_cuda(const at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t, int64_t);
void exl3x_group_cuda(const at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t, int64_t);
void exl3x_rot_in_cuda(const at::Tensor&, int64_t, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                       at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t, int64_t);
void exl3x_gateup_epilogue_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                                const at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                                double, int64_t, const void*, const void*, int64_t, int64_t, int64_t);
void exl3x_down_epilogue_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, int64_t, int64_t,
                              int64_t, int64_t, int64_t, int64_t);
void exl3x_combine_cuda(const at::Tensor&, const at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t);
void exl3x_down_combine_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, const at::Tensor&,
                             at::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, const float*, int64_t,
                             const void*, int64_t, int64_t);

static void check(const at::Tensor& x, at::ScalarType t, const char* name) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == t && x.is_contiguous(), name,
                ": expected a contiguous CUDA tensor of the right dtype");
}

// L2 discards only on 128-byte aligned scratch (rows are whole lines: K, N, D, I multiples of 128); else none.
static bool line_aligned(const at::Tensor& x) { return reinterpret_cast<uintptr_t>(x.data_ptr()) % 128 == 0; }

void grouped(const at::Tensor& X0, const at::Tensor& X1, const at::Tensor& TP0, const at::Tensor& TP1,
             const at::Tensor& B0, const at::Tensor& B1, const at::Tensor& uids, const at::Tensor& ucount,
             const at::Tensor& members, at::Tensor Z, int64_t mats, int64_t K, int64_t N, int64_t P, int64_t SK,
             int64_t slots, int64_t cb, int64_t nt, int64_t warps, int64_t pf, int64_t lo, int64_t hi) {
    check(X0, at::kHalf, "X0");
    check(X1, at::kHalf, "X1");
    check(TP0, at::kLong, "TP0");
    check(TP1, at::kLong, "TP1");
    check(B0, at::kInt, "B0");
    check(B1, at::kInt, "B1");
    check(uids, at::kInt, "uids");
    check(ucount, at::kInt, "ucount");
    check(members, at::kInt, "members");
    check(Z, at::kFloat, "Z");
    TORCH_CHECK(Z.numel() >= mats * SK * P * N, "Z too small");
    TORCH_CHECK(X0.numel() >= P * K && X1.numel() >= P * K, "X too small");
    c10::cuda::CUDAGuard guard(X0.device());
    exl3x_grouped_cuda(X0, X1, TP0, TP1, B0, B1, uids, ucount, members, Z, mats, K, N, P, SK, slots, cb, nt, warps,
                       pf, lo, hi);
}

void dequant(const at::Tensor& T, at::Tensor out, int64_t k2, int64_t cb) {
    TORCH_CHECK(T.is_cuda() && T.scalar_type() == at::kShort && T.is_contiguous() && T.dim() == 3,
                "T: int16 [K/16, N/16, 8 * k2]");
    TORCH_CHECK(T.size(2) == 8 * k2, "trellis last dim must be 8 * k2");
    check(out, at::kHalf, "out");
    const int64_t K = T.size(0) * 16, N = T.size(1) * 16;
    TORCH_CHECK(out.numel() == K * N, "out must be [K, N]");
    c10::cuda::CUDAGuard guard(T.device());
    exl3x_dequant_cuda(T, out, K, N, k2, cb);
}

void group(const at::Tensor& pick, at::Tensor uids, at::Tensor ucount, at::Tensor members, int64_t R, int64_t slots,
           int64_t E, int64_t par) {
    check(pick, at::kInt, "pick");
    check(uids, at::kInt, "uids");
    check(ucount, at::kInt, "ucount");
    check(members, at::kInt, "members");
    TORCH_CHECK(pick.numel() >= R * slots, "pick too small");
    TORCH_CHECK(uids.numel() >= std::min<int64_t>(R * slots, E), "uids too small");
    TORCH_CHECK(members.size(0) >= uids.numel() && members.size(1) >= 1, "members too small");
    c10::cuda::CUDAGuard guard(pick.device());
    exl3x_group_cuda(pick, uids, ucount, members, R, slots, E, par);
}

void rot_in(const at::Tensor& x, int64_t x_stride, const at::Tensor& pick, const at::Tensor& suh0,
            const at::Tensor& suh1, at::Tensor out0, at::Tensor out1, int64_t rows, int64_t K, int64_t slots,
            int64_t E) {
    TORCH_CHECK(x.is_cuda() && (x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf), "x: bf16/fp16 CUDA");
    check(pick, at::kInt, "pick");
    check(suh0, at::kHalf, "suh0");
    check(suh1, at::kHalf, "suh1");
    check(out0, at::kHalf, "out0");
    check(out1, at::kHalf, "out1");
    TORCH_CHECK(K % 128 == 0, "K must be a multiple of 128");
    c10::cuda::CUDAGuard guard(x.device());
    exl3x_rot_in_cuda(x, x_stride, pick, suh0, suh1, out0, out1, rows, K, slots, E);
}

// discard (TF_DSV41_L2_DISCARD=moe): xg / xu [P, K] given, the down launch's Z ends at float zlive.
void gateup_epilogue(const at::Tensor& Z, const at::Tensor& pick, const at::Tensor& svh_g, const at::Tensor& svh_u,
                     const at::Tensor& suh_d, at::Tensor xd, int64_t rows, int64_t P, int64_t N, int64_t SK,
                     int64_t slots, int64_t E, double limit, int64_t act_mode, c10::optional<at::Tensor> xg,
                     c10::optional<at::Tensor> xu, int64_t zlive, int64_t discard) {
    check(Z, at::kFloat, "Z");
    check(pick, at::kInt, "pick");
    check(svh_g, at::kHalf, "svh_g");
    check(svh_u, at::kHalf, "svh_u");
    check(suh_d, at::kHalf, "suh_d");
    check(xd, at::kHalf, "xd");
    TORCH_CHECK(N % 128 == 0, "N must be a multiple of 128");
    const void *g = nullptr, *u = nullptr;
    int64_t K = 0;
    if (discard) {
        TORCH_CHECK(xg.has_value() && xu.has_value(), "discard needs xg and xu");
        check(*xg, at::kHalf, "xg");
        check(*xu, at::kHalf, "xu");
        K = xg->size(1);
        TORCH_CHECK(xg->dim() == 2 && xu->sizes() == xg->sizes() && xg->size(0) >= P && K % 128 == 0,
                    "xg / xu: [>= P, K], K a multiple of 128");
        TORCH_CHECK(zlive % 32 == 0, "zlive: a whole number of lines");
        discard = line_aligned(Z) && line_aligned(*xg) && line_aligned(*xu);
        g = xg->data_ptr();
        u = xu->data_ptr();
    }
    c10::cuda::CUDAGuard guard(Z.device());
    exl3x_gateup_epilogue_cuda(Z, pick, svh_g, svh_u, suh_d, xd, rows, P, N, SK, slots, E, limit, act_mode, g, u, K,
                               zlive, discard);
}

void down_epilogue(const at::Tensor& Z, const at::Tensor& pick, const at::Tensor& svh_d, at::Tensor y, int64_t rows,
                   int64_t P, int64_t D, int64_t SK, int64_t slots, int64_t E) {
    check(Z, at::kFloat, "Z");
    check(pick, at::kInt, "pick");
    check(svh_d, at::kHalf, "svh_d");
    check(y, at::kFloat, "y");
    TORCH_CHECK(D % 128 == 0, "D must be a multiple of 128");
    c10::cuda::CUDAGuard guard(Z.device());
    exl3x_down_epilogue_cuda(Z, pick, svh_d, y, rows, P, D, SK, slots, E);
}

void combine(const at::Tensor& y, const at::Tensor& wts, at::Tensor out, int64_t rows, int64_t D, int64_t slots) {
    check(y, at::kFloat, "y");
    check(wts, at::kFloat, "wts");
    check(out, at::kFloat, "out");
    c10::cuda::CUDAGuard guard(y.device());
    exl3x_combine_cuda(y, wts, out, rows, D, slots);
}

// res: fp32 [rows, D] added to out (TF_DSV41_RES_FOLD); store_y 0: y left unwritten; discard (needs xd [P, I]).
void down_combine(const at::Tensor& Z, const at::Tensor& pick, const at::Tensor& svh_d, at::Tensor y,
                  const at::Tensor& wts, at::Tensor out, int64_t rows, int64_t P, int64_t D, int64_t SK, int64_t slots,
                  int64_t E, c10::optional<at::Tensor> res, int64_t store_y, c10::optional<at::Tensor> xd,
                  int64_t discard) {
    check(Z, at::kFloat, "Z");
    check(pick, at::kInt, "pick");
    check(svh_d, at::kHalf, "svh_d");
    check(y, at::kFloat, "y");
    check(wts, at::kFloat, "wts");
    check(out, at::kFloat, "out");
    TORCH_CHECK(D % 128 == 0, "D must be a multiple of 128");
    const float* r = nullptr;
    if (res.has_value()) {
        check(*res, at::kFloat, "res");
        TORCH_CHECK(res->numel() == rows * D && out.numel() >= rows * D, "res: fp32 [rows, D]");
        r = res->data_ptr<float>();
    }
    const void* x = nullptr;
    int64_t I = 0;
    if (discard) {
        TORCH_CHECK(xd.has_value(), "discard needs xd");
        check(*xd, at::kHalf, "xd");
        I = xd->size(1);
        TORCH_CHECK(xd->dim() == 2 && xd->size(0) >= P && I % 128 == 0, "xd: [>= P, I], I a multiple of 128");
        discard = line_aligned(Z) && line_aligned(*xd);
        x = xd->data_ptr();
    }
    c10::cuda::CUDAGuard guard(Z.device());
    exl3x_down_combine_cuda(Z, pick, svh_d, y, wts, out, rows, P, D, SK, slots, E, r, store_y, x, I, discard);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("grouped", &grouped);
    m.def("dequant", &dequant);
    m.def("group", &group);
    m.def("rot_in", &rot_in);
    m.def("gateup_epilogue", &gateup_epilogue, py::arg("Z"), py::arg("pick"), py::arg("svh_g"), py::arg("svh_u"),
          py::arg("suh_d"), py::arg("xd"), py::arg("rows"), py::arg("P"), py::arg("N"), py::arg("SK"), py::arg("slots"),
          py::arg("E"), py::arg("limit"), py::arg("act_mode"), py::arg("xg") = py::none(), py::arg("xu") = py::none(),
          py::arg("zlive") = 0, py::arg("discard") = 0);
    m.def("down_epilogue", &down_epilogue);
    m.def("combine", &combine);
    m.def("down_combine", &down_combine, py::arg("Z"), py::arg("pick"), py::arg("svh_d"), py::arg("y"), py::arg("wts"),
          py::arg("out"), py::arg("rows"), py::arg("P"), py::arg("D"), py::arg("SK"), py::arg("slots"), py::arg("E"),
          py::arg("res") = py::none(), py::arg("store_y") = 1, py::arg("xd") = py::none(), py::arg("discard") = 0);
}
