// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jay Leaton. The DeepSeek-V4.1-Flash family of TensorFold (Apache-2.0): see THIRD_PARTY_NOTICES.md.
// Modified for TensorFold's serial DeepSeek-V4.1 engine (2026-10-06): his l2pace.cu kernel and l2pace.cpp binding in
// one file (the binding takes our kernels.bulk_table's (address, bytes) pair as an int64 [n, 2] tensor and launches
// on the current stream, so the decode graphs capture it); the kernel is unchanged but for its namespace. Used by the
// TF_L2_PREFETCH=bulk sites of serial.py when TF_L2_PACE_GBPS > 0 (kernels.l2_paced).
//
// The L2 prefetch of a site PACED, so it stops starving the exchange (all-gather) beside it.
//
// Why. An unpaced bulk prefetch (kernels._l2_bulk) hands every 32 KiB piece of a site to its own lane: all of it is
// requested at once, and anything else that needs DRAM or the C2C path in the next tens of us (the all-gather the
// site is forked beside: a chain of dependent round trips) waits behind it.
//
// How. One thread per CTA (``ctas`` CTAs, 32 threads each, the rest idle) walks the site's table in order, the pieces
// dealt round-robin over the CTAs, and issues piece g no earlier than t0 + delay + g x ns_per_piece (%globaltimer).
// The prefetch then streams at ``rate`` = chunk / ns_per_piece with only ~rate x latency bytes in flight. ``delay``
// lets the exchange's stage pass first.
//
// Exactness: it only READS weight memory into L2 (cp.async.bulk.prefetch.L2) and writes nothing.

#include <stdint.h>

#include <stdexcept>
#include <string>

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace tf_dsv41_l2pace {

__device__ __forceinline__ uint64_t now_ns() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ void bulk_prefetch(const void *p, uint32_t bytes) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p), "r"(bytes) : "memory");
#else
    for (uint32_t o = 0; o < bytes; o += 128) asm volatile("prefetch.global.L2 [%0];" ::"l"((const char *)p + o));
#endif
}

__device__ __forceinline__ void wait_until(uint64_t due) {
    while (true) {
        const uint64_t t = now_ns();
        if (t >= due) {
            return;
        }
        const uint64_t left = due - t;
        __nanosleep(static_cast<unsigned>(left > 2000u ? 1000u : (left > 64u ? left / 2u : 32u)));
    }
}

// table: n rows of int64 (address, bytes): 16-byte aligned addresses, byte counts multiples of 16 (bulk_table)
__global__ void __launch_bounds__(32) paced_kernel(const int64_t *table, int n, uint32_t chunk, uint64_t ns_per_piece,
                                                   uint64_t delay_ns) {
    if (threadIdx.x != 0) {
        return;
    }
    const uint64_t c = blockIdx.x;
    const uint64_t g = gridDim.x;
    const uint64_t t0 = now_ns() + delay_ns;
    uint64_t k = 0;                                     // the piece's index over the whole table
    for (int s = 0; s < n; s++) {
        const uint8_t *p = reinterpret_cast<const uint8_t *>(static_cast<uintptr_t>(table[2 * s]));
        const uint64_t bytes = static_cast<uint64_t>(table[2 * s + 1]);
        for (uint64_t off = 0; off < bytes; off += chunk, k++) {
            if (k % g != c) {
                continue;
            }
            wait_until(t0 + k * ns_per_piece);
            const uint64_t left = bytes - off;
            bulk_prefetch(p + off, static_cast<uint32_t>(left < chunk ? left : chunk));
        }
    }
}

// table: int64 [n, 2] (address, bytes) on the device; pieces of ``chunk`` bytes, one every ``ns_per_piece`` over all
// ``ctas`` CTAs, the first ``delay_ns`` after the kernel starts
void paced(torch::Tensor table, int64_t ctas, int64_t chunk, int64_t ns_per_piece, int64_t delay_ns) {
    TORCH_CHECK(table.is_cuda() && table.scalar_type() == torch::kInt64 && table.dim() == 2 && table.size(1) == 2 &&
                    table.is_contiguous(),
                "l2pace.paced: an int64 [n, 2] contiguous CUDA table");
    if (chunk < 16 || chunk % 16 != 0 || chunk > (int64_t(1) << 24)) {
        throw std::invalid_argument("l2pace: chunk must be a multiple of 16 bytes in [16, 16 MiB]");
    }
    if (ctas < 1 || ctas > 64 || ns_per_piece < 0 || delay_ns < 0) {
        throw std::invalid_argument("l2pace: ctas 1..64, ns_per_piece and delay_ns >= 0");
    }
    if (table.size(0) == 0) {
        return;
    }
    paced_kernel<<<int(ctas), 32, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(
        table.data_ptr<int64_t>(), int(table.size(0)), uint32_t(chunk), uint64_t(ns_per_piece), uint64_t(delay_ns));
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("l2pace: launch: ") + cudaGetErrorString(e));
    }
}

}  // namespace tf_dsv41_l2pace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("paced", &tf_dsv41_l2pace::paced);
}
