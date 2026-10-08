// Binding of the two-rank all-gather over RoCE (gather.cu); protocol after b12x RoCEnante via MiaAI-Lab patch 0006,
// Apache-2.0; see THIRD_PARTY_NOTICES.md.
#include <cuda_runtime.h>
#include <torch/extension.h>

void rdma_gather(const at::Tensor& in, at::Tensor& out, int64_t region, int64_t flag_off, int64_t send_off,
                 int64_t recv_off, int64_t slot_bytes, at::Tensor& state, int64_t spin, int64_t rank);

// The device address of host memory registered with cudaHostRegister (0 when it has none).
int64_t device_pointer(int64_t host) {
    void* dev = nullptr;
    if (cudaHostGetDevicePointer(&dev, reinterpret_cast<void*>(host), 0) != cudaSuccess) {
        (void)cudaGetLastError();
        return 0;
    }
    return reinterpret_cast<int64_t>(dev);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gather", &rdma_gather, "two-rank all-gather over RoCE");
    m.def("device_pointer", &device_pointer, "the device address of registered host memory (0: none)");
}
