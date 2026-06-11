#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <cstdint>

namespace qdp {
namespace native {

// ---------------------------------------------------------------------------
// ImplicitHadamardNativeEngine
//
// PR011: Native FP32/FP16 Blocked TC-FWT
//
// Bypasses the Ozaki Scheme entirely to provide extremely fast FP32/FP16
// execution for Quantum Machine Learning (QML) and other applications where
// FP64 numerical stability is not strictly required.
// ---------------------------------------------------------------------------
class ImplicitHadamardNativeEngine {
public:
    explicit ImplicitHadamardNativeEngine() = default;
    ~ImplicitHadamardNativeEngine() = default;

    // Compute C = A × H_n × norm_factor  (H_n is the n×n Walsh-Hadamard matrix)
    // A: [m × k] FP32 row-major on device
    // C: [m × n] FP32 row-major on device (overwritten; must be pre-allocated)
    // k == n
    void execute_implicit_hadamard_fp32(
        const float* d_A,
        float*       d_C,
        size_t m, size_t n, size_t k,
        float norm_factor,
        cudaStream_t stream = 0,
        bool transpose_batch = false,
        size_t batch_rows = 0
    );

    void execute_implicit_hadamard_fp64(
        const double* d_A,
        double*       d_C,
        size_t m, size_t n, size_t k,
        double norm_factor,
        cudaStream_t stream = 0,
        bool transpose_batch = false,
        size_t batch_rows = 0
    );

    // PR9: Fuse IQP phase encoding with native_fp64_extreme_fwt (PR8's fastest kernel).
    void execute_iqp_fused_fp64(
        const double* d_data_batch,
        void*         d_state_batch,
        size_t        num_samples,
        unsigned int  num_qubits,
        unsigned int  data_len,
        int           enable_zz,
        cudaStream_t  stream = 0
    );

    // PR9c: FP32 fused IQP path for QML (mirrors execute_iqp_fused_fp64).
    void execute_iqp_fused_fp32(
        const float* d_data_batch,
        void*        d_state_batch,
        size_t       num_samples,
        unsigned int num_qubits,
        unsigned int data_len,
        int          enable_zz,
        cudaStream_t stream = 0
    );

};

} // namespace native
} // namespace qdp
