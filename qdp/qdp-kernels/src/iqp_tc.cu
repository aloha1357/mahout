// iqp_tc.cu
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <iostream>
#include "kernel_config.h"
#include "ImplicitHadamardOzaki.h"

// Phase 計算副程式 (從 iqp.cu 中借用)
__device__ double compute_phase_tc(
    const double* __restrict__ data,
    size_t x,
    unsigned int num_qubits,
    int enable_zz
) {
    double phase = 0.0;
    for (unsigned int i = 0; i < num_qubits; ++i) {
        if ((x >> i) & 1U) {
            phase += data[i];
        }
    }
    if (enable_zz) {
        unsigned int pair_idx = num_qubits;
        for (unsigned int i = 0; i < num_qubits; ++i) {
            for (unsigned int j = i + 1; j < num_qubits; ++j) {
                if (((x >> i) & 1U) && ((x >> j) & 1U)) {
                    phase += data[pair_idx];
                }
                pair_idx++;
            }
        }
    }
    return phase;
}

// PR-C: 算子融合 (Operator Fusion) - 將 Phase 計算, FWT, Normalize 融合在 Shared Memory
__global__ void iqp_phase_fwt_normalize_tc_kernel(
    const double* __restrict__ data_batch,
    cuDoubleComplex* __restrict__ state_batch,
    size_t num_samples,
    size_t state_len,
    unsigned int num_qubits,
    unsigned int data_len,
    int enable_zz,
    double norm_factor
) {
    extern __shared__ cuDoubleComplex shared_state[];

    size_t tid = threadIdx.x;
    size_t sample_idx = blockIdx.x;

    if (sample_idx >= num_samples) return;

    const double* data = data_batch + sample_idx * data_len;
    cuDoubleComplex* state = state_batch + sample_idx * state_len;

    // 1. Phase 計算直接寫入 Shared Memory
    for (size_t i = tid; i < state_len; i += blockDim.x) {
        double phase = compute_phase_tc(data, i, num_qubits, enable_zz);
        double cos_phase, sin_phase;
        sincos(phase, &sin_phase, &cos_phase);
        shared_state[i] = make_cuDoubleComplex(cos_phase, sin_phase);
    }
    __syncthreads();

    // 2. 利用 Shared Memory 進行 Hadamard FWT 轉換
    for (unsigned int stage = 0; stage < num_qubits; ++stage) {
        size_t stride = 1ULL << stage;
        size_t block_size = stride << 1;
        size_t num_pairs = state_len >> 1;

        for (size_t pair_idx = tid; pair_idx < num_pairs; pair_idx += blockDim.x) {
            size_t block_idx = pair_idx / stride;
            size_t pair_offset = pair_idx % stride;
            size_t i = block_idx * block_size + pair_offset;
            size_t j = i + stride;

            cuDoubleComplex a = shared_state[i];
            cuDoubleComplex b = shared_state[j];

            shared_state[i] = cuCadd(a, b);
            shared_state[j] = cuCsub(a, b);
        }
        __syncthreads();
    }

    // 3. Normalize 並寫回 Global Memory
    for (size_t i = tid; i < state_len; i += blockDim.x) {
        cuDoubleComplex val = shared_state[i];
        state[i] = make_cuDoubleComplex(
            cuCreal(val) * norm_factor,
            cuCimag(val) * norm_factor
        );
    }
}

// Phase 2: GEMM 準備 - 將 Batch 展開並計算初始 Phase (純實數/虛數分離)
__global__ void iqp_phase_split_kernel(
    const double* __restrict__ data_batch,
    double* __restrict__ state_real,
    double* __restrict__ state_imag,
    size_t num_samples,
    size_t state_len,
    unsigned int num_qubits,
    unsigned int data_len,
    int enable_zz
) {
    const size_t total_elements = num_samples * state_len;
    const size_t stride = gridDim.x * blockDim.x;
    const size_t state_mask = state_len - 1;

    for (size_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
         global_idx < total_elements;
         global_idx += stride) {
        const size_t sample_idx = global_idx >> num_qubits;
        const size_t x = global_idx & state_mask;
        const double* data = data_batch + sample_idx * data_len;

        double phase = compute_phase_tc(data, x, num_qubits, enable_zz);
        double cos_phase, sin_phase;
        sincos(phase, &sin_phase, &cos_phase);
        
        state_real[global_idx] = cos_phase;
        state_imag[global_idx] = sin_phase;
    }
}

// GEMM 結果重新組合回 cuDoubleComplex
__global__ void recombine_complex_kernel(
    const double* __restrict__ real_part,
    const double* __restrict__ imag_part,
    cuDoubleComplex* __restrict__ out,
    size_t total_elements
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        out[idx] = make_cuDoubleComplex(real_part[idx], imag_part[idx]);
    }
}

extern "C" int launch_iqp_encode_tc(
    const double* data_batch_d,
    void*         state_batch_d,
    size_t        num_samples,
    size_t        state_len,
    unsigned int  num_qubits,
    int           enable_zz,
    cudaStream_t  stream
) {
    // 判斷是否滿足 Shared Memory FWT 的條件
    if (num_qubits <= FWT_SHARED_MEM_THRESHOLD) {
        // 使用 PR-C 的融合算子
        double norm_factor = 1.0 / (double)state_len;
        unsigned int data_len = num_qubits; 
        iqp_phase_fwt_normalize_tc_kernel<<<num_samples, DEFAULT_BLOCK_SIZE, state_len * sizeof(cuDoubleComplex), stream>>>(
            data_batch_d, static_cast<cuDoubleComplex*>(state_batch_d), num_samples, state_len, num_qubits, data_len, enable_zz, norm_factor
        );
    } else {
        // [Phase 3] FWT to Matrix-Free GEMM
        size_t m_samples = num_samples;
        size_t n_dim = state_len;
        size_t k_dim = state_len;
        
        double *d_state_real, *d_state_imag;
        double *d_out_real, *d_out_imag;
        cudaMalloc(&d_state_real, m_samples * k_dim * sizeof(double));
        cudaMalloc(&d_state_imag, m_samples * k_dim * sizeof(double));
        cudaMalloc(&d_out_real, m_samples * n_dim * sizeof(double));
        cudaMalloc(&d_out_imag, m_samples * n_dim * sizeof(double));

        // 1. 初始化 Phase (拆分為 Real / Imag)
        unsigned int data_len = num_qubits;
        const size_t total_elements = m_samples * k_dim;
        const size_t blocks = (total_elements + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE;
        iqp_phase_split_kernel<<<blocks, DEFAULT_BLOCK_SIZE, 0, stream>>>(
            data_batch_d, d_state_real, d_state_imag, num_samples, state_len, num_qubits, data_len, enable_zz
        );

        // 2. 準備 Implicit Engine
        ozaki::OzakiConfig config;
        ozaki::ImplicitHadamardOzakiEngine engine(config);
        
        // 3. 執行 Matrix-Free Implicit Hadamard Tensor Core GEMM
        double norm_factor = 1.0 / (double)state_len;
        
        // 實部計算: out_real = state_real * H (Implicit)
        engine.execute_implicit_hadamard(d_state_real, d_out_real, m_samples, n_dim, k_dim, norm_factor);
        
        // 虛部計算: out_imag = state_imag * H (Implicit)
        engine.execute_implicit_hadamard(d_state_imag, d_out_imag, m_samples, n_dim, k_dim, norm_factor);

        // 4. 清理與寫回
        recombine_complex_kernel<<<blocks, DEFAULT_BLOCK_SIZE, 0, stream>>>(
            d_out_real, d_out_imag, static_cast<cuDoubleComplex*>(state_batch_d), total_elements
        );

        cudaFree(d_state_real);
        cudaFree(d_state_imag);
        cudaFree(d_out_real);
        cudaFree(d_out_imag);
    }

    return (int)cudaSuccess;
}
