#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include "ImplicitHadamardOzaki.h"

#define TRANSPOSE_TILE_DIM 32
#define TRANSPOSE_BLOCK_ROWS 8

// Shared Memory Bank-Conflict-Free Batch Transpose
__global__ void batch_transpose_kernel(const double* __restrict__ in, double* __restrict__ out, int B, int rows, int cols) {
    // TILE_DIM x (TILE_DIM+1) pad to avoid shared memory bank conflicts
    __shared__ double tile[TRANSPOSE_TILE_DIM][TRANSPOSE_TILE_DIM + 1];

    int b = blockIdx.z;
    int x = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.y;

    // Load from global memory (coalesced) into shared memory
    for (int j = 0; j < TRANSPOSE_TILE_DIM; j += TRANSPOSE_BLOCK_ROWS) {
        if (x < cols && (y + j) < rows) {
            tile[threadIdx.y + j][threadIdx.x] = in[b * rows * cols + (y + j) * cols + x];
        }
    }

    __syncthreads();

    // Transposed block coordinates
    x = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.x; 
    y = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.y;

    // Store from shared memory to global memory (coalesced)
    for (int j = 0; j < TRANSPOSE_TILE_DIM; j += TRANSPOSE_BLOCK_ROWS) {
        if (x < rows && (y + j) < cols) {
            out[b * rows * cols + (y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

void launch_batch_transpose(const double* d_in, double* d_out, int B, int rows, int cols) {
    dim3 block(TRANSPOSE_TILE_DIM, TRANSPOSE_BLOCK_ROWS, 1);
    dim3 grid((cols + TRANSPOSE_TILE_DIM - 1) / TRANSPOSE_TILE_DIM, 
              (rows + TRANSPOSE_TILE_DIM - 1) / TRANSPOSE_TILE_DIM, B);
    batch_transpose_kernel<<<grid, block>>>(d_in, d_out, B, rows, cols);
}

int main() {
    int n_qubits = 14;
    int batch_size = 128;
    
    size_t state_len = 1ULL << n_qubits;
    size_t total_elements = batch_size * state_len;
    size_t bytes = total_elements * sizeof(double);

    double *d_state, *d_out_old, *d_out_new, *d_temp;
    cudaMalloc(&d_state, bytes);
    cudaMalloc(&d_out_old, bytes);
    cudaMalloc(&d_out_new, bytes);
    cudaMalloc(&d_temp, bytes);

    // Initialize with 1.0
    cudaMemset(d_state, 0, bytes); // For simplicity, just test execution time, not correctness here yet
    
    std::cout << "==============================================================\n";
    std::cout << " TC-FWT Algorithmic Breakthrough Benchmark\n";
    std::cout << " N = " << n_qubits << ", Batch = " << batch_size << " (Total " << total_elements << " elements)\n";
    std::cout << "==============================================================\n";

    ozaki::OzakiConfig config;
    ozaki::ImplicitHadamardOzakiEngine engine(config);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0;

    // ----------------------------------------------------
    // 1. OLD Approach: O(4^N) Massive GEMM
    // ----------------------------------------------------
    double norm = 1.0;
    
    cudaEventRecord(start);
    engine.execute_implicit_hadamard(d_state, d_out_old, batch_size, state_len, state_len, norm);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float old_time = ms;
    std::cout << " [OLD] Massive O(4^N) Tensor Core GEMM : " << std::fixed << std::setprecision(3) << old_time << " ms\n";

    // ----------------------------------------------------
    // 2. NEW Approach: O(N * 2^N) Blocked TC-FWT
    // Split N=14 into n1=7, n2=7
    // ----------------------------------------------------
    int n1 = n_qubits / 2;
    int n2 = n_qubits - n1;
    int dim1 = 1 << n1;
    int dim2 = 1 << n2;

    cudaEventRecord(start);
    
    // Step 1: Z = X * H_{n2}
    // X is (B * dim1) x dim2. H is dim2 x dim2.
    engine.execute_implicit_hadamard(d_state, d_out_new, batch_size * dim1, dim2, dim2, 1.0);
    
    // Step 2: Transpose Z to Z_T
    // Z is (B, dim1, dim2). Z_T is (B, dim2, dim1)
    launch_batch_transpose(d_out_new, d_temp, batch_size, dim1, dim2);

    // Step 3: Y_T = Z_T * H_{n1}
    // Z_T is (B * dim2) x dim1. H is dim1 x dim1.
    engine.execute_implicit_hadamard(d_temp, d_out_new, batch_size * dim2, dim1, dim1, norm);

    // Step 4: Transpose Y_T back to Y
    // Y_T is (B, dim2, dim1). Y is (B, dim1, dim2)
    launch_batch_transpose(d_out_new, d_temp, batch_size, dim2, dim1);
    
    // Result is now in d_temp
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float new_time = ms;
    
    std::cout << " [NEW] Blocked O(N 2^N) TC-FWT GEMM  : " << std::fixed << std::setprecision(3) << new_time << " ms\n";

    float speedup = old_time / new_time;
    std::cout << "--------------------------------------------------------------\n";
    std::cout << " >> TC-FWT is " << speedup << "x FASTER than Massive GEMM!\n";
    std::cout << "==============================================================\n";

    cudaFree(d_state);
    cudaFree(d_out_old);
    cudaFree(d_out_new);
    cudaFree(d_temp);
    return 0;
}
