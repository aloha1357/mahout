//
// Licensed to the Apache Software Foundation (ASF) under one or more
// contributor license agreements.  See the NOTICE file distributed with
// this work for additional information regarding copyright ownership.
// The ASF licenses this file to You under the Apache License, Version 2.0
// (the "License"); you may not use this file except in compliance with
// the License.  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <iostream>
#include <iomanip>
#include <chrono>

// ============================================================================
// TDD-1: Pure SIMT Register-based Warp FWT (N=5, 32 elements)
// Zero shared memory, zero global sync, purely __shfl_xor_sync
// ============================================================================
__global__ void warp_fwt_batch_kernel(cuDoubleComplex* state, size_t num_samples) {
    // Each warp processes one sample (32 elements)
    size_t sample_idx = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if (sample_idx >= num_samples) return;

    int lane_id = threadIdx.x % 32;
    size_t idx = sample_idx * 32 + lane_id;

    // Load from Global to Register
    cuDoubleComplex val = state[idx];

    // 5 Stages of FWT strictly in registers!
    #pragma unroll
    for (int d = 0; d < 5; ++d) {
        int mask = 1 << d;
        double r_other = __shfl_xor_sync(0xffffffff, val.x, mask);
        double i_other = __shfl_xor_sync(0xffffffff, val.y, mask);
        
        if ((lane_id & mask) == 0) {
            val.x = val.x + r_other;
            val.y = val.y + i_other;
        } else {
            val.x = r_other - val.x;
            val.y = i_other - val.y;
        }
    }
    
    // Store back to Global
    state[idx] = val;
}

// ============================================================================
// TDD-2 Baseline: Shared Memory FWT (Current approach)
// ============================================================================
__global__ void shared_fwt_batch_kernel(cuDoubleComplex* state, size_t num_samples) {
    extern __shared__ cuDoubleComplex smem[];
    
    size_t sample_idx = blockIdx.x;
    if (sample_idx >= num_samples) return;

    size_t tid = threadIdx.x; // 32 threads per block
    size_t idx = sample_idx * 32 + tid;
    
    // Load to Shared Memory
    smem[tid] = state[idx];
    __syncthreads();

    // 5 Stages of FWT in Shared Memory
    for (int stage = 0; stage < 5; ++stage) {
        size_t stride = 1ULL << stage;
        size_t block_size = stride << 1;
        
        if (tid < 16) {
            size_t block_idx = tid / stride;
            size_t pair_offset = tid % stride;
            size_t i = block_idx * block_size + pair_offset;
            size_t j = i + stride;

            cuDoubleComplex a = smem[i];
            cuDoubleComplex b = smem[j];

            smem[i] = make_cuDoubleComplex(a.x + b.x, a.y + b.y);
            smem[j] = make_cuDoubleComplex(a.x - b.x, a.y - b.y);
        }
        __syncthreads();
    }
    
    // Write back
    state[idx] = smem[tid];
}

int main() {
    // Test for a massive batch of N=5 chunks to saturate GPU
    size_t num_samples = 1024 * 1024 * 2; // 2 Million samples of 32 elements (1GB total)
    size_t total_elements = num_samples * 32;
    size_t bytes = total_elements * sizeof(cuDoubleComplex);

    cuDoubleComplex* d_state;
    cudaMalloc(&d_state, bytes);

    std::cout << "====================================================\n";
    std::cout << " TDD-1 & TDD-2: Microkernel FWT Benchmark (N=5)\n";
    std::cout << " Batch Size: " << num_samples << " (Total 64M elements, 1 GB)\n";
    std::cout << "====================================================\n";

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0;

    // ----------------------------------------------------
    // 1. Shared Memory Benchmark (TDD-2 Baseline)
    // ----------------------------------------------------
    int threads_sm = 32;
    int blocks_sm = num_samples;
    size_t smem_size = 32 * sizeof(cuDoubleComplex);
    
    // Warmup
    shared_fwt_batch_kernel<<<blocks_sm, threads_sm, smem_size>>>(d_state, num_samples);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for(int i=0; i<10; i++) {
        shared_fwt_batch_kernel<<<blocks_sm, threads_sm, smem_size>>>(d_state, num_samples);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float smem_time = ms / 10.0f;
    std::cout << " [TDD-2] Shared Memory FWT : " << std::fixed << std::setprecision(3) << smem_time << " ms\n";

    // ----------------------------------------------------
    // 2. SIMT Warp Shuffle Benchmark (TDD-1)
    // ----------------------------------------------------
    int threads_warp = 256; // 8 warps per block
    int blocks_warp = (num_samples + 7) / 8;
    
    // Warmup
    warp_fwt_batch_kernel<<<blocks_warp, threads_warp>>>(d_state, num_samples);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for(int i=0; i<10; i++) {
        warp_fwt_batch_kernel<<<blocks_warp, threads_warp>>>(d_state, num_samples);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float warp_time = ms / 10.0f;
    std::cout << " [TDD-1] SIMT Register FWT : " << std::fixed << std::setprecision(3) << warp_time << " ms\n";
    
    float speedup = smem_time / warp_time;
    std::cout << "----------------------------------------------------\n";
    std::cout << " >> TDD-1 Speedup over TDD-2: " << speedup << "x\n";
    std::cout << "====================================================\n";

    cudaFree(d_state);
    return 0;
}
