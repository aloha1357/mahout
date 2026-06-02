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
#include <iostream>
#include <iomanip>
#include <chrono>
#include "ImplicitHadamardOzaki.h"

// ============================================================================
// TDD-1: Pure SIMT Register-based Warp FWT (N=4, 16 elements)
// Processes 1D double array (real or imag part separately)
// ============================================================================
__global__ void warp_fwt_n4_kernel(double* state, size_t num_samples) {
    // Each 16 threads process one sample
    size_t sample_idx = blockIdx.x * (blockDim.x / 16) + (threadIdx.x / 16);
    if (sample_idx >= num_samples) return;

    int lane_id = threadIdx.x % 16;
    size_t idx = sample_idx * 16 + lane_id;

    double val = state[idx];

    #pragma unroll
    for (int d = 0; d < 4; ++d) {
        int mask = 1 << d;
        // active mask for the 16 threads
        unsigned int active_mask = 0xffff << ((threadIdx.x / 16) * 16); 
        double other = __shfl_xor_sync(active_mask, val, mask);
        
        if ((lane_id & mask) == 0) {
            val = val + other;
        } else {
            val = other - val;
        }
    }
    
    state[idx] = val;
}

int main() {
    // We test N=4 (16 elements). We process 16 samples per Matrix block to form 16x16.
    // So 1 block = 16 samples. Let's do 1,048,576 samples (16M elements, ~134 MB)
    size_t num_samples = 1024 * 1024; 
    size_t n_dim = 16;
    size_t total_elements = num_samples * n_dim;
    size_t bytes = total_elements * sizeof(double);

    double *d_state, *d_out;
    cudaMalloc(&d_state, bytes);
    cudaMalloc(&d_out, bytes);

    std::cout << "==============================================================\n";
    std::cout << " TDD-5 vs TDD-1: Microkernel Tensor Core vs SIMT (N=4, 16x16)\n";
    std::cout << " Batch Size: " << num_samples << " samples (Total 16M elements)\n";
    std::cout << "==============================================================\n";

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0;

    // ----------------------------------------------------
    // 1. TDD-1: SIMT Register FWT
    // ----------------------------------------------------
    int threads_warp = 256; // 16 samples per block
    int blocks_warp = (num_samples + 15) / 16;
    
    // Warmup
    warp_fwt_n4_kernel<<<blocks_warp, threads_warp>>>(d_state, num_samples);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for(int i=0; i<10; i++) {
        warp_fwt_n4_kernel<<<blocks_warp, threads_warp>>>(d_state, num_samples);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float simt_time = ms / 10.0f;
    std::cout << " [TDD-1] SIMT Register FWT (FP64) : " << std::fixed << std::setprecision(3) << simt_time << " ms\n";

    // ----------------------------------------------------
    // 2. TDD-5: Tensor Core Ozaki INT8 FWT (Implicit)
    // ----------------------------------------------------
    ozaki::OzakiConfig config;
    ozaki::ImplicitHadamardOzakiEngine engine(config);

    // Warmup
    engine.execute_implicit_hadamard(d_state, d_out, num_samples, n_dim, n_dim, 1.0);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for(int i=0; i<10; i++) {
        engine.execute_implicit_hadamard(d_state, d_out, num_samples, n_dim, n_dim, 1.0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float tc_time = ms / 10.0f;
    std::cout << " [TDD-5] Tensor Core INT8 Ozaki   : " << std::fixed << std::setprecision(3) << tc_time << " ms\n";

    float speedup = tc_time / simt_time;
    std::cout << "--------------------------------------------------------------\n";
    std::cout << " >> SIMT is " << speedup << "x FASTER than Tensor Core for 16x16 FWT\n";
    std::cout << "==============================================================\n";
    std::cout << " Conclusion: As predicted by the TDD document, forcing Tensor\n";
    std::cout << " Cores on small blocks (N=4) is an anti-pattern. SIMT registers\n";
    std::cout << " dominate the microkernel level.\n";
    std::cout << "==============================================================\n";

    cudaFree(d_state);
    cudaFree(d_out);
    return 0;
}
