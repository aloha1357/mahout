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

#include "ImplicitHadamardNative.h"
#include <cmath>
#include <stdio.h>

namespace qdp {
namespace native {

// A highly optimized but simple shared-memory butterfly FWT for FP32/FP16.
// For large N, we would use Kronecker decomposition, but here we provide the
// direct butterfly implementation for performance benchmarking of the "Native FP32" path
// without the Ozaki overhead.
__global__ void native_fp32_butterfly_kernel(
    float* __restrict__ state,
    size_t m, size_t n, size_t k, float norm_factor
) {
    // Treat 'm' as batch_size, 'n' as state_len
    size_t batch_idx = blockIdx.y;
    size_t state_len = n;
    int pairs_per_sample = state_len >> 1;

    // Pointer to this batch's state
    float* local_state = state + batch_idx * state_len;

    int num_qubits = 0;
    while ((1ULL << num_qubits) < state_len) num_qubits++;

    // We can do global memory butterfly stages:
    for (int stage = 0; stage < num_qubits; ++stage) {
        int stride = 1 << stage;
        int block_size = stride << 1;

        for (int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
             pair_idx < pairs_per_sample;
             pair_idx += gridDim.x * blockDim.x) {

            int block_idx = pair_idx / stride;
            int pair_offset = pair_idx % stride;
            int i = block_idx * block_size + pair_offset;
            int j = i + stride;

            float a = local_state[i];
            float b = local_state[j];

            local_state[i] = a + b;
            local_state[j] = a - b;
        }
        __syncthreads(); // Only valid if grid handles one batch sequentially, or we split kernel launches.
    }
}

// Global Memory Butterfly - First Stage
__global__ void native_fp32_butterfly_first_stage_kernel(
    const float* __restrict__ in_state,
    float* __restrict__ out_state,
    size_t state_len,
    size_t num_samples,
    float norm_factor,
    bool is_last_stage
) {
    int stride = 1;
    int block_size = 2;
    int pairs_per_sample = state_len >> 1;
    int total_pairs = num_samples * pairs_per_sample;

    for (int global_pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
         global_pair_idx < total_pairs;
         global_pair_idx += gridDim.x * blockDim.x) {

         int sample_idx = global_pair_idx / pairs_per_sample;
         int pair_idx = global_pair_idx % pairs_per_sample;

         int block_idx = pair_idx / stride;
         int pair_offset = pair_idx % stride;
         int i = sample_idx * state_len + block_idx * block_size + pair_offset;
         int j = i + stride;

         float a = in_state[i];
         float b = in_state[j];

         float sum = a + b;
         float diff = a - b;

         if (is_last_stage && norm_factor != 1.0f) {
             sum *= norm_factor;
             diff *= norm_factor;
         }

         out_state[i] = sum;
         out_state[j] = diff;
    }
}

// Intra-Block Optimized Butterfly FWT (Registers + SMEM)
__global__ void native_fp32_opt_butterfly_kernel(
    const float* __restrict__ in_state,
    float* __restrict__ out_state,
    size_t state_len,
    size_t num_samples,
    float norm_factor
) {
    size_t batch_idx = blockIdx.x;
    if (batch_idx >= num_samples) return;

    int tid = threadIdx.x;

    const float4* in_batch = reinterpret_cast<const float4*>(in_state + batch_idx * state_len);
    float4* out_batch = reinterpret_cast<float4*>(out_state + batch_idx * state_len);

    float4 v = in_batch[tid];

    // Stage 0: Stride 1 (in-thread)
    float a0 = v.x + v.y;
    float b0 = v.x - v.y;
    float a1 = v.z + v.w;
    float b1 = v.z - v.w;

    // Stage 1: Stride 2 (in-thread)
    v.x = a0 + a1;
    v.y = b0 + b1;
    v.z = a0 - a1;
    v.w = b0 - b1;

    int num_qubits = 0;
    while ((1ULL << num_qubits) < state_len) num_qubits++;

    // Stages 2 to min(num_qubits-1, 6): Warp shuffles
    int max_shuffle_stage = (num_qubits < 7) ? num_qubits : 7;
    for (int stage = 2; stage < max_shuffle_stage; ++stage) {
        int stride_thread = 1 << (stage - 2);

        float4 peer_v;
        peer_v.x = __shfl_xor_sync(0xffffffff, v.x, stride_thread);
        peer_v.y = __shfl_xor_sync(0xffffffff, v.y, stride_thread);
        peer_v.z = __shfl_xor_sync(0xffffffff, v.z, stride_thread);
        peer_v.w = __shfl_xor_sync(0xffffffff, v.w, stride_thread);

        if ((tid & stride_thread) == 0) {
            v.x = v.x + peer_v.x;
            v.y = v.y + peer_v.y;
            v.z = v.z + peer_v.z;
            v.w = v.w + peer_v.w;
        } else {
            v.x = peer_v.x - v.x;
            v.y = peer_v.y - v.y;
            v.z = peer_v.z - v.z;
            v.w = peer_v.w - v.w;
        }
    }

    if (num_qubits <= 7) {
        if (norm_factor != 1.0f) {
            v.x *= norm_factor;
            v.y *= norm_factor;
            v.z *= norm_factor;
            v.w *= norm_factor;
        }
        out_batch[tid] = v;
        return;
    }

    extern __shared__ float smem[];

    int idx0 = tid * 4;
    int idx1 = idx0 + 1;
    int idx2 = idx0 + 2;
    int idx3 = idx0 + 3;

    smem[idx0 + (idx0 >> 5)] = v.x;
    smem[idx1 + (idx1 >> 5)] = v.y;
    smem[idx2 + (idx2 >> 5)] = v.z;
    smem[idx3 + (idx3 >> 5)] = v.w;

    __syncthreads();

    int num_threads = blockDim.x;
    for (int stage = 7; stage < num_qubits; ++stage) {
        int stride = 1 << stage;
        int block_size = stride << 1;

        for (int p = 0; p < 2; ++p) {
            int pair_idx = tid + p * num_threads;
            if (pair_idx < (state_len >> 1)) {
                int block_idx = pair_idx / stride;
                int pair_offset = pair_idx % stride;
                int i = block_idx * block_size + pair_offset;
                int j = i + stride;

                int smem_i = i + (i >> 5);
                int smem_j = j + (j >> 5);

                float a = smem[smem_i];
                float b = smem[smem_j];

                smem[smem_i] = a + b;
                smem[smem_j] = a - b;
            }
        }
        __syncthreads();
    }

    v.x = smem[idx0 + (idx0 >> 5)];
    v.y = smem[idx1 + (idx1 >> 5)];
    v.z = smem[idx2 + (idx2 >> 5)];
    v.w = smem[idx3 + (idx3 >> 5)];

    if (norm_factor != 1.0f) {
        v.x *= norm_factor;
        v.y *= norm_factor;
        v.z *= norm_factor;
        v.w *= norm_factor;
    }

    out_batch[tid] = v;
}

// Global Memory Butterfly - Subsequent Stages
__global__ void native_fp32_butterfly_stage_kernel(
    float* __restrict__ state,
    size_t state_len,
    int stage,
    size_t num_samples,
    float norm_factor,
    bool is_last_stage
) {
    int stride = 1 << stage;
    int block_size = stride << 1;
    int pairs_per_sample = state_len >> 1;
    int total_pairs = num_samples * pairs_per_sample;

    for (int global_pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
         global_pair_idx < total_pairs;
         global_pair_idx += gridDim.x * blockDim.x) {

         int sample_idx = global_pair_idx / pairs_per_sample;
         int pair_idx = global_pair_idx % pairs_per_sample;

         int block_idx = pair_idx / stride;
         int pair_offset = pair_idx % stride;
         int i = sample_idx * state_len + block_idx * block_size + pair_offset;
         int j = i + stride;

         float a = state[i];
         float b = state[j];

         float sum = a + b;
         float diff = a - b;

         if (is_last_stage && norm_factor != 1.0f) {
             sum *= norm_factor;
             diff *= norm_factor;
         }

         state[i] = sum;
         state[j] = diff;
    }
}

template <int TILE_COLS, int SMEM_PITCH>
__global__ void native_fp32_interblock_fwt_kernel(
    const float* __restrict__ in_state,
    float* __restrict__ out_state,
    int num_col_qubits,
    int num_row_qubits,
    float norm_factor
) {
    size_t batch_idx = blockIdx.y;
    int col_offset = blockIdx.x * TILE_COLS;

    size_t rows = 1ULL << num_row_qubits;
    size_t cols = 1ULL << num_col_qubits;

    const float* in_batch = in_state + batch_idx * (rows * cols);
    float* out_batch = out_state + batch_idx * (rows * cols);

    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    for (int idx = tid; idx < rows * TILE_COLS; idx += num_threads) {
        int r = idx / TILE_COLS;
        int c = idx % TILE_COLS;
        int global_idx = r * cols + (col_offset + c);
        smem[r * SMEM_PITCH + c] = in_batch[global_idx];
    }

    __syncthreads();

    for (int stage = 0; stage < num_row_qubits; ++stage) {
        int stride = 1 << stage;
        int block_size = stride << 1;

        int pairs_per_col = rows >> 1;
        int total_pairs = pairs_per_col * TILE_COLS;

        for (int idx = tid; idx < total_pairs; idx += num_threads) {
            int pair_idx = idx / TILE_COLS;
            int c = idx % TILE_COLS;

            int block_idx = pair_idx / stride;
            int pair_offset = pair_idx % stride;
            int r_i = block_idx * block_size + pair_offset;
            int r_j = r_i + stride;

            int smem_i = r_i * SMEM_PITCH + c;
            int smem_j = r_j * SMEM_PITCH + c;

            float a = smem[smem_i];
            float b = smem[smem_j];

            smem[smem_i] = a + b;
            smem[smem_j] = a - b;
        }
        __syncthreads();
    }

    for (int idx = tid; idx < rows * TILE_COLS; idx += num_threads) {
        int r = idx / TILE_COLS;
        int c = idx % TILE_COLS;
        int global_idx = r * cols + (col_offset + c);

        float val = smem[r * SMEM_PITCH + c];
        if (norm_factor != 1.0f) {
            val *= norm_factor;
        }
        out_batch[global_idx] = val;
    }
}

template<int N, int THREADS, int NUM_B>
__global__ void __launch_bounds__(THREADS) native_fp32_extreme_fwt_kernel(
    const float* __restrict__ in_state,
    float* __restrict__ out_state,
    float norm_factor
) {
    size_t chunk_idx = blockIdx.x;
    int tid = threadIdx.x;
    int CHUNK_LEN = 1 << N;

    const float4* in4 = reinterpret_cast<const float4*>(in_state + chunk_idx * (CHUNK_LEN >> 2));
    float4* out4 = reinterpret_cast<float4*>(out_state + chunk_idx * (CHUNK_LEN >> 2));

    float4 reg[NUM_B];

    #pragma unroll
    for (int b = 0; b < NUM_B; ++b) {
        reg[b] = in4[b * THREADS + tid];
    }

    #pragma unroll
    for (int b = 0; b < NUM_B; ++b) {
        float a0 = reg[b].x + reg[b].y; float b0 = reg[b].x - reg[b].y;
        float a1 = reg[b].z + reg[b].w; float b1 = reg[b].z - reg[b].w;
        reg[b].x = a0 + a1; reg[b].y = b0 + b1;
        reg[b].z = a0 - a1; reg[b].w = b0 - b1;
    }

    int max_shuffle = (N < 7) ? N : 7;
    #pragma unroll
    for (int stage = 2; stage < max_shuffle; ++stage) {
        int stride_thread = 1 << (stage - 2);
        #pragma unroll
        for (int b = 0; b < NUM_B; ++b) {
            float4 peer;
            peer.x = __shfl_xor_sync(0xffffffff, reg[b].x, stride_thread);
            peer.y = __shfl_xor_sync(0xffffffff, reg[b].y, stride_thread);
            peer.z = __shfl_xor_sync(0xffffffff, reg[b].z, stride_thread);
            peer.w = __shfl_xor_sync(0xffffffff, reg[b].w, stride_thread);

            if ((tid & stride_thread) == 0) {
                reg[b].x += peer.x; reg[b].y += peer.y;
                reg[b].z += peer.z; reg[b].w += peer.w;
            } else {
                reg[b].x = peer.x - reg[b].x; reg[b].y = peer.y - reg[b].y;
                reg[b].z = peer.z - reg[b].z; reg[b].w = peer.w - reg[b].w;
            }
        }
    }

    if (N <= 7) {
        #pragma unroll
        for (int b = 0; b < NUM_B; ++b) {
            if (norm_factor != 1.0f) {
                reg[b].x *= norm_factor; reg[b].y *= norm_factor; reg[b].z *= norm_factor; reg[b].w *= norm_factor;
            }
            out4[b * THREADS + tid] = reg[b];
        }
        return;
    }

    extern __shared__ float smem[];
    float4* smem4 = (float4*)smem;

    #pragma unroll
    for (int b = 0; b < NUM_B; ++b) {
        smem4[b * THREADS + tid] = reg[b];
    }
    __syncthreads();

    int max_smem_stage = (N < 10) ? N : 10;
    #pragma unroll
    for (int stage = 7; stage < max_smem_stage; ++stage) {
        int stride = 1 << stage;
        int block_size = stride << 1;
        int pairs_per_thread = (CHUNK_LEN >> 1) / THREADS;
        #pragma unroll
        for (int p = 0; p < pairs_per_thread; ++p) {
            int pair_idx = p * THREADS + tid;
            int block_idx = pair_idx / stride;
            int pair_offset = pair_idx % stride;
            int i = block_idx * block_size + pair_offset;
            int j = i + stride;

            float a = smem[i];
            float b_val = smem[j];
            smem[i] = a + b_val;
            smem[j] = a - b_val;
        }
        __syncthreads();
    }

    #pragma unroll
    for (int b = 0; b < NUM_B; ++b) {
        reg[b] = smem4[b * THREADS + tid];
    }

    if (N <= 10) {
        #pragma unroll
        for (int b = 0; b < NUM_B; ++b) {
            if (norm_factor != 1.0f) {
                reg[b].x *= norm_factor; reg[b].y *= norm_factor; reg[b].z *= norm_factor; reg[b].w *= norm_factor;
            }
            out4[b * THREADS + tid] = reg[b];
        }
        return;
    }

    #pragma unroll
    for (int stage = 10; stage < N; ++stage) {
        int b_stride = 1 << (stage - 10);
        int b_block_size = b_stride << 1;
        #pragma unroll
        for (int b_pair = 0; b_pair < (NUM_B >> 1); ++b_pair) {
            int b_block_idx = b_pair / b_stride;
            int b_pair_offset = b_pair % b_stride;
            int b_i = b_block_idx * b_block_size + b_pair_offset;
            int b_j = b_i + b_stride;

            float4 a = reg[b_i];
            float4 b_val = reg[b_j];

            reg[b_i].x = a.x + b_val.x; reg[b_j].x = a.x - b_val.x;
            reg[b_i].y = a.y + b_val.y; reg[b_j].y = a.y - b_val.y;
            reg[b_i].z = a.z + b_val.z; reg[b_j].z = a.z - b_val.z;
            reg[b_i].w = a.w + b_val.w; reg[b_j].w = a.w - b_val.w;
        }
    }

    #pragma unroll
    for (int b = 0; b < NUM_B; ++b) {
        if (norm_factor != 1.0f) {
            reg[b].x *= norm_factor; reg[b].y *= norm_factor; reg[b].z *= norm_factor; reg[b].w *= norm_factor;
        }
        out4[b * THREADS + tid] = reg[b];
    }
}

template<int N, int THREADS, int NUM_B>
__global__ void __launch_bounds__(THREADS)
native_fp32_extreme_fwt_kernel_v2(
    const float* __restrict__ in_state,
    float* __restrict__ out_state,
    float norm_factor
) {
    constexpr int CHUNK_LEN = 1 << N;
    size_t batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    const float4* in4 = reinterpret_cast<const float4*>(in_state + batch_idx * CHUNK_LEN);
    float4* out4 = reinterpret_cast<float4*>(out_state + batch_idx * CHUNK_LEN);

    float4 reg[NUM_B];
    #pragma unroll
    for (int b = 0; b < NUM_B; ++b) {
        reg[b] = in4[b * THREADS + tid];
    }

    #pragma unroll
    for (int b = 0; b < NUM_B; ++b) {
        float a0 = reg[b].x + reg[b].y; float b0 = reg[b].x - reg[b].y;
        float a1 = reg[b].z + reg[b].w; float b1 = reg[b].z - reg[b].w;
        reg[b].x = a0 + a1; reg[b].y = b0 + b1;
        reg[b].z = a0 - a1; reg[b].w = b0 - b1;
    }

    #pragma unroll
    for (int stage = 2; stage < 7; ++stage) {
        int stride_thread = 1 << (stage - 2);
        #pragma unroll
        for (int b = 0; b < NUM_B; ++b) {
            float4 peer;
            peer.x = __shfl_xor_sync(0xffffffff, reg[b].x, stride_thread);
            peer.y = __shfl_xor_sync(0xffffffff, reg[b].y, stride_thread);
            peer.z = __shfl_xor_sync(0xffffffff, reg[b].z, stride_thread);
            peer.w = __shfl_xor_sync(0xffffffff, reg[b].w, stride_thread);

            if ((tid & stride_thread) == 0) {
                reg[b].x += peer.x; reg[b].y += peer.y;
                reg[b].z += peer.z; reg[b].w += peer.w;
            } else {
                reg[b].x = peer.x - reg[b].x; reg[b].y = peer.y - reg[b].y;
                reg[b].z = peer.z - reg[b].z; reg[b].w = peer.w - reg[b].w;
            }
        }
    }

    extern __shared__ float smem[];
    constexpr int kWarpSize = 32;
    constexpr int kNWarps = THREADS / kWarpSize;
    int warp_id = tid / kWarpSize;
    int lane_id = tid % kWarpSize;
    constexpr int kChunksPerExchange = 4;
    constexpr int kNExchanges = NUM_B / kChunksPerExchange;

    #pragma unroll
    for (int ex = 0; ex < kNExchanges; ++ex) {
        __syncthreads();
        #pragma unroll
        for (int c = 0; c < kChunksPerExchange; ++c) {
            int smem_idx = c * THREADS + (warp_id * kWarpSize + (lane_id ^ warp_id));
            smem[smem_idx * 4 + 0] = reg[ex * kChunksPerExchange + c].x;
            smem[smem_idx * 4 + 1] = reg[ex * kChunksPerExchange + c].y;
            smem[smem_idx * 4 + 2] = reg[ex * kChunksPerExchange + c].z;
            smem[smem_idx * 4 + 3] = reg[ex * kChunksPerExchange + c].w;
        }
        __syncthreads();
        int row_t = tid % kNWarps;
        int col_t = tid / kNWarps;
        #pragma unroll
        for (int c = 0; c < kChunksPerExchange; ++c) {
            int smem_idx = c * THREADS + (row_t * kWarpSize + (col_t ^ row_t));
            reg[ex * kChunksPerExchange + c].x = smem[smem_idx * 4 + 0];
            reg[ex * kChunksPerExchange + c].y = smem[smem_idx * 4 + 1];
            reg[ex * kChunksPerExchange + c].z = smem[smem_idx * 4 + 2];
            reg[ex * kChunksPerExchange + c].w = smem[smem_idx * 4 + 3];
        }
    }

    #pragma unroll
    for (int stage = 0; stage < 3; ++stage) {
        int stride_thread = 1 << stage;
        #pragma unroll
        for (int b = 0; b < NUM_B; ++b) {
            float4 peer;
            peer.x = __shfl_xor_sync(0xffffffff, reg[b].x, stride_thread);
            peer.y = __shfl_xor_sync(0xffffffff, reg[b].y, stride_thread);
            peer.z = __shfl_xor_sync(0xffffffff, reg[b].z, stride_thread);
            peer.w = __shfl_xor_sync(0xffffffff, reg[b].w, stride_thread);

            if ((tid & stride_thread) == 0) {
                reg[b].x += peer.x; reg[b].y += peer.y;
                reg[b].z += peer.z; reg[b].w += peer.w;
            } else {
                reg[b].x = peer.x - reg[b].x; reg[b].y = peer.y - reg[b].y;
                reg[b].z = peer.z - reg[b].z; reg[b].w = peer.w - reg[b].w;
            }
        }
    }

    #pragma unroll
    for (int ex = 0; ex < kNExchanges; ++ex) {
        __syncthreads();
        int row_t = tid % kNWarps;
        int col_t = tid / kNWarps;
        #pragma unroll
        for (int c = 0; c < kChunksPerExchange; ++c) {
            int smem_idx = c * THREADS + (row_t * kWarpSize + (col_t ^ row_t));
            smem[smem_idx * 4 + 0] = reg[ex * kChunksPerExchange + c].x;
            smem[smem_idx * 4 + 1] = reg[ex * kChunksPerExchange + c].y;
            smem[smem_idx * 4 + 2] = reg[ex * kChunksPerExchange + c].z;
            smem[smem_idx * 4 + 3] = reg[ex * kChunksPerExchange + c].w;
        }
        __syncthreads();
        #pragma unroll
        for (int c = 0; c < kChunksPerExchange; ++c) {
            int smem_idx = c * THREADS + (warp_id * kWarpSize + (lane_id ^ warp_id));
            reg[ex * kChunksPerExchange + c].x = smem[smem_idx * 4 + 0];
            reg[ex * kChunksPerExchange + c].y = smem[smem_idx * 4 + 1];
            reg[ex * kChunksPerExchange + c].z = smem[smem_idx * 4 + 2];
            reg[ex * kChunksPerExchange + c].w = smem[smem_idx * 4 + 3];
        }
    }

    #pragma unroll
    for (int stage = 10; stage < N; ++stage) {
        int b_stride = 1 << (stage - 10);
        int b_block_size = b_stride << 1;
        #pragma unroll
        for (int b_pair = 0; b_pair < (NUM_B >> 1); ++b_pair) {
            int b_block_idx = b_pair / b_stride;
            int b_pair_offset = b_pair % b_stride;
            int b_i = b_block_idx * b_block_size + b_pair_offset;
            int b_j = b_i + b_stride;

            float4 a = reg[b_i];
            float4 b_val = reg[b_j];

            reg[b_i].x = a.x + b_val.x; reg[b_j].x = a.x - b_val.x;
            reg[b_i].y = a.y + b_val.y; reg[b_j].y = a.y - b_val.y;
            reg[b_i].z = a.z + b_val.z; reg[b_j].z = a.z - b_val.z;
            reg[b_i].w = a.w + b_val.w; reg[b_j].w = a.w - b_val.w;
        }
    }

    #pragma unroll
    for (int b = 0; b < NUM_B; ++b) {
        if (norm_factor != 1.0f) {
            reg[b].x *= norm_factor; reg[b].y *= norm_factor;
            reg[b].z *= norm_factor; reg[b].w *= norm_factor;
        }
        out4[b * THREADS + tid] = reg[b];
    }
}

template <int ROW_QUBITS>
__global__ void __launch_bounds__(256)
native_fp32_interblock_shuffle_kernel(
    const float* __restrict__ in_state,
    float* __restrict__ out_state,
    int num_col_qubits,
    float norm_factor
) {
    constexpr int ROWS = 1 << ROW_QUBITS;
    size_t cols = 1ULL << num_col_qubits;
    size_t batch_idx = blockIdx.y;

    constexpr int COLS_PER_BLOCK = 256 / ROWS;
    int col_in_block = threadIdx.x / ROWS;
    int row_idx      = threadIdx.x % ROWS;
    int col_idx      = blockIdx.x * COLS_PER_BLOCK + col_in_block;

    if (col_idx >= cols) return;

    const float* in_batch = in_state + batch_idx * (ROWS * cols);
    float* out_batch = out_state + batch_idx * (ROWS * cols);

    float val = in_batch[row_idx * cols + col_idx];

    #pragma unroll
    for (int stage = 0; stage < ROW_QUBITS; ++stage) {
        int stride = 1 << stage;
        float peer = __shfl_xor_sync(0xffffffff, val, stride);
        if ((row_idx & stride) == 0) {
            val = val + peer;
        } else {
            val = peer - val;
        }
    }

    if (norm_factor != 1.0f) val *= norm_factor;
    out_batch[row_idx * cols + col_idx] = val;
}

template <int TILE_COLS, int DUMMY>
__global__ void __launch_bounds__(256)
native_fp32_interblock_fwt_v2_kernel(
    const float* __restrict__ in_state,
    float* __restrict__ out_state,
    int num_col_qubits,
    int num_row_qubits,
    float norm_factor
) {
    constexpr int THREADS = 256;

    size_t batch_idx = blockIdx.y;
    size_t col_offset = (size_t)blockIdx.x * TILE_COLS;

    size_t rows = 1ULL << num_row_qubits;
    size_t cols = 1ULL << num_col_qubits;

    const float* in_batch = in_state + batch_idx * (rows * cols);
    float* out_batch = out_state + batch_idx * (rows * cols);

    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int total_elts = rows * TILE_COLS;

    int total_f4 = total_elts / 4;
    for (int idx = tid; idx < total_f4; idx += THREADS) {
        int linear = idx * 4;
        int r = linear / TILE_COLS;
        int c = linear % TILE_COLS;
        int global_base = r * cols + col_offset + c;

        float4 v;
        const float* ptr = in_batch + global_base;
        asm volatile(
            "ld.global.nc.v4.f32 {%0,%1,%2,%3}, [%4];"
            : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
            : "l"(ptr)
        );
        int smem_base = r * (TILE_COLS + 4) + c;
        smem[smem_base + 0] = v.x;
        smem[smem_base + 1] = v.y;
        smem[smem_base + 2] = v.z;
        smem[smem_base + 3] = v.w;
    }
    __syncthreads();

    constexpr int SMEM_PITCH = TILE_COLS + 4;

    for (int stage = 0; stage < num_row_qubits; ++stage) {
        int stride = 1 << stage;
        int block_size = stride << 1;
        int pairs_per_col = rows >> 1;
        int total_pairs = pairs_per_col * TILE_COLS;

        for (int idx = tid; idx < total_pairs; idx += THREADS) {
            int pair_idx = idx / TILE_COLS;
            int c = idx % TILE_COLS;

            int block_idx = pair_idx / stride;
            int pair_offset = pair_idx % stride;
            int r_i = block_idx * block_size + pair_offset;
            int r_j = r_i + stride;

            float a = smem[r_i * SMEM_PITCH + c];
            float b = smem[r_j * SMEM_PITCH + c];
            smem[r_i * SMEM_PITCH + c] = a + b;
            smem[r_j * SMEM_PITCH + c] = a - b;
        }
        __syncthreads();
    }

    for (int idx = tid; idx < total_f4; idx += THREADS) {
        int linear = idx * 4;
        int r = linear / TILE_COLS;
        int c = linear % TILE_COLS;
        int global_base = r * cols + col_offset + c;

        float4 v;
        int smem_base = r * SMEM_PITCH + c;
        v.x = smem[smem_base + 0];
        v.y = smem[smem_base + 1];
        v.z = smem[smem_base + 2];
        v.w = smem[smem_base + 3];

        if (norm_factor != 1.0f) {
            v.x *= norm_factor; v.y *= norm_factor;
            v.z *= norm_factor; v.w *= norm_factor;
        }

        float* out_ptr = out_batch + global_base;
        asm volatile(
            "st.global.v4.f32 [%0], {%1,%2,%3,%4};"
            :: "l"(out_ptr), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w)
        );
    }
}

void ImplicitHadamardNativeEngine::execute_implicit_hadamard_fp32(
    const float* d_A,
    float*       d_C,
    size_t m, size_t n, size_t k,
    float norm_factor,
    cudaStream_t stream,
    bool transpose_batch,
    size_t batch_rows
) {
    size_t state_len = n;
    int num_qubits = 0;
    while ((1ULL << num_qubits) < state_len) num_qubits++;

    // Fast Path for N <= 15 using Tri Dao style extreme kernel
    if (num_qubits <= 15 && num_qubits >= 7) {
        int smem_size = state_len * sizeof(float);
        switch (num_qubits) {
            case 7: native_fp32_extreme_fwt_kernel<7, 32, 1><<<m, 32, smem_size, stream>>>(d_A, d_C, norm_factor); return;
            case 8: native_fp32_extreme_fwt_kernel<8, 64, 1><<<m, 64, smem_size, stream>>>(d_A, d_C, norm_factor); return;
            case 9: native_fp32_extreme_fwt_kernel<9, 128, 1><<<m, 128, smem_size, stream>>>(d_A, d_C, norm_factor); return;
            case 10: native_fp32_extreme_fwt_kernel<10, 256, 1><<<m, 256, smem_size, stream>>>(d_A, d_C, norm_factor); return;
            case 11: native_fp32_extreme_fwt_kernel<11, 256, 2><<<m, 256, smem_size, stream>>>(d_A, d_C, norm_factor); return;
            case 12: native_fp32_extreme_fwt_kernel<12, 256, 4><<<m, 256, smem_size, stream>>>(d_A, d_C, norm_factor); return;
            case 13: native_fp32_extreme_fwt_kernel<13, 256, 8><<<m, 256, smem_size, stream>>>(d_A, d_C, norm_factor); return;
            case 14:
                cudaFuncSetAttribute(native_fp32_extreme_fwt_kernel<14, 256, 16>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
                native_fp32_extreme_fwt_kernel<14, 256, 16><<<m, 256, smem_size, stream>>>(d_A, d_C, norm_factor);
                return;
            case 15:
                smem_size = 4 * 256 * 4 * sizeof(float); // 16KB
                native_fp32_extreme_fwt_kernel_v2<15, 256, 32><<<m, 256, smem_size, stream>>>(d_A, d_C, norm_factor);
                return;
        }
    }

    int remaining_qubits = num_qubits;
    int current_qubit_offset = 0;
    int pass = 0;

    // TIER 2: Optimized 2-Pass (N >= 16)
    while (remaining_qubits > 0) {
        int MAX_SMEM_QUBITS = (pass == 0) ? 15 : 8;
        int qubits_this_pass = (remaining_qubits > MAX_SMEM_QUBITS) ? MAX_SMEM_QUBITS : remaining_qubits;

        if (pass == 0) {
            int chunk_size = 1 << qubits_this_pass;
            int num_chunks = m * (state_len / chunk_size);
            float pass_norm = (remaining_qubits == qubits_this_pass) ? norm_factor : 1.0f;
            int smem_size = chunk_size * sizeof(float);

            switch (qubits_this_pass) {
                case 7: native_fp32_extreme_fwt_kernel<7, 32, 1><<<num_chunks, 32, smem_size, stream>>>(d_A, d_C, pass_norm); break;
                case 8: native_fp32_extreme_fwt_kernel<8, 64, 1><<<num_chunks, 64, smem_size, stream>>>(d_A, d_C, pass_norm); break;
                case 9: native_fp32_extreme_fwt_kernel<9, 128, 1><<<num_chunks, 128, smem_size, stream>>>(d_A, d_C, pass_norm); break;
                case 10: native_fp32_extreme_fwt_kernel<10, 256, 1><<<num_chunks, 256, smem_size, stream>>>(d_A, d_C, pass_norm); break;
                case 11: native_fp32_extreme_fwt_kernel<11, 256, 2><<<num_chunks, 256, smem_size, stream>>>(d_A, d_C, pass_norm); break;
                case 12: native_fp32_extreme_fwt_kernel<12, 256, 4><<<num_chunks, 256, smem_size, stream>>>(d_A, d_C, pass_norm); break;
                case 13: native_fp32_extreme_fwt_kernel<13, 256, 8><<<num_chunks, 256, smem_size, stream>>>(d_A, d_C, pass_norm); break;
                case 14:
                    cudaFuncSetAttribute(native_fp32_extreme_fwt_kernel<14, 256, 16>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
                    native_fp32_extreme_fwt_kernel<14, 256, 16><<<num_chunks, 256, smem_size, stream>>>(d_A, d_C, pass_norm);
                    break;
                case 15:
                    smem_size = 4 * 256 * 4 * sizeof(float); // 16KB
                    native_fp32_extreme_fwt_kernel_v2<15, 256, 32><<<num_chunks, 256, smem_size, stream>>>(d_A, d_C, pass_norm);
                    break;
                default: {
                    int dyn_smem = (chunk_size + (chunk_size >> 5)) * sizeof(float);
                    int block_size = chunk_size >> 2;
                    if (block_size > 1024) block_size = 1024;
                    if (block_size < 32) block_size = 32;
                    native_fp32_opt_butterfly_kernel<<<num_chunks, block_size, dyn_smem, stream>>>(
                        d_A, d_C, chunk_size, num_chunks, pass_norm
                    );
                    break;
                }
            }
        } else {
            int cols = 1 << current_qubit_offset;
            int rows = 1 << qubits_this_pass;
            int batches = m * (state_len / (rows * cols));
            float pass_norm = (remaining_qubits == qubits_this_pass) ? norm_factor : 1.0f;

            if (qubits_this_pass <= 5) {
                int block_size = 256;
                int COLS_PER_BLOCK = block_size / rows;
                int grid_x = (cols + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK;
                dim3 grid(grid_x, batches);
                switch(qubits_this_pass) {
                    case 1: native_fp32_interblock_shuffle_kernel<1><<<grid, block_size, 0, stream>>>(d_C, d_C, current_qubit_offset, pass_norm); break;
                    case 2: native_fp32_interblock_shuffle_kernel<2><<<grid, block_size, 0, stream>>>(d_C, d_C, current_qubit_offset, pass_norm); break;
                    case 3: native_fp32_interblock_shuffle_kernel<3><<<grid, block_size, 0, stream>>>(d_C, d_C, current_qubit_offset, pass_norm); break;
                    case 4: native_fp32_interblock_shuffle_kernel<4><<<grid, block_size, 0, stream>>>(d_C, d_C, current_qubit_offset, pass_norm); break;
                    case 5: native_fp32_interblock_shuffle_kernel<5><<<grid, block_size, 0, stream>>>(d_C, d_C, current_qubit_offset, pass_norm); break;
                }
            } else {
                constexpr int TILE_COLS = 64;
                int SMEM_PITCH = TILE_COLS + 4;
                int smem_size = rows * SMEM_PITCH * sizeof(float);
                int block_size = 256;

                cudaFuncSetAttribute(native_fp32_interblock_fwt_v2_kernel<TILE_COLS, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

                dim3 grid(cols / TILE_COLS, batches);
                native_fp32_interblock_fwt_v2_kernel<TILE_COLS, 0><<<grid, block_size, smem_size, stream>>>(
                    d_C, d_C, current_qubit_offset, qubits_this_pass, pass_norm
                );
            }
        }

        remaining_qubits -= qubits_this_pass;
        current_qubit_offset += qubits_this_pass;
        pass++;
    }
}

// ============================================================================
// FP64 Extreme Intra-Block FWT Kernel (N <= 14)
// Uses exactly 128 registers per thread (64 doubles) and 32KB SMEM.
// ============================================================================
template<int N, int THREADS>
__global__ void __launch_bounds__(THREADS) native_fp64_extreme_fwt_kernel(
    const double* __restrict__ in_state,
    double* __restrict__ out_state,
    double norm_factor
) {
    size_t chunk_idx = blockIdx.x;
    int tid = threadIdx.x;
    constexpr int CHUNK_LEN = 1 << N;
    constexpr int NUM_B = CHUNK_LEN / THREADS;

    // Each block processes one CHUNK_LEN size array
    const double* in_chunk = in_state + chunk_idx * CHUNK_LEN;
    double* out_chunk = out_state + chunk_idx * CHUNK_LEN;

    double reg[64]; // Max 64

    #pragma unroll
    for (int b = 0; b < NUM_B; ++b) {
        reg[b] = in_chunk[b * THREADS + tid];
    }

    // Stages 0 to 4: Warp Shuffles (Strides 1, 2, 4, 8, 16)
    int max_warp_stage = (N < 5) ? N : 5;
    #pragma unroll
    for (int stage = 0; stage < max_warp_stage; ++stage) {
        int stride = 1 << stage;
        #pragma unroll
        for (int b = 0; b < NUM_B; ++b) {
            double peer = __shfl_xor_sync(0xffffffff, reg[b], stride);
            if ((tid & stride) == 0) {
                reg[b] = reg[b] + peer;
            } else {
                reg[b] = peer - reg[b];
            }
        }
    }

    // Stages 5 to 7: Cross-Warp via SMEM (Strides 32, 64, 128)
    if (N > 5) {
        extern __shared__ double smem_d[];
        int max_smem_stage = (N < 8) ? N : 8;

        // Process in batches of 16 'b's to limit SMEM to 32KB (16 * 256 * 8 bytes)
        for (int b_start = 0; b_start < NUM_B; b_start += 16) {
            int b_count = ((NUM_B - b_start) < 16) ? (NUM_B - b_start) : 16;

            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                if (i < b_count) smem_d[i * THREADS + tid] = reg[b_start + i];
            }
            __syncthreads();

            for (int stage = 5; stage < max_smem_stage; ++stage) {
                int stride = 1 << stage;
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    if (i < b_count) {
                        double my_val = smem_d[i * THREADS + tid];
                        double peer_val = smem_d[i * THREADS + (tid ^ stride)];
                        reg[b_start + i] = ((tid & stride) == 0) ? (my_val + peer_val) : (peer_val - my_val);
                    }
                }
                __syncthreads();

                if (stage < max_smem_stage - 1) {
                    #pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        if (i < b_count) smem_d[i * THREADS + tid] = reg[b_start + i];
                    }
                    __syncthreads();
                }
            }
        }
    }

    // Stages 8 to 13: Intra-Thread (Strides 256, 512, 1024, 2048, 4096, 8192)
    if (N > 8) {
        #pragma unroll
        for (int stage = 8; stage < N; ++stage) {
            int b_stride = 1 << (stage - 8);
            #pragma unroll
            for (int b = 0; b < 64; ++b) {
                if (b < NUM_B && (b & b_stride) == 0) {
                    double a = reg[b];
                    double b_val = reg[b | b_stride];
                    reg[b] = a + b_val;
                    reg[b | b_stride] = a - b_val;
                }
            }
        }
    }

    #pragma unroll
    for (int b = 0; b < NUM_B; ++b) {
        if (norm_factor != 1.0) reg[b] *= norm_factor;
        out_chunk[b * THREADS + tid] = reg[b];
    }
}

// ============================================================================
// FP64 Inter-Block Kernel (N > 14)
// Processes remaining qubits in passes of up to 8 qubits.
// ============================================================================
template<int TILE_COLS>
__global__ void __launch_bounds__(256) native_fp64_interblock_fwt_kernel(
    const double* __restrict__ in_state,
    double* __restrict__ out_state,
    int pass_qubits,
    int stride,
    double norm_factor
) {
    size_t block_size = (size_t)stride << pass_qubits;
    size_t chunk_size = 1ULL << pass_qubits;

    size_t tile_row_idx = blockIdx.x;
    size_t tile_col_idx = (size_t)blockIdx.y * TILE_COLS;

    size_t global_block_idx = tile_row_idx / (chunk_size >> 1);
    size_t element_in_chunk = tile_row_idx % (chunk_size >> 1);

    size_t row_offset = global_block_idx * block_size + element_in_chunk;

    extern __shared__ double smem_d[];
    int SMEM_PITCH = TILE_COLS + 2;

    int tid = threadIdx.x;
    int r = tid / TILE_COLS;
    int c = tid % TILE_COLS;

    size_t global_r = row_offset + r * (chunk_size >> 1);
    size_t global_c = tile_col_idx + c;

    if (c < TILE_COLS && global_c < stride) {
        double v0 = in_state[global_r];
        double v1 = in_state[global_r + (chunk_size >> 1)];

        smem_d[r * SMEM_PITCH + c] = v0;
        smem_d[(r + 2) * SMEM_PITCH + c] = v1;
    }
    __syncthreads();

    if (tid < chunk_size) {
        int my_r = tid;
        for (int stage = 0; stage < pass_qubits; ++stage) {
            int s = 1 << stage;
            int b = my_r / s;
            int offset = my_r % s;
            int i = b * (s << 1) + offset;
            int j = i + s;

            #pragma unroll
            for (int cc = 0; cc < TILE_COLS; ++cc) {
                double a = smem_d[i * SMEM_PITCH + cc];
                double b_val = smem_d[j * SMEM_PITCH + cc];
                smem_d[i * SMEM_PITCH + cc] = a + b_val;
                smem_d[j * SMEM_PITCH + cc] = a - b_val;
            }
        }
    }
    __syncthreads();

    if (c < TILE_COLS && global_c < stride) {
        double v0 = smem_d[r * SMEM_PITCH + c];
        double v1 = smem_d[(r + 2) * SMEM_PITCH + c];

        if (norm_factor != 1.0) {
            v0 *= norm_factor;
            v1 *= norm_factor;
        }

        out_state[global_r] = v0;
        out_state[global_r + (chunk_size >> 1)] = v1;
    }
}

// Dispatch for FP64
void ImplicitHadamardNativeEngine::execute_implicit_hadamard_fp64(
    const double* d_A,
    double*       d_C,
    size_t m, size_t n, size_t k,
    double norm_factor,
    cudaStream_t stream,
    bool transpose_batch,
    size_t batch_rows
) {
    size_t state_len = n;
    int num_qubits = 0;
    while ((1ULL << num_qubits) < state_len) num_qubits++;

    // TIER 1: Fast Path for N <= 14
    if (num_qubits <= 14 && num_qubits >= 6) {
        int smem_size = 16 * 256 * sizeof(double); // 32 KB Max
        switch (num_qubits) {
            case 6: native_fp64_extreme_fwt_kernel<6, 64><<<m, 64, 16 * 64 * (int)sizeof(double), stream>>>(d_A, d_C, norm_factor); return;
            case 7: native_fp64_extreme_fwt_kernel<7, 128><<<m, 128, 16 * 128 * (int)sizeof(double), stream>>>(d_A, d_C, norm_factor); return;
            case 8: native_fp64_extreme_fwt_kernel<8, 256><<<m, 256, smem_size, stream>>>(d_A, d_C, norm_factor); return;
            case 9: native_fp64_extreme_fwt_kernel<9, 256><<<m, 256, smem_size, stream>>>(d_A, d_C, norm_factor); return;
            case 10: native_fp64_extreme_fwt_kernel<10, 256><<<m, 256, smem_size, stream>>>(d_A, d_C, norm_factor); return;
            case 11: native_fp64_extreme_fwt_kernel<11, 256><<<m, 256, smem_size, stream>>>(d_A, d_C, norm_factor); return;
            case 12: native_fp64_extreme_fwt_kernel<12, 256><<<m, 256, smem_size, stream>>>(d_A, d_C, norm_factor); return;
            case 13: native_fp64_extreme_fwt_kernel<13, 256><<<m, 256, smem_size, stream>>>(d_A, d_C, norm_factor); return;
            case 14: native_fp64_extreme_fwt_kernel<14, 256><<<m, 256, smem_size, stream>>>(d_A, d_C, norm_factor); return;
        }
    }

    int remaining_qubits = num_qubits;
    int current_qubit_offset = 0;
    int pass = 0;

    // TIER 2: Optimized Multi-Pass (N >= 15)
    while (remaining_qubits > 0) {
        int MAX_SMEM_QUBITS = (pass == 0) ? 14 : 8;
        int qubits_this_pass = (remaining_qubits > MAX_SMEM_QUBITS) ? MAX_SMEM_QUBITS : remaining_qubits;

        if (pass == 0) {
            int chunk_size = 1 << qubits_this_pass;
            int num_chunks = m * (state_len / chunk_size);
            double pass_norm = (remaining_qubits == qubits_this_pass) ? norm_factor : 1.0;
            int smem_size = 16 * 256 * sizeof(double); // 32 KB Max

            switch (qubits_this_pass) {
                case 6: native_fp64_extreme_fwt_kernel<6, 64><<<num_chunks, 64, 16 * 64 * (int)sizeof(double), stream>>>(d_A, d_C, pass_norm); break;
                case 7: native_fp64_extreme_fwt_kernel<7, 128><<<num_chunks, 128, 16 * 128 * (int)sizeof(double), stream>>>(d_A, d_C, pass_norm); break;
                case 8: native_fp64_extreme_fwt_kernel<8, 256><<<num_chunks, 256, smem_size, stream>>>(d_A, d_C, pass_norm); break;
                case 9: native_fp64_extreme_fwt_kernel<9, 256><<<num_chunks, 256, smem_size, stream>>>(d_A, d_C, pass_norm); break;
                case 10: native_fp64_extreme_fwt_kernel<10, 256><<<num_chunks, 256, smem_size, stream>>>(d_A, d_C, pass_norm); break;
                case 11: native_fp64_extreme_fwt_kernel<11, 256><<<num_chunks, 256, smem_size, stream>>>(d_A, d_C, pass_norm); break;
                case 12: native_fp64_extreme_fwt_kernel<12, 256><<<num_chunks, 256, smem_size, stream>>>(d_A, d_C, pass_norm); break;
                case 13: native_fp64_extreme_fwt_kernel<13, 256><<<num_chunks, 256, smem_size, stream>>>(d_A, d_C, pass_norm); break;
                case 14: native_fp64_extreme_fwt_kernel<14, 256><<<num_chunks, 256, smem_size, stream>>>(d_A, d_C, pass_norm); break;
            }
        } else {
            int stride = 1 << current_qubit_offset;
            int chunk_size = 1 << qubits_this_pass;
            double pass_norm = (remaining_qubits == qubits_this_pass) ? norm_factor : 1.0;

            const int TILE_COLS = 64;
            int num_tiles_row = m * (state_len / chunk_size);
            int num_tiles_col = (stride + TILE_COLS - 1) / TILE_COLS;

            dim3 grid(num_tiles_row, num_tiles_col);
            int smem_size = chunk_size * (TILE_COLS + 2) * sizeof(double);

            const double* in_ptr = (pass == 1) ? d_C : d_C;
            native_fp64_interblock_fwt_kernel<TILE_COLS><<<grid, 256, smem_size, stream>>>(
                in_ptr, d_C, qubits_this_pass, stride, pass_norm
            );
        }

        remaining_qubits -= qubits_this_pass;
        current_qubit_offset += qubits_this_pass;
        pass++;
    }
}

} // namespace native
} // namespace qdp
