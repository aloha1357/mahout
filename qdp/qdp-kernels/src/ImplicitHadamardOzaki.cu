#include "ImplicitHadamardOzaki.h"
#include <iostream>
#include <cmath>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

namespace implicit_ozaki_kernels {

__device__ __forceinline__ void mma_m16n8k32_s8(
    int32_t* d, const uint32_t* a, const uint32_t* b, const int32_t* c) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3])
    );
}


__device__ __forceinline__ int get_A8_offset(int r, int c, int m, int k) {
    int tile_m = r / 128;
    int tile_k = c / 32;
    int num_tiles_k = (k + 31) / 32;
    int in_tile_r = r % 128;
    int in_tile_c = c % 32;
    return (tile_m * num_tiles_k + tile_k) * 4096 + in_tile_r * 32 + in_tile_c;
}

__global__ void precompute_modulo_kernel_p26_implicit(const double* __restrict__ s, int8_t* __restrict__ d, int r, int c, int m, double sh, double sl) {
    int local_k = threadIdx.x; int local_m = threadIdx.y;
    int tile_k = blockIdx.x; int tile_m = blockIdx.y;
    const int pr[7] = {127, 113, 109, 107, 103, 101, 97};
    size_t padded_size = (size_t)((r + 127) / 128) * ((c + 31) / 32) * 4096;
    int num_tiles_k = (c + 31) / 32;
    int m_idx = (tile_m / 4) * 128 + (tile_m % 4) * 32 + local_m;
    int k_idx = tile_k * 32 + local_k;
    if (m_idx < r && k_idx < c) {
        double v = s[(size_t)m_idx * c + k_idx];
        int32_t iv = (v == 0.0) ? 0 : __double2int_rn(v * sh);
        size_t out_off = (size_t)( (m_idx / 128) * num_tiles_k + tile_k ) * 4096 + (m_idx % 128) * 32 + local_k;
        for (int p = 0; p < 7; p++) { int32_t rem = iv % pr[p]; if (rem < 0) rem += pr[p]; d[p * padded_size + out_off] = (int8_t)rem; }
    }
}

__global__ void implicit_hadamard_ozaki_persistent_kernel_implicit(
    const int8_t* __restrict__ A8_h,
    double* __restrict__ C, int m, int n, int k, double inv, int* __restrict__ d_work_queue,
    double norm_factor) {
    
    extern __shared__ int8_t shared_mem[];
    int8_t* sA8 = &shared_mem[0];
    int8_t* sB8 = &sA8[28672];

    int warp_id = threadIdx.x / 32, lane_id = threadIdx.x % 32;
    __shared__ int tile_idx;
    int total_tiles_m = (m + 63) / 64, total_tiles_n = (n + 63) / 64, total_tiles = total_tiles_m * total_tiles_n;
    
    const int pr[7] = {127, 113, 109, 107, 103, 101, 97};
    const uint64_t M = 168897325606883ULL;
    const uint64_t f[7] = {
        147618922380819ULL, 112099994871825ULL, 134807957135769ULL,
        34726552928518ULL, 96747011755399ULL, 130435558389474ULL,
        19153304965729ULL
    };

    __shared__ int8_t h_pos[7];
    __shared__ int8_t h_neg[7];

    if (threadIdx.x == 0) {
        int32_t iv_pos = 1;
        int32_t iv_neg = -1;
        for(int p=0; p<7; p++) {
            int rem_pos = iv_pos % pr[p]; if(rem_pos < 0) rem_pos += pr[p];
            int rem_neg = iv_neg % pr[p]; if(rem_neg < 0) rem_neg += pr[p];
            h_pos[p] = rem_pos;
            h_neg[p] = rem_neg;
        }
    }

    while (true) {
        if (threadIdx.x == 0) tile_idx = atomicAdd(d_work_queue, 1);
        __syncthreads();
        int ct = tile_idx; if (ct >= total_tiles) break;
        int tile_m = (ct % total_tiles_m) * 64, tile_n = (ct / total_tiles_m) * 64;

        uint64_t final_acc[8][4]; 
        for(int i=0; i<8; i++) for(int j=0; j<4; j++) final_acc[i][j] = 0;

        int32_t prime_acc[7][8][4];
        for(int p=0; p<7; p++) for(int i=0; i<8; i++) for(int j=0; j<4; j++) prime_acc[p][i][j] = 0;

        for (int kk = 0; kk < k; kk += 32) {
            int b_idx = (kk / 32) % 2;
            int k_size = min(32, k - kk);
            size_t padded_mk = (size_t)((m + 127) / 128) * ((k + 31) / 32) * 4096;

            for (int p = 0; p < 7; ++p) {
                const int8_t* Ap = A8_h + p * padded_mk;
                
                for (int i = threadIdx.x; i < 64 * k_size; i += 256) {
                    int r = i / k_size, c = i % k_size;
                    int8_t val = (tile_m + r < m && kk + c < k) ? Ap[get_A8_offset(tile_m + r, kk + c, m, k)] : 0;
                    sA8[p * 4096 + b_idx * 2048 + r * 32 + c] = val;
                }
                for (int i = threadIdx.x; i < k_size * 64; i += 256) {
                    int r = i / 64, c = i % 64;
                    int8_t val = 0;
                    if (kk + r < k && tile_n + c < n) {
                        int parity = __popcll((kk + r) & (tile_n + c)) & 1;
                        val = (parity == 0) ? h_pos[p] : h_neg[p];
                    }
                    sB8[p * 4096 + b_idx * 2048 + r * 64 + c] = val;
                }
            }
            __syncthreads();

            if (warp_id < 4) {
                int wr = warp_id / 2;
                int wc = warp_id % 2;
                
                for (int p = 0; p < 7; ++p) {
                    int8_t* pA = &sA8[p * 4096 + b_idx * 2048];
                    int8_t* pB = &sB8[p * 4096 + b_idx * 2048];
                    
                    #pragma unroll
                    for (int k_s = 0; k_s < 32; k_s += 32) {
                        #pragma unroll
                        for (int mt = 0; mt < 2; ++mt) {
                            #pragma unroll
                            for (int nt = 0; nt < 4; ++nt) {
                                uint32_t ra[4], rb[2];
                                
                                int r_a_0 = lane_id / 4;
                                int r_a_8 = r_a_0 + 8;
                                int k_base_a = (lane_id % 4) * 8;
                                int r0 = wr * 32 + mt * 16 + r_a_0;
                                int r8 = wr * 32 + mt * 16 + r_a_8;
                                int c_a = k_s + k_base_a;

                                uint32_t final_va0 = 0, final_va1 = 0, final_va2 = 0, final_va3 = 0;
                                final_va0 |= ((uint32_t)(uint8_t)pA[r0 * 32 + c_a + 0]) << 0;
                                final_va0 |= ((uint32_t)(uint8_t)pA[r0 * 32 + c_a + 1]) << 8;
                                final_va0 |= ((uint32_t)(uint8_t)pA[r0 * 32 + c_a + 2]) << 16;
                                final_va0 |= ((uint32_t)(uint8_t)pA[r0 * 32 + c_a + 3]) << 24;

                                final_va1 |= ((uint32_t)(uint8_t)pA[r8 * 32 + c_a + 0]) << 0;
                                final_va1 |= ((uint32_t)(uint8_t)pA[r8 * 32 + c_a + 1]) << 8;
                                final_va1 |= ((uint32_t)(uint8_t)pA[r8 * 32 + c_a + 2]) << 16;
                                final_va1 |= ((uint32_t)(uint8_t)pA[r8 * 32 + c_a + 3]) << 24;

                                final_va2 |= ((uint32_t)(uint8_t)pA[r0 * 32 + c_a + 4]) << 0;
                                final_va2 |= ((uint32_t)(uint8_t)pA[r0 * 32 + c_a + 5]) << 8;
                                final_va2 |= ((uint32_t)(uint8_t)pA[r0 * 32 + c_a + 6]) << 16;
                                final_va2 |= ((uint32_t)(uint8_t)pA[r0 * 32 + c_a + 7]) << 24;

                                final_va3 |= ((uint32_t)(uint8_t)pA[r8 * 32 + c_a + 4]) << 0;
                                final_va3 |= ((uint32_t)(uint8_t)pA[r8 * 32 + c_a + 5]) << 8;
                                final_va3 |= ((uint32_t)(uint8_t)pA[r8 * 32 + c_a + 6]) << 16;
                                final_va3 |= ((uint32_t)(uint8_t)pA[r8 * 32 + c_a + 7]) << 24;

                                ra[0] = final_va0; ra[1] = final_va1; ra[2] = final_va2; ra[3] = final_va3;
                                
                                int cb_base = wc * 32 + nt * 8;
                                int n_col = lane_id / 4;
                                int k_base = (lane_id % 4) * 8;
                                int c0 = cb_base + n_col;

                                uint32_t final_vb0 = 0, final_vb1 = 0;
                                final_vb0 |= ((uint32_t)(uint8_t)pB[(k_s + k_base + 0) * 64 + c0]) << 0;
                                final_vb0 |= ((uint32_t)(uint8_t)pB[(k_s + k_base + 1) * 64 + c0]) << 8;
                                final_vb0 |= ((uint32_t)(uint8_t)pB[(k_s + k_base + 2) * 64 + c0]) << 16;
                                final_vb0 |= ((uint32_t)(uint8_t)pB[(k_s + k_base + 3) * 64 + c0]) << 24;

                                final_vb1 |= ((uint32_t)(uint8_t)pB[(k_s + k_base + 4) * 64 + c0]) << 0;
                                final_vb1 |= ((uint32_t)(uint8_t)pB[(k_s + k_base + 5) * 64 + c0]) << 8;
                                final_vb1 |= ((uint32_t)(uint8_t)pB[(k_s + k_base + 6) * 64 + c0]) << 16;
                                final_vb1 |= ((uint32_t)(uint8_t)pB[(k_s + k_base + 7) * 64 + c0]) << 24;
                                rb[0] = final_vb0; rb[1] = final_vb1;

                                mma_m16n8k32_s8(prime_acc[p][mt * 4 + nt], ra, rb, prime_acc[p][mt * 4 + nt]);
                            }
                        }
                    }
                }
            }
            __syncthreads();
        }

        if (warp_id < 4) {
            for (int p = 0; p < 7; ++p) {
                for(int i=0; i<8; i++) {
                    for(int j=0; j<4; j++) {
                        uint32_t rem = (prime_acc[p][i][j] % pr[p] + pr[p]) % pr[p];
                        final_acc[i][j] += (uint64_t)rem * f[p];
                    }
                }
            }
        }

        if (warp_id < 4) {
            int wr = warp_id / 2, wc = warp_id % 2;
            for (int i = 0; i < 8; i++) {
                int mt = i / 4, nt = i % 4;
                int r_base = tile_m + wr * 32 + mt * 16 + (lane_id / 4);
                int c_base = tile_n + wc * 32 + nt * 8 + (lane_id % 4) * 2;
                
                uint64_t cv0 = final_acc[i][0];
                uint64_t cv1 = final_acc[i][1];
                uint64_t cv2 = final_acc[i][2];
                uint64_t cv3 = final_acc[i][3];

                auto store_res = [&](int r, int c, uint64_t cv) {
                    if (r < m && c < n) {
                        double fv = (double)(cv % M);
                        if (cv % M > M / 2) fv -= (double)M;
                        atomicAdd(&C[(size_t)r * n + c], fv * norm_factor * inv);
                    }
                };

                store_res(r_base, c_base, cv0);
                store_res(r_base, c_base + 1, cv1);
                store_res(r_base + 8, c_base, cv2);
                store_res(r_base + 8, c_base + 1, cv3);
            }
        }
    }
}

} // namespace implicit_ozaki_kernels

namespace ozaki {

void ImplicitHadamardOzakiEngine::execute_implicit_hadamard(const double* d_A, double* d_C, int m, int n, int k, double norm_factor) {
    size_t padded_mk = (size_t)((m + 127) / 128) * ((k + 31) / 32) * 4096;
    
    int8_t *dA8_h = nullptr;
    int *d_queue = nullptr;
    
    cudaMalloc(&dA8_h, 7ULL * padded_mk);
    cudaMalloc(&d_queue, sizeof(int));
    cudaMemset(d_queue, 0, sizeof(int));
    cudaMemset(d_C, 0, (size_t)m * n * 8);

    cudaStream_t st; cudaStreamCreate(&st);
    
    double scale_A = pow(2.0, 30.0);
    double inv_A = pow(2.0, -30.0);

    dim3 pre_block(32, 32);
    dim3 pre_grid_A((k + 31) / 32, ((m + 127) / 128) * 4);
    implicit_ozaki_kernels::precompute_modulo_kernel_p26_implicit<<<pre_grid_A, pre_block, 0, st>>>(d_A, dA8_h, m, k, 0, scale_A, 0.0);
    
    int num_sms; cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    
    cudaFuncSetAttribute(implicit_ozaki_kernels::implicit_hadamard_ozaki_persistent_kernel_implicit, cudaFuncAttributeMaxDynamicSharedMemorySize, 60*1024);
    implicit_ozaki_kernels::implicit_hadamard_ozaki_persistent_kernel_implicit<<<num_sms, 256, 58*1024, st>>>(
        dA8_h, d_C, m, n, k, inv_A, d_queue, norm_factor);
        
    cudaStreamSynchronize(st); 
    cudaStreamDestroy(st);

    cudaFree(dA8_h);
    cudaFree(d_queue);
}

} // namespace ozaki
