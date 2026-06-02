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
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cuComplex.h>

extern "C" int launch_iqp_encode_batch(
    const double* data_batch_d,
    void* state_batch_d,
    size_t num_samples,
    size_t state_len,
    unsigned int num_qubits,
    unsigned int data_len,
    int enable_zz,
    cudaStream_t stream
);

extern "C" int launch_iqp_encode_tc(
    const double* data_batch_d,
    void* state_batch_d,
    size_t num_samples,
    size_t state_len,
    unsigned int num_qubits,
    int enable_zz,
    cudaStream_t stream
);

int main(int argc, char* argv[]) {
    size_t num_samples = 128;
    unsigned int num_qubits = 14;
    if (argc > 1) {
        num_qubits = std::atoi(argv[1]);
    }
    if (argc > 2) {
        num_samples = std::strtoull(argv[2], nullptr, 10);
    }
    if (const char* env_samples = std::getenv("IQP_NUM_SAMPLES")) {
        num_samples = std::strtoull(env_samples, nullptr, 10);
    }
    size_t state_len = 1ULL << num_qubits;
    unsigned int data_len = num_qubits;
    int enable_zz = 0;

    double* data_batch_d;
    cuDoubleComplex *state_baseline_d, *state_tc_d;
    cudaMalloc(&data_batch_d, num_samples * data_len * sizeof(double));
    cudaMalloc(&state_baseline_d, num_samples * state_len * sizeof(cuDoubleComplex));
    cudaMalloc(&state_tc_d, num_samples * state_len * sizeof(cuDoubleComplex));

    std::vector<double> h_data(num_samples * data_len, 0.5);
    cudaMemcpy(data_batch_d, h_data.data(), num_samples * data_len * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(state_baseline_d, 0, num_samples * state_len * sizeof(cuDoubleComplex));
    cudaMemset(state_tc_d, 0, num_samples * state_len * sizeof(cuDoubleComplex));

    // 1. Run Baseline & Profile
    launch_iqp_encode_batch(data_batch_d, state_baseline_d, num_samples, state_len, num_qubits, data_len, enable_zz, 0);
    cudaDeviceSynchronize();

    auto start_baseline = std::chrono::high_resolution_clock::now();
    launch_iqp_encode_batch(data_batch_d, state_baseline_d, num_samples, state_len, num_qubits, data_len, enable_zz, 0);
    cudaDeviceSynchronize();
    auto end_baseline = std::chrono::high_resolution_clock::now();
    auto duration_baseline = std::chrono::duration_cast<std::chrono::microseconds>(end_baseline - start_baseline).count();

    // 2. Run TC Version & Profile
    launch_iqp_encode_tc(data_batch_d, state_tc_d, num_samples, state_len, num_qubits, enable_zz, 0);
    cudaDeviceSynchronize();

    auto start_tc = std::chrono::high_resolution_clock::now();
    launch_iqp_encode_tc(data_batch_d, state_tc_d, num_samples, state_len, num_qubits, enable_zz, 0);
    cudaDeviceSynchronize();
    auto end_tc = std::chrono::high_resolution_clock::now();
    auto duration_tc = std::chrono::duration_cast<std::chrono::microseconds>(end_tc - start_tc).count();

    // 3. Verify Correctness
    std::vector<cuDoubleComplex> h_state_baseline(num_samples * state_len);
    std::vector<cuDoubleComplex> h_state_tc(num_samples * state_len);
    cudaMemcpy(h_state_baseline.data(), state_baseline_d, num_samples * state_len * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_state_tc.data(), state_tc_d, num_samples * state_len * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    double max_err = 0.0;
    for(size_t i = 0; i < num_samples * state_len; i++) {
        double err_r = std::abs(cuCreal(h_state_baseline[i]) - cuCreal(h_state_tc[i]));
        double err_i = std::abs(cuCimag(h_state_baseline[i]) - cuCimag(h_state_tc[i]));
        if(err_r > max_err) max_err = err_r;
        if(err_i > max_err) max_err = err_i;
    }

    std::cout << "N=" << num_qubits << ", Batch=" << num_samples << std::endl;
    std::cout << "Baseline Duration: " << duration_baseline << " us" << std::endl;
    std::cout << "TC Path Duration:  " << duration_tc << " us" << std::endl;
    if (duration_tc > 0) {
        std::cout << "Speedup:           " << (double)duration_baseline / duration_tc << "x" << std::endl;
    }
    std::cout << "Correctness Verification - Max Absolute Error: " << max_err << std::endl;

    if (max_err < 1e-6) {
        std::cout << "[  PASSED  ] KernelBench.Correctness" << std::endl;
    } else {
        std::cout << "[  FAILED  ] KernelBench.Correctness" << std::endl;
    }

    cudaFree(data_batch_d);
    cudaFree(state_baseline_d);
    cudaFree(state_tc_d);
    return 0;
}
