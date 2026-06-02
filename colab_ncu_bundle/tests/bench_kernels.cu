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
    size_t num_samples = 128; // Reduced batch size for larger qubits to fit memory
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
    
    // Fill data with some dummy values
    std::vector<double> h_data(num_samples * data_len, 0.5);
    cudaMemcpy(data_batch_d, h_data.data(), num_samples * data_len * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(state_baseline_d, 0, num_samples * state_len * sizeof(cuDoubleComplex));
    cudaMemset(state_tc_d, 0, num_samples * state_len * sizeof(cuDoubleComplex));

    // 1. Run Baseline
    launch_iqp_encode_batch(data_batch_d, state_baseline_d, num_samples, state_len, num_qubits, data_len, enable_zz, 0);
    cudaDeviceSynchronize();

    // 2. Run TC Version & Profile
    launch_iqp_encode_tc(data_batch_d, state_tc_d, num_samples, state_len, num_qubits, enable_zz, 0);
    cudaDeviceSynchronize();
    
    auto start = std::chrono::high_resolution_clock::now();
    launch_iqp_encode_tc(data_batch_d, state_tc_d, num_samples, state_len, num_qubits, enable_zz, 0);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

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

    std::cout << "Bench execution complete. Duration: " << duration << " us" << std::endl;
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
