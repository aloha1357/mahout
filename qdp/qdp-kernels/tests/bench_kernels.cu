#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

extern "C" int launch_iqp_encode_tc(
    const double* data_batch_d,
    void* state_batch_d,
    size_t num_samples,
    size_t state_len,
    unsigned int num_qubits,
    int enable_zz,
    cudaStream_t stream
);

int main() {
    size_t num_samples = 1024;
    unsigned int num_qubits = 10;
    size_t state_len = 1ULL << num_qubits;
    unsigned int data_len = num_qubits;
    int enable_zz = 0;

    double* data_batch_d;
    void* state_batch_d;
    cudaMalloc(&data_batch_d, num_samples * data_len * sizeof(double));
    cudaMalloc(&state_batch_d, num_samples * state_len * sizeof(double) * 2);
    cudaMemset(data_batch_d, 0, num_samples * data_len * sizeof(double));
    cudaMemset(state_batch_d, 0, num_samples * state_len * sizeof(double) * 2);

    // Warmup
    launch_iqp_encode_tc(data_batch_d, state_batch_d, num_samples, state_len, num_qubits, enable_zz, 0);
    cudaDeviceSynchronize();

    auto start = std::chrono::high_resolution_clock::now();
    launch_iqp_encode_tc(data_batch_d, state_batch_d, num_samples, state_len, num_qubits, enable_zz, 0);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::cout << "Bench execution complete. Duration: " << duration << " us" << std::endl;
    std::cerr << "[  FAILED  ] KernelBench.IqpFwtBaseline (No baseline available)" << std::endl;

    cudaFree(data_batch_d);
    cudaFree(state_batch_d);
    return 1;
}
