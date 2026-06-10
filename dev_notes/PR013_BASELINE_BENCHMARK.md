# PR2-6 Rework: Baseline Benchmark

This document records the performance of the original, matrix-based approach to the Fast Walsh-Hadamard Transform (FWT) before the PR2-6 optimizations (Implicit FWT, Shared Memory Optimization, TensorCore Ozaki scheme, etc.).

## Setup
- **Environment**: WSL2 + PyTorch Eager (Python 3.12, no `torch.compile` due to Dynamo limitations)
- **Batch Size**: 128
- **Data Type**: FP64 (`torch.float64`)
- **Device**: CUDA
- **Implementation**: Naive dense Hadamard matrix multiplication via `torch.kron`.

## Latency Results

| Qubits (N) | Matrix Size (2^N) | PyTorch Eager Baseline | Status |
| :--------- | :---------------- | :----------------------- | :----- |
| 10         | 1024              | 1.26 ms                  | OK     |
| 12         | 4096              | 24.39 ms                 | OK     |
| 14         | 16384             | 339.42 ms                | OK     |

## Observations
- The legacy approach constructs a massive dense matrix for the Hadamard transform in global memory.
- It struggles significantly at `N=14` (batch size 128) taking ~339.42 ms, which highlights the critical DRAM bottleneck.
- For `N>14`, the `torch.kron` dense generation scales horribly with $O(2^{2N})$ memory, causing OOMs on typical consumer GPUs (8GB).
- **PR2 (Implicit FWT)** aims to fix the scaling (OOM issue) for $N > 14$ and reduce memory usage from $O(2^{2N})$ to $O(2^N)$.
