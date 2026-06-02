# Report: Baseline Benchmark

## Goal
Establish the baseline performance metrics of the legacy dense implementation.

## Hypothesis
Baseline is memory-bound due to $O(N \times 2^N)$ dense matrix materialization.

## Implementation
Reference to existing PyTorch eager/compiled implementations and naive dense Hadamard matrices evaluated on NVIDIA RTX 4060 (Driver 581.57, CUDA 13.0).

## NCU Analysis
- **SM Throughput**: < 5% (Compute is starved)
- **DRAM Throughput**: > 80% (Memory bandwidth saturated)

## Bottleneck
**Memory-Bound**. The state vector memory requirement for N=30 is ~16GB, and for N=50 is ~16PB. Constructing the dense FWT matrix causes rapid Out-of-Memory (OOM) errors. For instance, PyTorch Eager/Compiled OOMs at 32GB RAM around N=16.

## Next Step
Implement Operator Fusion (Report 01).