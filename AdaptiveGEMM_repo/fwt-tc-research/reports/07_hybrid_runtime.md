# Report: Hybrid Runtime Dispatch

## Goal
Dynamically route traffic to either CUDA Core (SIMT) or Tensor Core (Ozaki) based on $N$ (number of qubits) and batch size.

## Hypothesis
TC overhead isn't worth it for small $N$, but provides massive speedups for large $N$.

## Implementation
[To be filled]

## NCU Analysis
- **SM Throughput**: [To be filled]
- **DRAM Throughput**: [To be filled]

## Bottleneck
[To be filled]

## Next Step
Final validation and Multi-GPU roadmap (Report 08).