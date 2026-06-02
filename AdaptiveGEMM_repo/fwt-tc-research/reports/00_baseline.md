# Report: Baseline Benchmark

## Goal
Establish the baseline performance metrics of the legacy dense implementation.

## Hypothesis
Baseline is memory-bound due to $O(N \times 2^N)$ dense matrix materialization.

## Implementation
Reference to existing dense Hadamard/unitary logic.

## NCU Analysis
- **SM Throughput**: [To be filled]
- **DRAM Throughput**: [To be filled]
- **L2 Cache Hit Rate**: [To be filled]

## Bottleneck
Currently unknown. Suspected DRAM throughput limitations.

## Next Step
Implement Operator Fusion (Report 01).