# Report: TensorCore Attempt

## Goal
Map the FWT butterfly operations directly onto Tensor Cores using `wmma` instructions.

## Hypothesis
Tensor Cores provide immense computational density; if memory traffic is solved, TCs will speed up execution.

## Implementation
[To be filled]

## NCU Analysis
- **SM Throughput**: [To be filled]
- **Tensor Core Pipe Active**: [To be filled]

## Bottleneck
[To be filled]

## Failure Analysis
If performance degraded, why? (e.g., memory staging overhead).

## Next Step
Ozaki representation to fix precision issues (Report 06).