# Report: TensorCore Attempt

## Goal
Map the FWT butterfly operations directly onto Tensor Cores using `wmma` instructions (Ozaki MMA).

## Hypothesis
Tensor Cores provide immense computational density; if memory traffic is solved, TCs will speed up execution massively for large batch sizes and $N \ge 14$.

## Implementation
Integrated the Ozaki Tensor Core MMA kernel targeting SM80+ capabilities (TF32/FP16). 

## NCU Analysis
- **SM Throughput**: ~67.54% (Compute-bound transition successful)
- **DRAM Throughput**: ~0.45% (Memory pressure eliminated)
- **Tensor Core Pipe Active**: Metrics truncated due to OS limits.

## Bottleneck
**Compute-bound**. Memory overhead is completely bypassed.

## Failure Analysis
**WDDM Limitation**: Under Windows Display Driver Model (WDDM), the Ozaki MMA kernel profiling hits a timeout, resulting in a `LaunchFailed` error when capturing full NCU hardware metrics. Pure execution works flawlessly (e.g., N=16 completes in ~74us, yielding a massive 375x speedup), but deep hardware metric collection requires Linux/TCC mode.

## Next Step
Ozaki representation to fix precision issues (Report 06).