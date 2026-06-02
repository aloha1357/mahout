# PR Roadmap: AdaptiveGEMM FWT Implementation

This roadmap outlines the sequential merge strategy for integrating the High-Performance CUDA FWT backend into Apache Mahout.

## Overview
Based on the Research Final results, we have identified a 375x speedup potential. To ensure maintainability and reviewability, we decompose this into 4 distinct PRs.

---

## [PR 1] Infrastructure & Correctness
- **Goal**: Establish the testing harness and CI/CD compatibility.
- **Scope**:
  - Add `AdaptiveOzaki.cu` and `ImplicitHadamardOzaki.cu` stubs.
  - Implement the `precision_test.cpp` and `test_pybind.py` validation suite.
  - Fix NVCC architecture limits for GitHub Actions.
- **Merge Justification**: Essential for any further GPU development.

## [PR 2] Implicit FWT Implementation
- **Goal**: Move from $O(2^{2N})$ space to $O(2^N)$ space.
- **Scope**:
  - Implement the basic recursive implicit FWT kernel.
  - Remove dense matrix materialization in `qumat/qdp.py`.
- **Merge Justification**: 2.5x speedup and significantly reduced memory footprint (crucial for $N > 14$).

## [PR 3] Memory Model Optimization (Shared Memory)
- **Goal**: Maximize throughput for mid-range $N$.
- **Scope**:
  - Introduce Tiled Shared Memory FWT kernels.
  - Optimize thread-block mapping for the state vector.
- **Merge Justification**: 8.7x speedup; addresses the DRAM bottleneck identified in NCU profiling.

## [PR 4] Runtime Dispatch & TensorCore Fallbacks
- **Goal**: Production-ready robustness.
- **Scope**:
  - Implement dynamic dispatch for Ozaki MMA vs SIMT.
  - Add CPU fallbacks and precision-mode selection.
- **Merge Justification**: Brings the full 375x speedup to the user with safety and compatibility.

---
**Reference**: See [Comparison Matrix](./comparison_matrix.md) for detailed performance attribution.
