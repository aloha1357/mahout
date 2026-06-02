# PR Roadmap: QDP TensorCore Integration

This roadmap defines the multi-stage integration of the AdaptiveGEMM research kernels into the Apache Mahout project.

## PR1: Core Logic & Test Infrastructure
- **Scope**: Define the backend traits, unit test harness, and CPU/SIMT fallbacks.
- **Goal**: Establish correctness baseline without performance focus.
- **Metric**: 100% pass on all `qumat` integration tests.

## PR2: Implicit FWT (Matrix-Free)
- **Scope**: Implement the matrix-free `iqp_tc.cu` kernels using standard SIMT.
- **Goal**: Resolve the OOM issue for $N > 14$.
- **Metric**: Scalability up to $N=28$ on 8GB VRAM.

## PR3: Memory Model Optimization
- **Scope**: Introduce Shared Memory tiling and L2-cache awareness.
- **Goal**: Reach the "SIMT Peak" performance.
- **Metric**: ~10x speedup vs PR2 for batch workloads.

## PR4: Runtime Dispatch & TF32/Ozaki
- **Scope**: Add runtime architecture detection and dispatch to Ozaki MMA kernels.
- **Goal**: Final performance peak (300x+ vs baseline).
- **Metric**: Match `research-final-v1` performance metrics.
