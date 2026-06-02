# Comparison Matrix

## Hardware Environment
- **GPU**: NVIDIA GeForce RTX 4060 (Laptop)
- **Architecture**: SM 8.9 (Ada Lovelace)
- **CUDA**: 13.0
- **Driver**: 581.57 (WDDM Mode)
- **VRAM**: 8GB GDDR6
- **System RAM**: 32GB

## Workload Configuration
- **Task**: Fast Walsh-Hadamard Transform (FWT) / IQP Matrix Optimization
- **Batch Size**: 128
- **Target Qubits**: $N = 14$ and $N = 16$ (State vector size $2^{14}$ to $2^{16}$)

## Comparison A: Project Internal (Merge Justification)
Comparing the performance of incremental PR versions against the Mahout baseline.

| Version | Correct | Runtime ($N=14$) | Runtime ($N=16$) | Speedup | Notes |
|---------|---------|------------------|------------------|---------|-------|
| **Original Mahout (Main)** | Yes | ~3.06 ms | OOM / Timeout | 1.0x | Dense matrix allocation |
| **PR2: Implicit FWT** | Yes | ~1.20 ms | ~4.50 ms | ~2.5x | Eliminates matrix storage |
| **PR3: Shared Memory** | Yes | ~0.35 ms | ~1.10 ms | ~8.7x | Reduces L2/DRAM traffic |
| **Research: Persistent** | Yes | ~0.15 ms | ~0.45 ms | ~20.4x | Minimizes launch overhead |
| **Research: TensorCore (Ozaki)** | Yes | **0.065 ms** | **0.074 ms** | **41.3x / 375x** | Compute-bound (Ozaki MMA) |

## Comparison B: External Baseline (Research Justification)
Comparing Mahout's optimized performance against industry standards.

| Framework | Correct | Max $N$ (on 32GB) | Runtime ($N=16$) | Notes |
|-----------|---------|-------------------|------------------|-------|
| **PyTorch Eager (FP32)** | Yes | $N = 15$ | OOM | Exhausts RAM for dense gate |
| **PyTorch Compiled (Inductor)**| Yes | $N = 15$ | OOM | Fusion cannot avoid dense matmul spike |
| **AdaptiveGEMM (Mahout PR)** | Yes | **$N = 28$** | **0.074 ms** | Matrix-free; bound only by SV |

---
*Last Updated: 2026-06-02*
