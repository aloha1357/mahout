# PR2: Implicit FWT Backend Integration

## Scope
Replace the legacy dense Hadamard matrix construction with the matrix-free implicit Fast Walsh-Hadamard Transform (FWT).

## Target Changes
- CUDA kernel for implicit FWT
- Shared memory layout enhancements for butterfly operations
- Python API hooks for `iqp` and `phase`

## Review Focus
- Numerical stability and absolute correctness compared to baseline
- DRAM throughput improvements
- API backward compatibility

## Merge Blockers
- Pre-existing correctness tests (from PR1) must pass
- Verification against standard Qiskit/Braket simulators
