### Related Issues

<!-- Closes #123 -->
N/A

### Changes

- [ ] Bug fix
- [x] New feature
- [x] Refactoring
- [ ] Documentation
- [ ] Test
- [ ] CI/CD pipeline
- [ ] Other

### Why

For smaller qubit counts ($N \le 12$), calculating the Fast Walsh-Hadamard Transform (FWT) using multiple global memory kernel launches becomes heavily DRAM bandwidth-bound (launch overhead and global memory roundtrips dominate the execution time). By keeping the entire state vector within the GPU's Shared Memory (which is much faster and has lower latency than DRAM), we can fuse the operations into a single kernel launch.

### How

- **Operator Fusion Kernel (`iqp_phase_fwt_normalize_tc_kernel`):** Created a new fused kernel that handles three steps entirely within Shared Memory:
  1. Computes the IQP phase and writes it directly to `extern __shared__ cuDoubleComplex shared_state[]`.
  2. Performs the in-place Hadamard FWT over the shared memory buffer.
  3. Normalizes the final amplitudes and writes them out to Global Memory.
- **Dynamic Dispatch:** Updated `launch_iqp_encode_tc` to dynamically allocate Shared Memory and dispatch to this fused kernel when `num_qubits <= FWT_SHARED_MEM_THRESHOLD`.

## Checklist

- [x] Added or updated unit tests for all changes (Verified passing against existing CI test suite)
- [x] Added or updated documentation for all changes (Added explanatory inline comments for PR)