### Related Issues

<!-- Closes #123 -->
N/A

### Changes

- [ ] Bug fix
- [ ] New feature
- [x] Refactoring
- [ ] Documentation
- [ ] Test
- [ ] CI/CD pipeline
- [ ] Other

### Why

As part of the IQP Encoding Optimization PR Split Plan, PR 2 focuses on "Batch throughput optimization". To scale executions efficiently and prepare for Tensor Core acceleration (which will be fully introduced in PR 5 & 6), we need a robust scaffolding for batch data transformation. The original code processed matrices sequentially; this refactoring introduces batched layouts and kernels required for the Kronecker-based matrix multiplication that Tensor Cores will eventually execute.

### How

- **Created `iqp_tc.cu`:** Introduced new kernels specifically designed to manage memory layout for batched operations.
- **Phase Split Kernel (`iqp_phase_split_kernel`):** Unrolls the batch and splits the initial phase computation into pure real and imaginary parts to prepare for INT8 matrix multiplication.
- **Batch Transpose Kernel (`iqp_tc_batch_transpose_kernel`):** Implemented a Shared Memory Bank-Conflict-Free matrix transpose kernel, essential for efficiently reordering data between Tensor Core FWT stages.
- **Recombine Kernel (`recombine_complex_kernel`):** Restores the split real and imaginary parts back into the standard `cuDoubleComplex` format expected by downstream processes.
- **Rust Integration:** Updated `lib.rs` and `iqp.rs` to expose and call the new `launch_iqp_encode_tc` function from Rust, laying the structural groundwork for the full Tensor Core pipeline.

## Checklist

- [x] Added or updated unit tests for all changes (Verified that existing tests pass, and batching logic doesn't break `qdp-core`)
- [x] Added or updated documentation for all changes (Added explicit comments describing the purpose of the new kernels)