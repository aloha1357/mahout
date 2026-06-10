### Related Issues

<!-- Closes #123 -->
N/A

### Changes

- [ ] Bug fix
- [x] New feature
- [ ] Refactoring
- [ ] Documentation
- [ ] Test
- [ ] CI/CD pipeline
- [ ] Other

### Why

As established in the previous Kronecker Decomposition PR, a significant bottleneck in processing high-qubit circuits ($N \ge 14$) is memory. A traditional $O(4^N)$ matrix representation for the full Dense Hadamard transform completely exhausts modern GPU VRAM limits (causing Out-Of-Memory errors).

Even with the Kronecker Decomposition splitting the matrix into smaller blocks, generating and storing the explicit dense $H$ matrices in memory before applying Tensor Core operations is highly inefficient.

We need a way to perform Dense Matrix Multiplications (GEMM) on the Tensor Cores *without ever storing the Hadamard Matrix in Global Memory*.

### How

This PR introduces the **Matrix-Free Implicit Hadamard Ozaki Engine**.

- **Implicit Matrix Generation:** The `ImplicitHadamardOzakiEngine` leverages the structural properties of the Hadamard matrix ($h_{i,j} = (-1)^{\text{popc}(i \& j)}$) to calculate the matrix elements *on-the-fly* directly inside Shared Memory.
- **Ozaki Multi-pass Tensor Core Execution:** Using the Ozaki INT8 scheme, we utilize the `.m16n8k32.s8` Tensor Core instructions to perform the GEMM natively in hardware. Because the Hadamard values are always $\pm 1$, we experience absolutely zero quantization error despite using the INT8 pipeline.
- **Removed Fallback:** Replaced the `naive_implicit_hadamard_gemm_kernel` placeholder from PR 4 with the actual calls to `engine.execute_implicit_hadamard`.
- **Build System Fix:** Updated `build.rs` to drop the unsupported `sm_75` (Turing) target fallback, as this specific Tensor Core instruction explicitly requires `sm_80` (Ampere) or higher.

## Checklist

- [x] Added or updated unit tests for all changes (Verified passing against existing CI test suite)
- [x] Added or updated documentation for all changes (Added explanatory inline comments for PR)
