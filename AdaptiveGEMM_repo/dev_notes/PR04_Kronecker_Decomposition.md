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

For large qubit counts ($N > 12$), the standard Fast Walsh-Hadamard Transform (FWT) algorithm becomes severely bound by Global Memory bandwidth. The FWT requires $\log_2(N)$ stages of in-place memory access across the entire $2^N$ state vector, which causes cache thrashing and massive DRAM roundtrips.

To overcome this, we mathematically restructure the FWT into a Kronecker Product Decomposition: $H_n = H_{n/2} \otimes H_{n/2}$. This transforms the sparse, memory-bound butterfly operations into standard, dense matrix multiplications (GEMM) using a Blocked architecture.

While the implicit Hadamard engine is not yet introduced (coming in PR 5), this PR establishes the **structural memory layout, allocation, and transpose logic** necessary for the decomposition.

### How

- **Kronecker Decomposition Logic:** Updated `launch_iqp_encode_tc` to dynamically split the state vector into two dimensions ($n_1$ and $n_2$).
- **Intermediate Allocations:** Added temporary memory allocations (`d_temp_real`, `d_temp_imag`) to store the transposed matrix blocks during the 4-step blocked algorithm.
- **Naive GEMM Placeholder:** Introduced `naive_implicit_hadamard_gemm_kernel` as a fallback structural placeholder. It calculates the Hadamard values on-the-fly ($\text{popc}(k \ \&\ i)$) and executes the block multiplication $Z = X \cdot H_{n2}$.
- **Matrix Layout Transform:** Leveraged the `iqp_tc_batch_transpose_kernel` (introduced in PR 2) to transpose the blocks between the two GEMM stages, achieving the mathematical equivalent of the $O(N \log N)$ FWT through dense GEMMs.

## Checklist

- [x] Added or updated unit tests for all changes (Verified passing against existing CI test suite)
- [x] Added or updated documentation for all changes (Added explanatory inline comments for PR)
