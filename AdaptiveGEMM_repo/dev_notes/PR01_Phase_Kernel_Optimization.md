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

The original phase encoding and IQP encoding kernels suffered from GPU thread divergence due to conditional branching (`if (val != 0.0)` or `if ((x >> i) & 1U)`). Furthermore, the normalization factor (`norm_factor`) was being redundantly calculated inside the GPU kernel, consuming extra cycles. Eliminating these inefficiencies significantly improves the kernel's execution speed on the GPU.

### How

- **Replaced Conditional Branching:** In both `phase.cu` and `iqp.cu`, the `if` conditions checking bit states were replaced with boolean arithmetic casting and multiplication (e.g., `phases[bit] * (double)((idx >> bit) & 1U)`). This ensures that all threads in a warp follow the exact same instruction path, eliminating warp divergence.
- **Host-side Pre-calculation:** Moved the `norm_factor` calculation to the host (CPU) before launching the kernel in `phase.cu`, passing the result as an immutable parameter.
- **Added Explanatory Comments:** Included inline documentation near the bitwise arithmetic lines to aid code reviewers in understanding the optimizations.

## Checklist

- [x] Added or updated unit tests for all changes (Verified passing against existing CI test suite)
- [x] Added or updated documentation for all changes (Added explanatory inline comments for PR)
