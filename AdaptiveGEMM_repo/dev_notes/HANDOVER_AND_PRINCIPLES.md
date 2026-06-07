# IQP Encoding Optimization: Handover & Execution Principles

This document serves as a centralized guide for the execution, testing, and documentation of the IQP Encoding Optimization PR Sequence. It outlines the project roadmap, the methodology for extracting code from the final research branch, and the strict standards for PR submission.

## 1. The Roadmap (6-PR Split Strategy)

We are incrementally introducing a highly complex "Matrix-Free Implicit Hadamard Tensor Core Engine" to the Apache Mahout quantum computing module (`qdp`). To make this reviewable by the open-source community, we are splitting the final research state into 6 logical PRs:

- **[PR 1] Phase kernel optimization (Done):** Removed warp divergence in `compute_phase` and moved `norm_factor` calculation to the host.
- **[PR 2] Batch throughput optimization (Done):** Scaffolded `iqp_tc.cu`, implemented batch unrolling, real/imaginary splitting, and Shared-Memory Bank-Conflict-Free Transpose.
- **[PR 3] Shared-memory FWT path (Next):** Implement the `iqp_phase_fwt_normalize_tc_kernel` for $N \le 12$ (or `FWT_SHARED_MEM_THRESHOLD`). Fuses phase computation, FWT, and normalization directly in Shared Memory to bypass DRAM bottlenecks.
- **[PR 4] Kronecker decomposition based FWT:** Introduce the structural logic for Kronecker product FWT, bridging the gap between standard FWT and the Tensor Core implementation.
- **[PR 5] Implicit Hadamard engine:** Implement the Matrix-Free `ImplicitHadamardOzakiEngine`. Eliminates the $O(4^N)$ dense matrix allocation, resolving OOM issues for $N \ge 14$.
- **[PR 6] Tensor Core acceleration:** Hook up the Ozaki INT8 mixed-precision engine to dispatch the Kronecker matrix multiplications to hardware Tensor Cores.

---

## 2. Execution Methodology (Git Archaeology)

We are essentially performing "Git Archaeology". The final, fully working code resides in the local history (specifically on the `pr-final-version` branch or previous commits).

**Workflow per PR:**
1. **Branching:** Create a new branch from the `main` (or the previous PR's branch if dependent). E.g., `git checkout -b pr3-shared-memory-fwt`.
2. **Extraction:** Use `git show` or `git diff` against `pr-final-version` to identify the specific lines of code relevant to the current PR's scope.
3. **Implementation:** Use `replace` or manual edits to surgically inject *only* the necessary logic. Avoid bringing in variables, headers, or parameters that belong to later PRs.
4. **Commenting:** Add inline English comments prefixed with `// PR[X]:` (e.g., `// PR3: Fuse Phase and FWT in Shared Memory`) to explain *why* the optimization is done, aiding external code reviewers.

---

## 3. Testing Principles

Due to the Windows environment lacking native CUDA compilation tools, all Rust/CUDA testing **must** be executed within the WSL environment.

**Mandatory Test Command:**
```bash
wsl -e bash -ic 'export PATH=/usr/local/cuda/bin:$PATH && cd /mnt/d/D_backup/2025/tum/26S/apache_mout/qdp && cargo test --workspace --exclude qdp-python --lib'
```
*   **Rule:** Code **must not** be force-pushed to the remote repository until this test command passes with 0 failures.
*   **Rule:** CI/CD correctness is assumed; we do not need standalone correctness PRs. Feature PRs will be validated by the existing test suite.

---

## 4. PR Documentation Standards

For every PR, a standardized Markdown document must be created and saved in `AdaptiveGEMM_repo/dev_notes/` with the naming convention `PR0X_Feature_Name.md`.

**Required Template:**
```markdown
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
[1-2 paragraphs explaining the performance bottleneck or architectural limitation being solved.]

### How
[Bullet points mapping directly to the code changes. Explain the mechanism of the optimization.]

## Checklist

- [x] Added or updated unit tests for all changes (Verified passing against existing CI test suite)
- [x] Added or updated documentation for all changes (Added explanatory inline comments for PR)
```

---

## 5. Performance Profiling (Nsight Compute)

When reviewing the success of these optimizations later, follow the principles defined in `07_pr_sugestion.md`.

**Question-Driven Profiling:** Do not look at all metrics at once.
1. **Is it Launch Overhead?** Check `sm__throughput`, `smsp__inst_executed_pipe_tensor.sum`, and total `GPU time`. If SM/Tensor utilization is low and runtime is flat (e.g., N=14 and N=16 take the same time), you are bound by kernel launch overhead.
2. **Compute vs. Memory Bound?** Compare `dram__throughput` vs `l1tex__data_pipe_lsu_wavefronts_mem_shared.sum` vs `sm__throughput`.
3. **Resource Saturation (Why N=28 fails):** Look at `launch__shared_mem_per_block`, `launch__registers_per_thread`, and local memory spills to prove resource exhaustion, not algorithmic bugs.

Always test on target qubit counts: `N=14` (Small), `N=20` (Mid), `N=26` (Near Boundary), `N=28` (Failure state).