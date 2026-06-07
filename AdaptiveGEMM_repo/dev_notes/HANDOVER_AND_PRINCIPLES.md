# IQP Encoding Optimization: Handover & Execution Principles

This document is the **Comprehensive Handover Manual** for any developer or AI assistant continuing the work on the Apache Mahout `qdp` (Quantum Data Processing) module. 

It defines the exact development methodology ("Git Archaeology"), the strict branching rules, documentation tracking protocols, and environment setup required to reconstruct the Tensor Core integrations for open-source PR submission.

---

## 1. Project Overview & Roadmap (The 6-PR Split Strategy)

We are incrementally introducing a highly complex "Matrix-Free Implicit Hadamard Tensor Core Engine" to the Apache Mahout quantum computing module (`qdp`). To make this massive feature reviewable by the open-source community, we are splitting the final research state (found on the `pr-final-version` tag) into 6 logical PRs:

- **[PR 1] Phase kernel optimization (Done ✅):** Removed warp divergence in `compute_phase` and moved `norm_factor` calculation to the host.
- **[PR 2] Batch throughput optimization (Done ✅):** Scaffolded `iqp_tc.cu`, implemented batch unrolling, real/imaginary splitting, and Shared-Memory Bank-Conflict-Free Transpose.
- **[PR 3] Shared-memory FWT path (Done ✅):** Implemented `iqp_phase_fwt_normalize_tc_kernel` for $N \le 12$. Fused phase computation, FWT, and normalization in Shared Memory.
- **[PR 4] Kronecker decomposition based FWT (Done ✅):** Introduced structural logic for Kronecker product FWT, bridging the gap between standard FWT and Tensor Cores.
- **[PR 5] Implicit Hadamard engine (Done ✅):** Implemented the Matrix-Free `ImplicitHadamardOzakiEngine`. Eliminates $O(4^N)$ dense matrix allocation for $N \ge 14$. Also removed the unsupported `sm_75` fallback in `build.rs` to allow INT8 `.m16n8k32` compilation.
- **[PR 6] Tensor Core acceleration (Next ⏳):** Hook up the `AdaptiveOzakiEngine` for mixed-precision graded-ring Tensor Core operations on non-Hadamard logic, finalizing the full pipeline.

---

## 2. Execution Methodology ("Git Archaeology")

We are performing **"Git Archaeology"**. The final, fully working code already resides in our local repository (on the `pr-final-version` branch). The task is *not* to invent new code, but to carefully dissect the final version and transplant it step-by-step into clean, logical PRs.

### 🛑 Strict Branching & Tracking Rules

We use a strictly compartmentalized branching strategy. You must understand the roles of these branches:

1. **`upstream/main` (The Clean Baseline):** The original Apache Mahout codebase. All PR chains start from here.
2. **`prX-[feature-name]` (The Code-Only PR Branches):**
   - **GOLDEN RULE:** **NO MARKDOWN (`.md`) FILES ALLOWED.** These branches must contain *only* C++/Rust/Build file changes.
   - **Linear History:** PR branches must be strictly sequential. `pr2` branches from `pr1`. `pr3` branches from `pr2`, etc.
   - If a documentation file accidentally gets tracked in a PR branch, you MUST use `git rm --cached`, `git commit --amend`, and potentially `git rebase --onto` to deep-clean the history before pushing.
3. **`internal-dev-notes` (The Documentation Staging Branch):**
   - All PR drafts (`PR01_...md` to `PR06_...md`), checklists, and this `HANDOVER_AND_PRINCIPLES.md` file MUST be tracked *only* here.
   - We do not PR this branch. It's a synchronization hub.
4. **Local `main` (The Ultimate Archive & Research Record):**
   - We use the local `main` branch as a comprehensive archive.
   - **Protocol:** Whenever documentation is updated on `internal-dev-notes`, it must be synced over to local `main` (e.g., `git checkout main && git checkout internal-dev-notes -- path/to/md && git commit`). 
   - *Never* push local `main` to `upstream/main`.

### Execution Workflow per PR:
1. `git checkout pr[X-1]` -> `git checkout -b pr[X]`
2. Use `git show pr-final-version:path/to/file` or `git diff` to extract specific lines.
3. Implement surgically. Add inline English comments prefixed with `// PR[X]:`.
4. Run tests (see Testing Protocol).
5. Stage and commit code.
6. `git checkout internal-dev-notes`, write `PR0X_[Name].md`, commit, and push `internal-dev-notes`.
7. `git checkout main`, sync the new docs from `internal-dev-notes`, commit.
8. `git checkout pr[X]` and `git push -f mahout_fork pr[X]`.

---

## 3. Testing Principles (WSL Mandate)

Due to the Windows host lacking native `nvcc` compilation tools in the PATH, all Rust/CUDA testing **must** be executed within the WSL (Windows Subsystem for Linux) environment.

**Mandatory Test Command:**
```bash
wsl -e bash -ic 'export PATH=/usr/local/cuda/bin:$PATH && cd /mnt/d/D_backup/2025/tum/26S/apache_mout/qdp && cargo test --workspace --exclude qdp-python --lib'
```
*   **Zero-Failure Tolerance:** Code **must not** be force-pushed to the remote PR branch until this test command passes with 0 failures.
*   **Implicit Correctness:** CI/CD correctness is assumed by the repository maintainers; we do not submit standalone "Correctness" PRs. Feature PRs will be validated automatically by the CI test suite using the tests we run locally.

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
- [x] New feature
- [ ] Refactoring
- [ ] Documentation
- [ ] Test
- [ ] CI/CD pipeline
- [ ] Other

### Why
[Explain the performance bottleneck, mathematical rationale, or hardware limitation being solved (e.g., VRAM limit, warp divergence).]

### How
[Bullet points mapping directly to the code changes. Explain the mechanism of the optimization.]

## Checklist

- [x] Added or updated unit tests for all changes (Verified passing against existing CI test suite)
- [x] Added or updated documentation for all changes (Added explanatory inline comments for PR)
```

---

## 5. Performance Profiling (Nsight Compute) 

*(Reference for Future Maintainers)*

When reviewing the success of these optimizations later, follow the "Question-Driven Profiling" principles defined in our legacy notes (`07_pr_sugestion.md`):

1. **Launch Overhead Bound?** Check `sm__throughput`, `smsp__inst_executed_pipe_tensor.sum`, and total `GPU time`. If SM utilization is low and runtime is flat across $N=14$ and $N=16$, you are bound by kernel launch.
2. **Compute vs. Memory Bound?** Compare `dram__throughput` vs `l1tex__data_pipe_lsu_wavefronts_mem_shared.sum` vs `sm__throughput`.
3. **Resource Saturation (Why N=28 fails):** Look at `launch__shared_mem_per_block`, `launch__registers_per_thread`, and local memory spills to prove hardware exhaustion, not algorithmic flaws.