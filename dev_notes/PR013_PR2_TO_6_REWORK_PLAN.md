# PR2 to PR6 Rework and Refinement Plan

Based on the instructions, PR2 through PR6 need to be systematically reworked to inherit the robust testing infrastructure and structure introduced in PR1.

## Core Principles for PR2 - PR6 Rework

For every Pull Request (PR2 to PR6), the following steps MUST be strictly adhered to:

1. **Inherit PR1 Foundation:** All PRs must cleanly branch off or rebase on top of the latest PR1 state, ensuring the testing framework (`testing/`), benchmark wrappers, and CI checks are active.
2. **Code Cleanliness (Remove Comments):** Systematically remove all dead code, old experimental commented-out logic, and verbose debug `printf` / `print` statements.
3. **Adhere to `CONTRIBUTING.md`:**
   - Use `uv` for environment management.
   - Run `pre-commit run --all-files` locally before any commit.
   - Fix any `cargo fmt` or `cargo clippy` issues in the Rust backend.
4. **Mandatory Unit Tests:** For every new feature introduced, specific unit tests must be added to the `testing/` directory (or Rust `tests/` module) guaranteeing correctness.
5. **Baseline Benchmarking:** Establish a **Pre-PR1 Baseline** (the original naive implementation). Every PR must report its performance relative to this fixed baseline. We will not change the baseline for every PR to keep comparisons consistent.

---

## Pre-Requisite: Establish Baseline Benchmark
- **Action:** Run the performance benchmark (`benchmark_linux.py` or equivalent) on the pre-PR1 codebase.
- **Output:** Record the throughput, latency, and memory usage for batch sizes up to N=14. This will serve as the anchor for PR2-6 speedup claims.

---

## Specific PR Plans

### PR2: Implicit FWT (Matrix-Free) Integration — DONE (2026-06-10)
- **Goal:** Replace dense Hadamard matrix generation with standard SIMT implicit FWT.
- **Branch:** `pr2-implicit-fwt-rework` (from PR1 tip, pushed to `mahout_fork`).
- **Unit Tests:** `testing/qdp/test_implicit_fwt.py` — 12/12 passed.
- **Cleanup:** Removed naive `iqp_encode_kernel_naive` and small-N fallback dispatch in `iqp.cu`.
- **Pending:** Benchmark record on `internal-dev-notes` (code push is complete).

### PR3: Shared Memory Model Optimization
- **Goal:** Implement shared memory tiling and persistent cross-thread data exchanges to reduce DRAM bottlenecks.
- **Unit Tests:** Add memory-bound specific tests ensuring that multi-warp thread shuffling doesn't introduce race conditions.
- **Benchmark:** Compare memory bandwidth usage against the Pre-PR1 baseline.

### PR4: Runtime Dispatch & Hardware Fallbacks
- **Goal:** Dynamically detect SM architecture to route between standard SIMT and optimized hardware paths.
- **Unit Tests:** Mock hardware architecture queries in Python tests to verify that the runtime dispatch correctly routes to the fallback vs. advanced kernels.
- **Cleanup:** Remove hardcoded architecture bounds and assumptions.

### PR5: Implicit Hadamard Engine
- **Goal:** Advanced structural fusion for the Hadamard engine operations.
- **Unit Tests:** Verify the fused angle/amplitude encoding layers produce identical amplitudes as the separate passes.

### PR6: Tensor Core Acceleration (Ozaki/TF32)
- **Goal:** Introduce MMA (Matrix-Multiply-Accumulate) and Ozaki scheme for large N.
- **Unit Tests:** Rigorous precision testing to guarantee the Ozaki sliced-precision logic yields results within tolerance of the exact FP64 baseline.
- **Benchmark:** Compare extreme-N (N=14 to N=28) scale against the Pre-PR1 baseline to show the massive leap in capability.

## Conclusion
This structured approach ensures that PR2 through PR6 are not just experimental kernels, but production-ready modules that strictly follow Apache Mahout contribution standards, complete with automated tests and pre-commit hooks.
