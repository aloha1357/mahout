### Related Issues

<!-- Closes #123 -->
N/A

### Changes

- [x] Bug fix
- [x] New feature
- [x] Refactoring
- [ ] Documentation
- [x] Test
- [ ] CI/CD pipeline
- [ ] Other

### Why

PR6 delivered the Kronecker TC-FWT path (`encode_batch_tc`) and micro-benchmarks, but three gaps blocked a reviewable upstream submission:

1. **E2E breakage at N > 12:** C++ `bench_kernels` passed while the Python Mahout pipeline aborted on `CUDA_ERROR_MISALIGNED_ADDRESS`. Root cause: `launch_iqp_encode_tc` was invoked with a `NULL` CUDA stream from Rust, mixing Runtime API kernel launches with cudarc's Driver API context. Swallowed CUDA errors (`return cudaSuccess` unconditionally) hid the failure until `device.synchronize()`.
2. **No end-to-end comparison:** Reviewers need a cold-start benchmark (disk → Arrow IPC → GPU encode → forward pass) comparing **Mahout-Arrow (FWT)** vs **Mahout-TC** vs reference frameworks — not encode-only micro-benchmarks alone.
3. **Incomplete performance attribution:** Side-kernel NCU evidence (phase split, FWT butterfly, modulo precompute) and reproducible bench drivers were required to answer advisor Q1–Q4 (runtime, why faster, bottleneck, next steps).

### How

- **CUDA stream propagation (PR8):** Pass `device.cu_stream()` from `iqp.rs` into `launch_iqp_encode_tc` instead of `std::ptr::null_mut()`, so Ozaki kernels share the same stream as cudarc allocations.
- **Error handling (PR8):** Add `CHECK_CUDA_RETURN` and return `cudaGetLastError()` from `launch_iqp_encode_tc`; fix dynamic `data_len` for full `iqp` with ZZ interactions.
- **Ozaki engine stability (PR8/9):** Replace crash-prone persistent MMA dispatch with fused-batch Hadamard (`implicit_hadamard_ozaki_fused_batch_kernel`) plus naive FP64 fallback for small tiles; retain grid kernel symbols for native NCU bench attempts.
- **E2E benchmark (`benchmark_e2e.py`):** Add `mahout-tc` framework runner (`run_mahout_arrow_tc`) alongside `mahout-arrow`; default `iqp` / `iqp-z` runs compare FWT vs TC side-by-side with correctness verification (max amplitude / probability diff).
- **`encode_batch_tc` API:** Extend PyO3 / Rust binding with `encoding_method` (`iqp` | `iqp-z`); default `"iqp"` preserves PR6 test compatibility.
- **NCU infrastructure (PR7):** Add `bench_kernels.cu`, `wsl_build_and_ncu.sh`, NCU CSV summaries, `benchmark_pr7.py`, and `test_pr7_ncu_path.py`.
- **Inline comments:** Added `// PR7:` markers on changed CUDA/Rust sites where applicable.

**Measured (WSL2, RTX 4060 Laptop, 2026-06-10):**

| Benchmark | Key result |
|-----------|------------|
| E2E `benchmark_e2e.py` (iqp-z, N=14, 32 samples) | Mahout-TC **0.036 s** vs Mahout-Arrow **1.043 s** (**28.7×**); max amplitude diff **1.59e-8** |
| Micro `benchmark_pr7.py` (iqp-z, N=14, batch=64) | Encode-only TC still slower than FWT (1.98 ms vs 1.29 ms); E2E wins on full pipeline |
| NCU side kernels (N=14, batch=1) | FWT butterfly DRAM ~33%; phase split SM **~67%** (compute-bound shift) |

Full tables and reproduce commands: `reports/PR007_Benchmark.md` on `apache_mout` `main`.

## Checklist

- [x] Added or updated unit tests for all changes (`pytest testing/qdp/test_pr7_ncu_path.py testing/qdp/test_iqp_tc_path.py` — **9 passed**, WSL2 2026-06-10)
- [x] Added or updated documentation for all changes (this `PR07_E2E_TC_Integration.md` on `internal-dev-notes`; benchmark report on `apache_mout` `main`)
- [x] pre-commit passed on changed files (`SKIP=ty`)
- [x] No merge conflicts with PR4/PR5/PR6 stack tips (verified `git merge-tree` + GitHub `MERGEABLE`)