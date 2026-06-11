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

PR6 introduced `encode_batch_tc` and the Kronecker TC-FWT path, but three problems blocked production use:

1. **E2E crash at N > 12:** Native micro-benchmarks passed, yet the Python Mahout pipeline aborted with `CUDA_ERROR_MISALIGNED_ADDRESS`. `launch_iqp_encode_tc` was called with a `NULL` stream from Rust while Ozaki kernels used the CUDA Runtime API inside a cudarc Driver API context. Errors were also swallowed (`return cudaSuccess` unconditionally).
2. **Incorrect states at N > 12:** The PR6 persistent Ozaki MMA path (`implicit_hadamard_ozaki_persistent_kernel_implicit`, 58 KiB dynamic shared memory, `ldmatrix`) produced wrong amplitudes in E2E (~0.44 max diff) and was unstable under WDDM.
3. **No reviewable E2E comparison:** Reviewers need a cold-start benchmark comparing Mahout-Arrow (FWT) vs Mahout-TC with correctness verification, not encode-only micro-benchmarks alone.

### How

#### `qdp/qdp-kernels/src/iqp_tc.cu` — `launch_iqp_encode_tc`

- **N ≤ 12:** unchanged fused path (`iqp_phase_fwt_normalize_tc_kernel`); fixed `data_len` for ZZ (`enable_zz` → `num_qubits + n*(n-1)/2`).
- **N > 12:** replaced the PR6 four-step pipeline (2× `execute_implicit_hadamard` + 2× `iqp_tc_launch_transpose`) with a **two-step Kronecker path** that passes `transpose_batch=true` into Ozaki so transpose is fused into the Hadamard epilogue (fewer global memory round-trips).
- Added **`CHECK_CUDA_RETURN`** macro and return **`cudaGetLastError()`** instead of hardcoded success.
- Replaced per-call **`cudaMalloc`/`cudaFree`** with **static reusable buffers** (grow-on-demand).

#### `qdp/qdp-kernels/src/ImplicitHadamardOzaki.cu` / `.h`

- Removed PR6 **persistent/grid MMA Ozaki** dispatch (`OZAKI_NCU_PROFILE`, 58 KiB dynamic smem, internal `cudaStreamSynchronize`).
- Added **`implicit_hadamard_ozaki_fused_batch_kernel`**: on-the-fly modulo quantization + INT8 MMA + fused batch transpose for `n ∈ {64, 128}` tiles.
- Added **`naive_hadamard_ozaki_kernel`**: mathematically exact FP64 fallback for other tile sizes.
- Extended API: `execute_implicit_hadamard(..., transpose_batch, batch_rows)`.

#### `qdp/qdp-core/src/gpu/encodings/iqp.rs`

- Pass **`device.cu_stream()`** into `launch_iqp_encode_tc` instead of `std::ptr::null_mut()`.

#### `qdp/qdp-core/src/lib.rs`, `qdp/qdp-python/src/engine.rs`, `qdp/qdp-python/qumat_qdp/backend.py`

- **`encode_batch_tc(data, num_qubits, encoding_method="iqp")`** — dispatches `IqpEncoder::full()` or `z_only()` for `iqp` / `iqp-z`.

#### `qdp/qdp-python/benchmark/benchmark_e2e.py`

- Added **`run_mahout_arrow_tc`** and **`mahout-tc`** framework; default for `iqp`/`iqp-z` runs **Mahout-Arrow (FWT) vs Mahout-TC** side-by-side with amplitude/probability verification.

**Verified (WSL2, RTX 4060 Laptop, 2026-06-11, tip `1e7459a6e`):**

| Test | Result |
|------|--------|
| `pytest test_iqp_tc_path.py` | **12/12 passed** (N=8,12 normalized; N=14–18 FWT vs TC agreement) |
| Encode `iqp`/`iqp-z` N=14–18 | max_err **~1e-9 – 1e-17** vs FWT |
| E2E `iqp-z`, N=12, 32 samples | TC **~0.05 s** vs FWT **~0.83 s** (**~16×**); MAE **~1.7e-8** |
| E2E `iqp-z`, N=14, 32 samples | TC **~0.05 s** vs FWT **~1.1 s** (**~22×**); MAE **~1.5e-8** |
| E2E `iqp-z`, N=16, 32 samples | TC **0.057 s** vs FWT **0.827 s** (**14.5×**); MAE **1.48e-8** |
| Micro encode-only N>12, batch=32 | N=16 FWT **~10 ms** / TC **~32 ms** (E2E pipeline is the win) |

Kronecker fix: `naive_hadamard` + `transpose_batch` for n≥256; fused INT8 TC for n∈{64,128}; E2E DLPack `.clone()`.

Full tables: `reports/PR007_Benchmark.md` on `main`. Handover: `PR07_HANDOVER.md`.

## Checklist

- [x] Added or updated unit tests (`pytest testing/qdp/test_iqp_tc_path.py` — 12 passed, WSL2 2026-06-11)
- [x] Added or updated documentation for all changes (this file on `internal-dev-notes`; benchmark report on `apache_mout` `main`)
- [x] pre-commit passed on changed files (`SKIP=ty`)
- [x] No merge conflicts with PR6 stack tip (`MERGEABLE` on GitHub)