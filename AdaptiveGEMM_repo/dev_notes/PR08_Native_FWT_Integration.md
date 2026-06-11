### Related Issues

N/A

### Changes

- [x] Bug fix
- [x] New feature
- [ ] Refactoring
- [ ] Documentation
- [x] Test
- [ ] CI/CD pipeline
- [ ] Other

### Why

PR7 (`encode_batch_tc`) delivers correct, fast E2E IQP via Ozaki INT8 TC + fused SMEM FWT, but two gaps motivated PR8:

1. **Alternative Hadamard engine on disk:** `ImplicitHadamardNative.cu` (Tri Dao–style register/SMEM butterfly, no Ozaki quantization) was prototyped but not wired into the Mahout pipeline or benchmarked against PR7.
2. **Need evidence before changing default:** Reviewers and downstream QML users need an A/B path (`encode_batch_native` vs `encode_batch_tc`) with pytest + E2E numbers before PR9 lands a fused native kernel as default.

PR8 answers: **Native FP64 FWT beats PR7 E2E by 3.2–10.7× at N=12–16 with verification PASS.**

### How

#### `qdp/qdp-kernels/src/ImplicitHadamardNative.cu` / `.h` (new)

- `ImplicitHadamardNativeEngine::execute_implicit_hadamard_fp32/fp64` — extreme intra-block FWT (warp shuffle + SMEM) and multi-pass inter-block FWT for N≥15.
- **Bug fix:** FP64 dispatch for `N=7` (128-dim legs, e.g. N=14 Kronecker) used `THREADS=256` → `NUM_B=0`; fixed to `THREADS=128` (N=6 → 64).

#### `qdp/qdp-kernels/src/iqp_tc.cu` — `launch_iqp_encode_native`

- **N ≤ 12:** `iqp_phase_split_kernel` → Native FP64 FWT on real/imag → `recombine_complex_kernel`.
- **N > 12:** 4-step Kronecker (Native does not fuse `transpose_batch` yet):
  1. Phase split
  2. Native `H_{n2}` on each leg
  3. Explicit `iqp_tc_launch_transpose`
  4. Native `H_{n1}` + transpose + norm
- Reuses static grow-on-demand buffers and `device.cu_stream()` from PR7.

#### `qdp/qdp-kernels/build.rs`

- Compiles `ImplicitHadamardNative.cu`.

#### `qdp/qdp-core`, `qdp/qdp-python`

- **`encode_batch_native(data, num_qubits, encoding_method="iqp"|"iqp-z")`** — PyO3 + Rust API sibling to `encode_batch_tc`; does **not** change default `encode()` / `encode_batch()` behavior.

#### `qdp/qdp-python/benchmark/benchmark_e2e.py`

- Added **`mahout-native`** framework and `run_mahout_arrow_native` for side-by-side E2E vs Arrow-FWT and Mahout-TC.

#### `testing/qdp/`

- **`test_iqp_native_path.py`** — API, normalization, FWT vs Native agreement (N=8–18), Kronecker smoke.
- **`test_iqp_native_e2e.py`** — full Mahout pipeline (encode → DLPack → forward) vs FWT/TC.

#### `scripts/`

- `reproduce_pr8.sh` — one-shot build + unit + E2E pytest + N=12/14/16 benchmark.
- `build_wsl.sh`, `test_native_wsl.sh`, `bench_native_wsl.sh` — helper scripts.

**Verified (WSL2, RTX 4060 Laptop, 2026-06-11, tip `840f016f8`):**

| Test | Result |
|------|--------|
| `pytest test_iqp_native_path.py` | **29/29 passed** |
| `pytest test_iqp_tc_path.py` (regression) | **12/12 passed** |
| `pytest test_iqp_native_e2e.py -m "not slow"` | **6/6 passed** (2 slow optional) |
| E2E `iqp-z`, N=12, 32 samples | Native **0.008 s** vs TC **0.086 s** (**10.7×**); vs FWT **1.44 s**; verify **PASS** |
| E2E `iqp-z`, N=14, 32 samples | Native **0.020 s** vs TC **0.079 s** (**4.0×**); verify **PASS** |
| E2E `iqp-z`, N=16, 32 samples | Native **0.020 s** vs TC **0.063 s** (**3.2×**); verify **PASS** |
| Native vs TC amplitude @ N=16 | **1.94e-16** |

Reproduce: `wsl bash scripts/reproduce_pr8.sh`

Full tables: [`reports/PR008_Benchmark.md`](https://github.com/aloha1357/apache_mout/blob/main/reports/PR008_Benchmark.md) on `main`. PR9 plan: `PR09_Fused_Native_Plan.md`.

### What PR8 does NOT do (deferred to PR9)

- No fused `iqp_native_phase_fwt_normalize_*` kernel (N≤12 still uses phase_split + separate FWT).
- No 2-step fused-transpose Kronecker (still 4-step + explicit transpose; extra GM traffic).
- No default switch from `encode_batch_tc` to native.
- No FP32 QML fast path exposed in Python yet.

## Checklist

- [x] Added or updated unit tests (29 + 12 + 6 pytest, WSL2 2026-06-11)
- [x] Benchmark report on `apache_mout` `main` (`reports/PR008_Benchmark.md`, tip `840f016f8`)
- [x] PR body / handover on `internal-dev-notes`
- [x] E2E A/B vs PR7 with correctness verification
- [x] pre-commit on changed files (`SKIP=ty`; ruff, license, clippy pass in WSL)
- [x] No merge conflicts with PR6/PR7 stack (`git merge-tree` clean)

## Stack

**Base:** `mahout_fork/pr7-iqp-tc-ncu-profiling` (`ac7883296`)  
**Head:** `mahout_fork/pr8-native-hadamard-benchmark` (`840f016f8`)