# PR9 Plan — Fused Native IQP Kernel

**Date:** 2026-06-11  
**Baseline:** PR8 `pr8-native-hadamard-benchmark` tip `840f016f8`  
**Decision:** PR8 E2E benchmarks **justify PR9** — Native already 3.2–10.7× faster than PR7 with verify PASS.

---

## Goal

Close the remaining gaps between PR8 (correct but multi-kernel) and an optimal production path:

1. **Fuse** Phase + Native-FWT + Normalize for N≤15 (match PR7 fused SMEM ergonomics, beat PR8 buffer traffic).
2. **Fuse** Kronecker to **2-step** with native transpose epilogue (match PR7 step count, drop Ozaki INT8 overhead).
3. **Optionally** expose FP32 QML path; keep FP64 default for quantum encoding accuracy.

Target: **≥1.2× E2E vs PR8** at N=12,14,16, or demonstrably lower global-memory bytes moved.

---

## PR8 measured baseline (what PR9 must beat)

| N | PR8 Native E2E | PR7 TC E2E | PR8 advantage |
|---|----------------|------------|---------------|
| 12 | 0.008 s | 0.086 s | 10.7× vs TC |
| 14 | 0.020 s | 0.079 s | 4.0× vs TC |
| 16 | 0.020 s | 0.063 s | 3.2× vs TC |

PR8 encode-only @ N=16: Native **9.3 ms** vs TC **32.4 ms** — fusion should widen this gap further.

---

## Architecture

```
PR8 (today)                         PR9 (target)
─────────────────────────────────────────────────────────────
N≤12: phase_split                   N≤15: iqp_native_phase_fwt_normalize_fp64_kernel
      + native FWT ×2                     (single kernel, SMEM phase + extreme FWT)
      + recombine

N>12: phase_split                   N>12: phase_split (or fused phase in pass-0)
      + native H_n2                       + native_kronecker_2step_fp64
      + transpose ×2                        (fused transpose epilogue per leg)
      + native H_n1
      + recombine
```

---

## PR9 work packages

### 9.1 Fused N≤12 kernel (extend `ImplicitHadamardNative.cu`)

**Important:** PR9 fuses **PR8's fastest kernel** (`native_fp64_extreme_fwt_kernel`), **not** the PR7 TC naive SMEM butterfly (`iqp_phase_fwt_normalize_tc_kernel`).

**New kernels:**

```cuda
native_fp64_extreme_iqp_fused_kernel<N, THREADS>  // Phase + extreme FWT + norm
native_fp64_extreme_fwt_transform<THREADS>()      // shared __device__ helper
```

**Design:**

- Per-sample block: `blockIdx.x = sample_idx` (same grid as extreme FWT)
- Phase: `compute_phase_iqp` → cos/sin into `reg_real` / `reg_imag`
- FWT: `native_fp64_extreme_fwt_transform` on real, then imag (warp-shuffle + SMEM, same as PR8)
- Norm: `1/state_len`, write `cuDoubleComplex` once (no phase_split / recombine)
- Dispatch: N=6..12, same thread/SMEM table as `execute_implicit_hadamard_fp64` tier-1

**Dispatch:**

| N | Kernel | block | smem |
|---|--------|-------|------|
| 8–12 | fused fp64 | `num_samples` | `state_len * 16` bytes (complex) |
| 13–15 | fused fp64 multi-chunk or v2 | TBD from NCU | 48–64 KiB |

**API:** `launch_iqp_encode_native` N≤15 branch calls fused kernel directly (remove phase_split path).

**Tests:** extend `test_iqp_native_path.py`; add NCU smoke for bank conflicts.

---

### 9.2 Native fused-transpose Kronecker (N>12)

**Problem:** PR8 uses 4 kernels per leg (FWT + transpose) × 2 legs = **8** Hadamard-related launches + 2 transposes. PR7 uses **2** Ozaki calls with fused transpose.

**Solution:** Add to `ImplicitHadamardNative.cu`:

```cpp
void execute_implicit_hadamard_fp64_fused_transpose(
    const double* d_A, double* d_C,
    size_t m, size_t n, size_t k,
    double norm_factor, cudaStream_t stream,
    size_t batch_rows  // dim1 or dim2
);
```

Implement epilogue matching `naive_hadamard_ozaki_kernel` / `implicit_hadamard_ozaki_fused_batch_kernel` scatter:

```cpp
// After FWT on row r, col c:
C[batch_idx * (k * batch_rows) + c * batch_rows + r_in_batch] = val;
```

Then `launch_iqp_encode_native` N>12 becomes **2-step** (same structure as PR7 `launch_iqp_encode_tc`):

```cpp
engine.execute_implicit_hadamard_fp64_fused_transpose(d_state_real, d_temp_real, ...);
engine.execute_implicit_hadamard_fp64_fused_transpose(d_temp_real, d_out_real, ..., norm_factor);
// + imag leg; recombine
```

**Remove:** explicit `iqp_tc_launch_transpose` from native path.

**Tests:** N=14,16,17,18 agreement vs FWT; compare launch count via Nsight Systems.

---

### 9.3 API / product integration

| Decision | Recommendation |
|----------|----------------|
| Default IQP batch path | Switch `encode_batch_tc` internals to native fused **or** add `encoding_method="iqp-native"` and deprecate TC path for IQP-only |
| Keep Ozaki TC | Retain `encode_batch_tc` as alias for backward compat one release |
| FP32 | `encode_batch_native(..., precision="fp32")` behind explicit flag; document ~1e-4 tolerance |
| E2E benchmark | `--frameworks mahout-native-fused` or make `mahout-native` the fused path |

---

### 9.4 CI / upstream checklist

- [ ] ASF license headers on `ImplicitHadamardNative.cu/.h`
- [ ] pre-commit CUDA + Python on all touched files
- [ ] `pytest test_iqp_native_path.py` + `test_iqp_tc_path.py` green
- [ ] E2E benchmark table in `reports/PR009_Benchmark.md` on `main`
- [ ] PR body `PR09_Fused_Native_Integration.md`

---

## Branch / stack

```
PR7 pr7-iqp-tc-ncu-profiling
  └── PR8 pr8-native-hadamard-benchmark  ← merged first
        └── PR9 pr9-native-fused-iqp
```

**Suggested PR9 sub-PRs (optional stack):**

| Sub-PR | Scope | Depends |
|--------|-------|---------|
| PR9a | Fused N≤15 fp64 kernel | PR8 |
| PR9b | Native fused-transpose Kronecker | PR9a |
| PR9c | FP32 QML path + API default switch | PR9b |

---

## Success criteria

| Metric | Gate |
|--------|------|
| Correctness | FWT vs native fused: max err < **1e-8** @ N=8–18 |
| E2E vs PR8 | ≥ **1.2×** faster @ N=12,14,16 OR ≥ **30%** fewer GM bytes (NCU) |
| E2E vs PR7 TC | maintain ≥ **3×** advantage @ N=16 |
| Regression | `test_iqp_tc_path.py` still passes (Ozaki path untouched or aliased) |

---

## Risk register

| Risk | Mitigation |
|------|------------|
| Fused SMEM exceeds 48 KiB @ N=15 | Use chunking like `native_fp32_extreme_fwt_kernel_v2` |
| Fused transpose epilogue bugs @ N=14 | Unit test single 128×128 leg vs PR8 4-step before E2E |
| Default switch breaks downstream | Keep `encode_batch_tc` one release; changelog |
| FP32 training drift | Opt-in only; pytest tolerance 1e-4 |

---

## Progress (2026-06-11)

| Sub-PR | Status | Tip |
|--------|--------|-----|
| PR9a — fused N≤12 fp64 (extreme FWT) | **Done** | `ac3d6f13d` on `pr9-native-fused-iqp` |
| PR9b — 2-step Kronecker fused-transpose | Pending | — |
| PR9c — FP32 + default switch | Pending | — |

PR9a: `iqp_native_phase_fwt_normalize_fp64_kernel` in `iqp_tc.cu`; `launch_iqp_encode_native` N≤12 uses single kernel (no phase_split / recombine). **47/47** pytest pass (`-m "not slow"`).

## Immediate next steps

1. Open **PR8** upstream against PR7; body: `PR08_Native_FWT_Integration.md`.
2. ~~Cut `pr9-native-fused-iqp` from PR8 tip~~ ✅
3. ~~Implement 9.1 fused fp64 @ N≤12~~ ✅
4. **NCU / E2E benchmark** PR8 vs PR9a encode-only @ N=12,16 → `reports/PR009_Benchmark.md` on `main`.
5. Implement **9.2** `execute_implicit_hadamard_fp64_fused_transpose` + 2-step Kronecker.
6. PR9b smoke: N=14,16,17,18 agreement vs FWT; re-run `bench_native_wsl.sh`.

---

## References

- PR8 report: `reports/PR008_Benchmark.md` (`main`)
- PR8 handover: `PR08_HANDOVER.md`
- PR7 TC fused reference: `iqp_phase_fwt_normalize_tc_kernel` in `iqp_tc.cu`
- Native extreme FWT: `native_fp64_extreme_fwt_kernel` in `ImplicitHadamardNative.cu`
- Ozaki transpose epilogue: `implicit_hadamard_ozaki_fused_batch_kernel` in `ImplicitHadamardOzaki.cu`