# PR9 Handover — Fused Native IQP

**Date:** 2026-06-11  
**Code branch:** `pr9-native-fused-iqp` (base: PR8 `840f016f8`)  
**Tip commit:** `29a84d0d7` (PR9a + PR9b)  
**Environment:** WSL2 Ubuntu, RTX 4060 Laptop, `.venv_wsl`

---

## Status: PR9a+PR9b + benchmark complete

| Item | Status |
|------|--------|
| Fused N≤12 kernel | ✅ `native_fp64_extreme_iqp_fused_kernel` (Phase + extreme FWT) |
| N>12 Kronecker 2-step | ✅ extreme FWT + fused-transpose epilogue |
| pytest (47 fast) | ✅ pass |
| E2E benchmark report | ✅ `main` → `reports/PR009_Benchmark.md` (`8c3facdcb`) |
| FP32 QML path | ⏳ PR9c |

---

## What PR9 delivered

### PR9a (N = 6..12)

Fuses Phase + `native_fp64_extreme_fwt_transform` (real/imag) + norm in **one kernel per sample**.  
**Not** the PR7 TC naive SMEM butterfly.

### PR9b (N > 12)

2-step Kronecker with fused-transpose scatter (Ozaki-indexing parity):

```
state → temp  (H_n2 + transpose, batch_rows=dim1)
temp  → out   (H_n1 + transpose + norm, batch_rows=dim2)
```

Removed PR8's 4× `iqp_tc_launch_transpose` calls.

---

## Measurements (2026-06-11, tip `29a84d0d7`)

**Config:** `benchmark_e2e.py`, `iqp-z`, 32 samples, RTX 4060 Laptop WSL2  
**Script:** `scripts/bench_pr9_ab.sh`

### E2E total time

| N | PR9 Native | PR7 TC | PR8 Native (prior report) | PR9 vs PR8 | PR9 vs TC |
|---|------------|--------|---------------------------|------------|-----------|
| 12 | **0.0056 s** | 0.0246 s | 0.0080 s | **1.43×** | **4.4×** |
| 14 | **0.0086 s** | 0.0431 s | 0.0198 s | **2.30×** | **5.0×** |
| 16 | **0.0172 s** | 0.0738 s | 0.0197 s | **1.15×** | **4.3×** |

### Encode-only (`encode_batch_native` / `encode_batch_tc`)

| N | PR9 native encode | PR7 TC encode | PR8 native encode |
|---|-------------------|---------------|-------------------|
| 12 | **0.0021 s** | 0.0033 s | 0.0033 s |
| 14 | **0.0049 s** | 0.0053 s | 0.0052 s |
| 16 | 0.0105 s | 0.0301 s | 0.0093 s |

### Verification (E2E benchmark)

| N | vs FWT amp diff | vs TC amp diff |
|---|-----------------|----------------|
| 12 | 1.73e-8 | 0.00e+00 |
| 14 | 1.48e-8 | 4.69e-9 |
| 16 | 1.48e-8 | 1.94e-16 |

### pytest

| Suite | Result |
|-------|--------|
| `test_iqp_native_path.py` | 29/29 |
| `test_iqp_tc_path.py` | 12/12 |
| `test_iqp_native_e2e.py` (`-m "not slow"`) | 6/6 |

---

## Reproduce

```bash
wsl bash scripts/bench_pr9_ab.sh PR9-tip-29a84d0d7
```

Full write-up: [`reports/PR009_Benchmark.md`](https://github.com/aloha1357/apache_mout/blob/main/reports/PR009_Benchmark.md)

---

## Stack / remotes

- **Fork:** `mahout_fork/pr9-native-fused-iqp` @ `29a84d0d7`
- **Base:** `mahout_fork/pr8-native-hadamard-benchmark` @ `840f016f8`
- **Docs:** `internal-dev-notes` (this file + `PR09_Fused_Native_Plan.md`)
- **Benchmark:** `origin/main` → `reports/PR009_Benchmark.md`

---

## Next (PR9c / polish)

1. NCU: GM bytes + launch count PR8 vs PR9 @ N=14,16.
2. Tile coalesced scatter if bandwidth-bound.
3. Fuse `phase_split` into Kronecker step-1.
4. FP32 QML + optional default switch.