# PR8 Handover — Native FWT A/B (beats PR7 E2E)

**Date:** 2026-06-11
**Code branch:** `pr8-native-hadamard-benchmark` (base: PR7 `ac7883296`)
**Tip commit:** `840f016f8`
**Environment:** WSL2 Ubuntu, RTX 4060 Laptop, `.venv_wsl`

---

## Status: Ready for upstream PR review (base PR7)

| Item | Status |
|------|--------|
| Native wired + builds | ✅ |
| Kronecker N>12 correctness | ✅ (4-step + transpose) |
| FP64 N=7 thread dispatch fix | ✅ |
| Unit tests | ✅ `test_iqp_native_path.py` **29/29** |
| E2E pytest | ✅ `test_iqp_native_e2e.py` **6/6** (`-m "not slow"`) |
| PR7 regression | ✅ `test_iqp_tc_path.py` **12/12** |
| pre-commit | ✅ ruff, license, clippy (`SKIP=ty`) |
| E2E vs PR7 | ✅ **3.2–10.7×** faster @ N=12,14,16 |
| Benchmark report | ✅ `main` → `reports/PR008_Benchmark.md` |
| PR9 justified | ✅ proceed to fused kernel |

---

## What PR8 delivers

1. **`ImplicitHadamardNative.cu/.h`** in nvcc build
2. **`launch_iqp_encode_native`** + **`encode_batch_native`** API
3. **`benchmark_e2e.py`** `--frameworks mahout-native`
4. **`test_iqp_native_path.py`** correctness suite
5. **Dev scripts:** `scripts/build_wsl.sh`, `test_native_wsl.sh`, `bench_native_wsl.sh`

---

## E2E numbers (2026-06-11, `iqp-z`, 32 samples)

| N | Native | PR7 TC | Arrow FWT | Native/TC |
|---|--------|--------|-----------|-----------|
| 12 | 0.008 s | 0.086 s | 1.44 s | **10.7×** |
| 14 | 0.020 s | 0.079 s | 1.26 s | **4.0×** |
| 16 | 0.020 s | 0.063 s | 0.83 s | **3.2×** |

Verification: Native vs FWT and Native vs TC **PASS** at all three N.

---

## Reproduce (one command)

**Branch:** `pr8-native-hadamard-benchmark` tip `840f016f8`

```bash
wsl bash scripts/reproduce_pr8.sh
```

| Step | What |
|------|------|
| 1 | `maturin develop --release` |
| 2 | `pytest test_iqp_native_path.py` + `test_iqp_tc_path.py` (**41** tests) |
| 3 | `pytest test_iqp_native_e2e.py` (encode → DLPack → forward) |
| 4 | `benchmark_e2e.py` A/B @ N=12,14,16 |

Individual scripts: `build_wsl.sh`, `test_native_wsl.sh`, `bench_native_wsl.sh`

---

## PR stack / remotes

- **Fork PR branch:** `mahout_fork/pr8-native-hadamard-benchmark`
- **Base:** `mahout_fork/pr7-iqp-tc-ncu-profiling`
- **Docs:** `internal-dev-notes` (this file + `PR08_Native_FWT_Integration.md`)
- **Benchmark:** `origin/main` → `reports/PR008_Benchmark.md`
- **Next:** `PR09_Fused_Native_Plan.md`

---

## Known limitations → PR9 scope

1. N≤12: no fused Phase+FWT+Norm (extra buffers vs PR7 fused SMEM TC)
2. N>12: 4-step Kronecker (PR7 uses 2-step fused-transpose Ozaki)
3. `transpose_batch` unused in Native engine — needs fused epilogue in PR9
4. Native `.cu` files need ASF license headers before upstream merge
5. Default path still `encode_batch_tc`; native is opt-in for benchmark
