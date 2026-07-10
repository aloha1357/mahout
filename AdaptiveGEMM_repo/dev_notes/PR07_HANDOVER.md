# PR7 Handover ??E2E TC Integration (N>12 Correctness Complete)

**Date:** 2026-06-11
**Code branch:** `pr7-iqp-tc-ncu-profiling` (base: PR6 `543586ac1`)
**Tip commit:** `ac7883296` — dev benchmark cleanup atop `1e7459a6e`
**Environment:** WSL2 Ubuntu, RTX 4060 Laptop, `.venv_wsl`

---

## Status: Ready for upstream PR review

| Item | Status |
|------|--------|
| N??2 correctness + speed | ??|
| N=14??8 Kronecker correctness | ??(~1e-17 vs FWT) |
| E2E Mahout-Arrow vs Mahout-TC | ??N=12/14/16 verified |
| Unit tests | ??`pytest testing/qdp/test_iqp_tc_path.py` ??**12/12** |
| pre-commit (4 code files) | ??Python + CUDA license |
| Code branch has no `.md` | ??docs here + `main` only |

---

## What PR7 delivers (789 merged scope)

1. **Stream / error handling:** `device.cu_stream()` in `iqp.rs`; `CHECK_CUDA_RETURN` in `iqp_tc.cu`
2. **Ozaki kernel:** removed PR6 persistent MMA; fused INT8 TC for `n ??{64,128}`; naive FP64 + `transpose_batch` for `n ??256`
3. **API:** `encode_batch_tc(..., encoding_method="iqp"|"iqp-z")`
4. **E2E benchmark:** `benchmark_e2e.py` Mahout-TC path + verification; DLPack `.clone()` fix
5. **Tests:** N=8,12 normalized; N=14,16,17,18 FWT vs TC agreement

---

## Tensor Core dispatch (current policy)

| N (qubits) | Kronecker dims | Hadamard path | TC? |
|------------|----------------|---------------|-----|
| ??2 | ??| `iqp_phase_fwt_normalize_tc_kernel` | ??|
| 14 | 128?128 | fused Ozaki MMA | ??|
| 16 | 256?256 | naive FP64 + transpose | ??(stable; E2E still wins) |
| 17 | 256?512 | mixed (naive / fused by leg) | partial |
| 18 | 512?512 | naive FP64 | ??|

**Note:** n=256 fused TC was tested ??non-deterministic on WDDM and not faster than naive. Hybrid policy kept.

---

## Verified benchmarks (2026-06-11)

### Unit tests

```bash
pytest testing/qdp/test_iqp_tc_path.py -v   # 12 passed
```

### Encode agreement (FWT vs `encode_batch_tc`, float64)

| N | max_err |
|---|---------|
| 14 | ~1e-9 |
| 15 | ~1e-17 |
| 16 | ~1e-17 |
| 17 | ~1e-17 |
| 18 | ~1e-17 |

### E2E (`benchmark_e2e.py`, `iqp-z`, 32 samples)

| N | Mahout-TC | Mahout-Arrow FWT | Speedup | Verification |
|---|-----------|------------------|---------|--------------|
| 12 | ~0.05 s | ~0.83 s | ~16? | PASS ~1.7e-8 |
| 14 | ~0.05 s | ~1.1 s | ~22? | PASS ~1.5e-8 |
| 16 | **0.057 s** | **0.827 s** | **~14.5?** | **PASS ~1.5e-8** |

### (removed from code branches �X use `pytest` + optional local profiling)

N=16: FWT ~10 ms, TC ~32 ms (encode-only TC slower; **E2E is the metric**).

---

## Reproduce

```bash
export PATH=/usr/local/cuda/bin:$PATH
source .venv_wsl/bin/activate
cd qdp/qdp-python && maturin develop --release && cd ../..
pytest testing/qdp/test_iqp_tc_path.py -v
python qdp/qdp-python/benchmark/benchmark_e2e.py --qubits 14 16 --samples 32 --encoding-method iqp-z
```

---

## PR stack / remotes

- **Fork PR branch:** `mahout_fork/pr7-iqp-tc-ncu-profiling` — **pushed** (`ac7883296`, 2026-06-11)
- **Docs:** `internal-dev-notes` (this file + `PR07_E2E_TC_Integration.md`)
- **Benchmark report:** `origin/main` ??`reports/PR007_Benchmark.md`
- **Upstream base:** `mahout_fork/pr6-tensor-core-acceleration`

---

## Known limitations / next work

1. **n=256 fused TC:** fix WDDM non-determinism before re-enabling (optional perf win)
2. **Encode-only N>12:** still memory-bound vs FWT; acceptable for PR7 scope
3. **Upstream PR:** open `aloha1357/mahout` → `apache/mahout` when ready (PR body: `PR07_E2E_TC_Integration.md`)
4. **NCU / profiling artifacts:** stay on `main` / `internal-dev-notes` only, not code PR
5. **PR8 / PR9:** see `PR08_PR09_Native_Fused_Plan.md` — wire `ImplicitHadamardNative.cu`, A/B vs PR7, then optional fused native kernel

---

## Files changed in code PR (vs PR6)

| File | Role |
|------|------|
| `ImplicitHadamardOzaki.cu/.h` | fused TC + naive transpose |
| `iqp_tc.cu` | Kronecker 2-step, buffers, sync |
| `iqp.rs`, `lib.rs`, `engine.rs`, `backend.py` | stream + API |
| `benchmark_e2e.py` | Mahout-TC + verify + DLPack clone |
| `test_iqp_tc_path.py` | N=14??8 tests |
