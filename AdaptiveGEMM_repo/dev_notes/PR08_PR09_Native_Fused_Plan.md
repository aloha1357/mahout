# PR8 / PR9 Plan — Native Hadamard vs PR7 Ozaki-TC

**Date:** 2026-06-11 (updated after PR8 benchmark)
**PR7 baseline:** `ac7883296`
**PR8 tip:** `d3bcf7d37` on `pr8-native-hadamard-benchmark` (pushed to `mahout_fork`)
**PR9 plan:** see `PR09_Fused_Native_Plan.md`

---

## PR8 result — GO for PR9

| N | PR8 Native E2E | PR7 TC | Native/TC | Verify |
|---|----------------|--------|-----------|--------|
| 12 | 0.008 s | 0.086 s | **10.7×** | PASS |
| 14 | 0.020 s | 0.079 s | **4.0×** | PASS |
| 16 | 0.020 s | 0.063 s | **3.2×** | PASS |

- `pytest test_iqp_native_path.py`: **14/14**
- Report: `reports/PR008_Benchmark.md` on `main`
- PR8 exceeds §8.5 success gate (≥1.1× E2E vs PR7)

---

## Goal

Answer one question before changing the default IQP path:

> Can **Native FWT** (Tri Dao–style register/SMEM butterfly, no Ozaki quantization) beat PR7’s already-tuned Ozaki-TC + fused SMEM path **end-to-end**, at equal or acceptable numerical cost?

PR8 wires the prototype and runs controlled A/B benchmarks. PR9 lands only if benchmarks justify it: a **fully fused** Phase + Native-FWT + Normalize kernel (and optional Kronecker native legs for N>12).

---

## PR7 baseline (what we beat)

| Layer | N ≤ 12 | N > 12 (Kronecker) |
|-------|--------|---------------------|
| Phase + FWT + norm | `iqp_phase_fwt_normalize_tc_kernel` (fused SMEM, FP64 complex) | separate phase kernel → Ozaki Hadamard legs |
| Hadamard engine | in-kernel butterfly | `ImplicitHadamardOzakiEngine`: INT8 fused TC for n∈{64,128}; naive FP64 + `transpose_batch` for n≥256 |
| E2E win | ~16× vs Arrow-FWT @ N=12 | ~14–22× @ N=14–16 (`iqp-z`, 32 samples) |
| Encode-only N>12 | competitive | **slower** than FWT (memory-bound); E2E is the metric |

PR7 encode-only @ N=16: FWT ~10 ms vs TC ~32 ms. Any native win must show up in **E2E**, not micro-bench alone.

---

## ImplicitHadamardNative — what the prototype provides

`ImplicitHadamardNativeEngine` bypasses Ozaki entirely. Real/complex IQP still needs phase + normalize wrappers; native only replaces the **Hadamard matvec** leg.

### FP32 (`execute_implicit_hadamard_fp32`)

- **N = 7…15:** template `native_fp32_extreme_fwt_kernel<N, THREADS, NUM_B>` — warp shuffle + SMEM, float4 loads, optional 64 KiB dynamic SMEM @ N=14.
- **N ≥ 16:** multi-pass: first pass up to 15 qubits in-chunk, then `interblock_shuffle` (≤5 qubits) or `interblock_fwt_v2` (6–8 qubits) per pass.

### FP64 (`execute_implicit_hadamard_fp64`)

- **N = 6…14:** `native_fp64_extreme_fwt_kernel` — 64 doubles/thread, 32 KiB SMEM.
- **N ≥ 15:** multi-pass: extreme kernel on first chunk, `native_fp64_interblock_fwt_kernel` for remaining qubits.

### Design intent (from header)

Optimized for **QML / speed** when FP64 stability is not required. FP64 native path is a candidate to replace Ozaki **naive FP64** legs (n=256+) without INT8 quantization overhead.

### Not in prototype yet

- No `iqp_native_phase_fwt_normalize_kernel` (full fusion).
- No Kronecker orchestration in `iqp_tc.cu` calling Native instead of Ozaki.
- `transpose_batch` / `batch_rows` parameters exist in API but dispatch is primarily direct FWT on `[m × n]` layouts — must be validated for Kronecker tile geometry before PR9.

---

## PR8 — Native wiring + benchmark (no default switch)

**Branch:** `pr8-native-hadamard-benchmark` (base: PR7 tip)
**Merge criterion:** build green + benchmark report with go/no-go per N; **no** change to default `encode_batch_tc` behavior.

### 8.1 Build integration

| Task | File(s) |
|------|---------|
| Add `.cu/.h` to nvcc compile | `qdp/qdp-kernels/build.rs` — `.file("src/ImplicitHadamardNative.cu")`, `rerun-if-changed` |
| Smoke launch | minimal `extern "C"` wrapper or gtest-style CUDA test in `testing/qdp/` |

### 8.2 Benchmark harness (dev-only on PR8 branch)

Extend `benchmark_e2e.py` **or** add `benchmark_native_ab.py` (removed before upstream merge if ASF policy requires):

```
--backend arrow-fwt | mahout-tc | mahout-native-fp32 | mahout-native-fp64
--qubits 12 14 16 18
--samples 32
--encoding-method iqp-z
--verify   # max |amp_tc - amp_ref| vs Arrow-FWT
```

**Secondary (encode-only):** isolate Hadamard leg timing inside `launch_iqp_encode_tc` with NVTX ranges or a small `benchmark_hadamard_leg.py` calling Native vs Ozaki on synthetic `[batch × tile]` buffers.

### 8.3 Wiring options (pick one in PR8)

| Option | Pros | Cons |
|--------|------|------|
| **A.** Env flag `QDP_IQP_BACKEND=native|ozaki` inside `iqp_tc.cu` | zero Python API churn | hidden, harder to test in CI |
| **B.** `encode_batch_native(...)` PyO3 sibling | explicit, pytest-friendly | more surface area |
| **C.** `encoding_method="iqp-z-native"` | fits existing API | stringly-typed |

**Recommendation:** Option **B** for PR8 benchmark branch; fold into Option C only if E2E wins justify PR9 default.

### 8.4 Correctness gates

| Path | Reference | Tolerance |
|------|-----------|-----------|
| Native FP64 | Arrow / `encode_batch` FWT | max err ≤ **1e-8** (match PR7 TC tests) |
| Native FP32 | FP64 reference | max err ≤ **1e-4** @ N≤12; document drift @ N≥16 |
| E2E verify | same as PR7 | PASS if ≤ **1e-5** amplitude diff (practical training) |

Add pytest: `test_iqp_native_path.py` mirroring `test_iqp_tc_path.py` for N=8,12,14,16,18.

### 8.5 Benchmark matrix (RTX 4060 WSL2, primary = E2E)

| N | Compare |
|---|---------|
| 12 | PR7 fused TC vs Native FP32 extreme vs Native FP64 vs Arrow-FWT |
| 14 | PR7 Kronecker (128² Ozaki TC) vs Native multi-pass FP32/FP64 |
| 16 | PR7 Kronecker (256² naive FP64) vs Native FP64 multi-pass |
| 18 | PR7 naive legs vs Native FP64 3-pass |

**Success (proceed to PR9):** Native path ≥ **1.1× E2E speedup** vs PR7 at any target N **and** verification PASS.
**Partial success:** Native wins encode-only Hadamard leg ≥ **1.5×** @ N≥14 → PR9 focuses on fusion + less GM traffic.
**No-go:** Native slower or FP32 fails verify → keep PR7; archive native as QML-only opt-in.

### 8.6 Deliverables

- [ ] `ImplicitHadamardNative.cu/.h` on code branch
- [ ] `build.rs` + smoke test
- [ ] A/B harness + local numbers
- [ ] `reports/PR008_Benchmark.md` on `main` (after merge)
- [ ] This plan updated with measured table

---

## PR9 — Full fused native kernel (conditional on PR8)

**Branch:** `pr9-native-fused-iqp` (base: PR8)
**Only if PR8 shows a credible path to beat PR7 E2E or Hadamard-leg bottleneck.**

### 9.1 Fused kernel (N ≤ 15)

Mirror `iqp_phase_fwt_normalize_tc_kernel`:

```
iqp_native_phase_fwt_normalize_fp32_kernel   # QML fast path
iqp_native_phase_fwt_normalize_fp64_kernel   # precision path
```

- Phase → registers/SMEM (reuse `compute_phase_tc`)
- In-place native extreme FWT (reuse templates from Native.cu)
- Normalize + write `cuComplex` / `cuDoubleComplex` global state
- **Target:** beat PR7 @ N≤12 encode leg; match PR7 E2E @ N=12–14

### 9.2 Kronecker native (N > 12)

Replace Ozaki legs in `launch_iqp_encode_tc`:

```
Phase kernel (global) → Native execute_implicit_hadamard_* (per leg, transpose_batch=true)
```

Policy draft:

| n (leg size) | PR7 today | PR9 candidate |
|--------------|-----------|---------------|
| 64, 128 | Ozaki INT8 fused TC | Native FP32 if PR8 wins; else keep Ozaki |
| 256+ | naive FP64 Ozaki | **Native FP64** multi-pass (no INT8 quant overhead) |

Re-test n=256 fused TC only if WDDM determinism fix lands; PR7 kept naive for stability.

### 9.3 API / product

- Default stays `encode_batch_tc` until PR9 benchmarks confirm regression-free win.
- Optional: `precision="fp32"|"fp64"` on engine for QML workflows.
- Document numerical trade-off in PR body (FP32 not for all quantum chemistry use cases).

### 9.4 Tests & CI

- Extend `test_iqp_tc_path.py` or merge into unified `test_iqp_encode_paths.py`
- pre-commit on new `.cu` files
- E2E `benchmark_e2e.py` `--backend mahout-native` as release gate

---

## Branch / PR stack

```
PR6 (mahout_fork/pr6-tensor-core-acceleration)
  └── PR7 pr7-iqp-tc-ncu-profiling  ← ac7883296 PUSHED
        └── PR8 pr8-native-hadamard-benchmark  ← wire Native + A/B only
              └── PR9 pr9-native-fused-iqp     ← fusion + dispatch switch (if justified)
```

| PR | Repo branch | Docs |
|----|-------------|------|
| PR7 | `mahout_fork` code | `PR07_*`, `reports/PR007_Benchmark.md` |
| PR8–9 | `mahout_fork` code | this file + `reports/PR008_*.md` / `PR009_*.md` on `main` |

---

## Immediate next steps (execution order)

1. ~~PR8 wire + benchmark~~ **DONE** (`d3bcf7d37`)
2. **Open upstream PR8** against PR7; body: `PR08_Native_FWT_Integration.md`
3. **Cut `pr9-native-fused-iqp`** from PR8 tip — see `PR09_Fused_Native_Plan.md`
4. **PR9a:** fused `iqp_native_phase_fwt_normalize_fp64_kernel` for N≤15
5. **PR9b:** native `execute_implicit_hadamard_fp64_fused_transpose` (2-step Kronecker)
6. **PR9c:** optional FP32 + default API switch

---

## Risk register

| Risk | Mitigation |
|------|------------|
| FP32 insufficient for IQP training | gate PR9 default on FP64 path; FP32 opt-in only |
| Native multi-pass extra GM passes | PR9 fusion + compare bytes moved vs Ozaki Kronecker |
| `transpose_batch` semantics mismatch | unit test single 128×128 leg vs Ozaki before E2E |
| WDDM timing noise | same warmup/iter protocol as PR7; report median of 5 |
| Code branch bloat | dev benchmarks live on PR8 branch only; reports on `main` |

---

## References

- PR7 handover: `PR07_HANDOVER.md`
- PR7 benchmark: `reports/PR007_Benchmark.md` (`main`)
- Prototype: `ImplicitHadamardNative.cu` (Tri Dao–style `native_fp32_extreme_fwt_kernel`, FP64 `native_fp64_extreme_fwt_kernel`)
- Current fused reference: `iqp_tc.cu` → `iqp_phase_fwt_normalize_tc_kernel`
