# PR9: Fused Native Extreme-FWT IQP (PR9a–PR9d)

## Summary

Fuses PR8's `native_fp64_extreme_fwt` into production IQP encode paths (not PR7 TC butterfly). Delivers fused IQP for N≤12, fused-transpose Kronecker for N>12 (fp64 + fp32), and FP32 QML via the same `encode_batch_native` API.

**Branch:** `pr9-native-fused-iqp` · **Tip:** `c8fa9a1fa` (PR9d) · **Base:** PR8 `840f016f8`

| Sub-PR | Change | N |
|--------|--------|---|
| PR9a | `native_fp64_extreme_iqp_fused_kernel` — phase + extreme FWT + norm in one launch | 6–12 fp64 |
| PR9b | Fused-transpose epilogue on `native_fp64_extreme_fwt`; 2-step Kronecker | >12 fp64 |
| PR9c | FP32 fused IQP + `encode_batch_native` float32/float64 dtype dispatch | ≤12 fp32 |
| PR9d | FP32 Kronecker fused-transpose (scalar extreme FWT + scatter); drops 4× transpose | >12 fp32 |

## Benchmark highlights (RTX 4060 Laptop · WSL2 · 2026-06-11)

### fp64 E2E (`benchmark_e2e.py`, 32 samples, iqp-z)

| N | Native E2E | vs TC (PR7) | vs PR8 E2E |
|---|------------|-------------|------------|
| 12 | 8.7 ms | **4.7×** | ~1.0× (encode wins) |
| 14 | 15.9 ms | **3.8×** | **1.25×** |
| 16 | 19.7 ms | **4.3×** | ~1.0× |

### fp32 GPU encode (CUDA events on `encode_batch_native`, 32 samples)

| N | fp32 (PR9d) | fp64 (PR9d) | fp32/fp64 |
|---|-------------|-------------|-----------|
| 12 | 0.29 ms/batch | 0.52 ms/batch | **1.8×** |
| 14 | 0.66 ms/batch | 1.11 ms/batch | **1.7×** |
| 16 | 1.66 ms/batch | 4.19 ms/batch | **2.5×** |

PR9d fp32 Kronecker uses scalar fused-transpose (correct vs fp64). vs PR9c float4+transpose: N=14 **1.8× slower** — follow-up: float4 fused scatter.

Full tables: `reports/PR009_Benchmark.md`

## Tests

```bash
pytest testing/qdp/test_iqp_native_path.py testing/qdp/test_iqp_tc_path.py -v
pytest testing/qdp/test_iqp_native_e2e.py -v -m "not slow"
pytest testing/qdp/test_iqp_native_fp32.py -v
```

| Suite | Cases | Result |
|-------|-------|--------|
| `test_iqp_native_path.py` | 29 | PASS |
| `test_iqp_tc_path.py` | 12 | PASS |
| `test_iqp_native_e2e.py` (fast) | 6 | PASS |
| `test_iqp_native_fp32.py` | 4 | PASS |

## Reproduce benchmarks

```bash
export PATH="/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH"
source .venv_wsl/bin/activate   # or project venv
cd qdp/qdp-python && maturin develop --release && cd ../..

pytest testing/qdp/test_iqp_native_path.py testing/qdp/test_iqp_tc_path.py -v
pytest testing/qdp/test_iqp_native_e2e.py -v -m "not slow"
pytest testing/qdp/test_iqp_native_fp32.py -v

for N in 12 14 16; do
  python qdp/qdp-python/benchmark/benchmark_e2e.py \
    --qubits $N --samples 32 --encoding-method iqp-z \
    --frameworks mahout-arrow mahout-tc mahout-native
done
```

NCU comparison (optional, local): profile `encode_batch_native` @ N=14/16 with NCU; artifacts under `qdp/qdp-kernels/reports/pr9_ncu/` (untracked). See `reports/PR009_NCU.md`.

## Stack

```
PR7 → PR8 (840f016f8) → PR9 (pr9-native-fused-iqp)
  PR9a fused fp64 N≤12
  PR9b fused-transpose fp64 Kronecker
  PR9c fp32 fused N≤12
  PR9d fp32 fused-transpose Kronecker
```

## Follow-ups

1. Float4 fused-transpose for fp32 Kronecker.
2. N sweep: fp64 → 27, fp32 → 30.
3. Optional: fuse `phase_split` into Kronecker step-1.