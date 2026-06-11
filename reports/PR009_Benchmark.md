# PR9: Fused Native Extreme-FWT IQP — Benchmark Report

**Environment:** RTX 4060 Laptop GPU (CC 8.9), WSL2 Ubuntu
**Measured:** 2026-06-11
**Branch:** `pr9-native-fused-iqp` · **Tip:** `c8fa9a1fa` (PR9d) · **Base:** PR8 `840f016f8`

## Summary

| Sub-PR | Path | N |
|--------|------|---|
| PR9a | `native_fp64_extreme_iqp_fused_kernel` | fp64 6–12 |
| PR9b | Fused-transpose Kronecker | fp64 >12 |
| PR9c | `native_fp32_extreme_iqp_fused_kernel` | fp32 6–12 |
| PR9d | FP32 scalar fused-transpose Kronecker | fp32 >12 |

**fp64:** E2E **~1.0–1.25×** vs PR8 @ N=12/14/16; **3.8–4.7×** vs PR7 TC.
**fp32:** Encode **1.7–2.5×** faster than fp64; Kronecker vs fp64 **<1e-4**.

## Unit tests

```bash
export PATH="/usr/local/cuda/bin:$PATH"
source .venv_wsl/bin/activate
cd qdp/qdp-python && maturin develop --release && cd ../..

pytest testing/qdp/test_iqp_native_path.py testing/qdp/test_iqp_tc_path.py -v
pytest testing/qdp/test_iqp_native_e2e.py -v -m "not slow"
pytest testing/qdp/test_iqp_native_fp32.py -v
```

| Suite | Result (2026-06-11) |
|-------|------------------------|
| `test_iqp_native_path.py` (29) | PASS |
| `test_iqp_tc_path.py` (12) | PASS |
| `test_iqp_native_e2e.py` fast (6) | PASS |
| `test_iqp_native_fp32.py` (4) | PASS (~46 s) |

## fp64 E2E (`qdp/qdp-python/benchmark/benchmark_e2e.py`)

Config: `iqp-z`, 32 samples, `mahout-arrow mahout-tc mahout-native`

| N | Native E2E (s) | TC (s) | Arrow (s) | Native/TC | encode_native (s) |
|---|----------------|--------|-----------|-----------|-------------------|
| 12 | **0.0087** | 0.0413 | 0.7500 | **4.7×** | 0.0022 |
| 14 | **0.0159** | 0.0612 | 1.2589 | **3.8×** | 0.0061 |
| 16 | **0.0197** | 0.0852 | 1.1997 | **4.3×** | 0.0091 |

Verification vs FWT/TC: amplitude diff **<1e-8** @ N=12/14/16.

### PR9 vs PR8 E2E

| N | PR8 (s) | PR9 (s) | Speedup |
|---|---------|---------|---------|
| 12 | 0.0080 | 0.0087 | ~1.0× (encode wins) |
| 14 | 0.0198 | **0.0159** | **1.25×** |
| 16 | 0.0197 | 0.0197 | ~1.0× |

## fp32/fp64 encode (CUDA events on `encode_batch_native`)

32 samples, 30 rounds, 5 warmup, CUDA events.

### PR9d

| N | fp32 ms/batch | fp64 ms/batch | fp32/fp64 |
|---|---------------|---------------|-----------|
| 12 | **0.289** | 0.518 | **1.8×** |
| 14 | **0.658** | 1.107 | **1.7×** |
| 16 | **1.659** | 4.188 | **2.5×** |

### PR9c vs PR9d fp32 Kronecker

| N | PR9c (float4+transpose) | PR9d (scalar fused) |
|---|-------------------------|---------------------|
| 14 | **0.370** ms | 0.658 ms |
| 16 | **1.094** ms | 1.659 ms |

## Reproduce

```bash
export PATH="/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH"
source .venv_wsl/bin/activate
cd qdp/qdp-python && maturin develop --release && cd ../..

pytest testing/qdp/test_iqp_native_fp32.py -v

for N in 12 14 16; do
  python qdp/qdp-python/benchmark/benchmark_e2e.py \
    --qubits $N --samples 32 --encoding-method iqp-z \
    --frameworks mahout-arrow mahout-tc mahout-native
done
```

## Related

- `reports/PR009_BODY.md` — PR description
- `reports/PR009_NCU.md` — NCU PR8 vs PR9
- `reports/PR009_N_Test_Matrix.md` — N coverage
- `reports/PR008_Benchmark.md` — PR8 baseline