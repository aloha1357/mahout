# PR9 Native IQP — N Test Matrix

**Updated:** 2026-06-11 · **Tip:** `c8fa9a1fa` (PR9d)

## Architecture tiers

| Precision | Tier | N | Path |
|-----------|------|---|------|
| fp64 | Fused IQP | 6–12 | PR9a |
| fp64 | Kronecker | >12 | PR9b fused scatter |
| fp32 | Fused IQP | 6–12 | PR9c |
| fp32 | Kronecker | >12 | PR9d scalar fused scatter |

## Measured

### fp64

| N | E2E | pytest | NCU |
|---|-----|--------|-----|
| 12 | ✅ | ✅ | — |
| 14 | ✅ | ✅ | ✅ |
| 16 | ✅ | ✅ | ✅ |
| 17–18 | — | ✅ | — |
| 20–27 | — | ❌ | ❌ target |

### fp32 (PR9d)

| N | vs fp64 | encode bench | pytest |
|---|---------|--------------|--------|
| 8, 12 | ✅ <1e-4 | ✅ | ✅ |
| 14, 16 | ✅ <1e-4 | ✅ | ✅ |
| 20–30 | — | ❌ | ❌ target |

## Targets

| Precision | Target N |
|-----------|----------|
| fp64 | **27** |
| fp32 | **30** (`MAX_QUBITS`) |

## Reproduce

```bash
export PATH="/usr/local/cuda/bin:$PATH"
source .venv_wsl/bin/activate
cd qdp/qdp-python && maturin develop --release && cd ../..

pytest testing/qdp/test_iqp_native_fp32.py -v

for N in 12 14 16; do
  python qdp/qdp-python/benchmark/benchmark_e2e.py \
    --qubits $N --samples 32 --encoding-method iqp-z \
    --frameworks mahout-native
done
```

## Related

| Doc | Content |
|-----|---------|
| `reports/PR009_Benchmark.md` | E2E + encode timings |
| `reports/PR009_NCU.md` | NCU PR8 vs PR9 |
| `reports/PR008_Benchmark.md` | PR8 baseline |