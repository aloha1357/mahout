# PR009 NCU / nsys — PR8 vs PR9 Native Encode

**Hardware:** RTX 4060 Laptop (CC 8.9) · WSL2
**Profiler:** NCU 2024.3.2 · nsys 2024.5.1
**Workload:** `encode_batch_native` — 32 samples, iqp-z, fp64
**Compared:** PR8 `840f016f8` vs PR9 `c8fa9a1fa` (PR9d; fp64 path unchanged since PR9b)

**Raw artifacts** (local, not in git): `qdp/qdp-kernels/reports/pr9_ncu/`

## NCU @ N=14 (fp64)

| Kernel | PR8 / PR9 invocations | Notes |
|--------|----------------------|-------|
| `iqp_phase_split_kernel` | 8 | unchanged |
| `native_fp64_extreme_fwt_kernel<7,128>` | 32 | fused-transpose epilogue |
| `recombine_complex_kernel` | 8 | unchanged |
| `iqp_tc_batch_transpose_kernel` | **0** | eliminated |

## Path summary

| N | PR8 | PR9 fp64 | PR9 fp32 (PR9d) |
|---|-----|----------|-----------------|
| ≤12 | phase_split + 2× FWT | 1× fused IQP | 1× fused IQP |
| >12 | 4× FWT + 4× transpose | 4× FWT + fused scatter | 4× scalar FWT + fused scatter |

NCU @ N=14 confirms **transpose elimination** on the fp64 Kronecker path.

## PR9d note (FP32 Kronecker)

PR9d replaces PR9c's explicit `iqp_tc_batch_transpose_kernel_f32` with scalar fused-transpose scatter (correct vs fp64). **NCU fp32 sweep pending** — see `reports/PR009_N_Test_Matrix.md`.

## Related

- E2E benchmark: `reports/PR009_Benchmark.md`
- N coverage: `reports/PR009_N_Test_Matrix.md`