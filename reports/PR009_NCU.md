# PR009 NCU / nsys — PR8 vs PR9 Native Encode

**Hardware:** RTX 4060 Laptop (CC 8.9) · WSL2
**Workload:** `scripts/ncu_profile_native_encode.py` — 32 samples, iqp-z, fp64

## Reproduce

```bash
export PATH="/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH"
source .venv_wsl/bin/activate
bash scripts/ncu_pr9_compare.sh 840f016f8 HEAD
```

Artifacts: `qdp/qdp-kernels/reports/pr9_ncu/` (local, untracked)

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
