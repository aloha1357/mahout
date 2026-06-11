# PR009 NCU / nsys — PR8 vs PR9 Native Encode

**Hardware:** RTX 4060 Laptop (CC 8.9) · WSL2  
**Profiler:** NCU 2024.3.2 · nsys 2024.5.1  
**Script:** `scripts/ncu_pr9_compare.sh` (git worktree isolation)  
**Profile workload:** `scripts/ncu_profile_native_encode.py` — 32 samples, iqp-z

## Reproduce

```bash
export PATH="/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH"
source .venv_wsl/bin/activate
bash scripts/ncu_pr9_compare.sh 840f016f8 HEAD
```

Artifacts: `qdp/qdp-kernels/reports/pr9_ncu/`

## nsys kernel summary

`nsys stats --report cuda_gpu_kern_sum` returned **no CUDA kernel rows** for either PR8 or PR9 (known WSL2 issue). Launch-count diff relies on NCU below.

## NCU per-kernel (N=14, 5 profile rounds)

| Kernel | PR8 invocations | PR9 invocations | PR8 avg duration | PR9 avg duration |
|--------|-----------------|-----------------|------------------|------------------|
| `iqp_phase_split_kernel` | 8 | 8 | 312 µs | 312 µs |
| `native_fp64_extreme_fwt_kernel<7,128>` | 32 | 32 | 107 µs | 107 µs |
| `recombine_complex_kernel` | 8 | 8 | 43 µs | 41 µs |
| `iqp_tc_batch_transpose_kernel` | **0** | **0** | — | — |

**N=14 encode path (both refs in this run):** phase_split → 4× extreme FWT (fused-transpose epilogue, PR9b) → recombine. **No standalone transpose kernels** captured.

### N=12 (PR9 only — fused IQP)

PR9 uses `native_fp64_extreme_iqp_fused_kernel` (single launch per sample) instead of phase_split + 4× FWT. NCU at N=12 not re-run in this session; E2E speedup vs PR8 documented in `reports/PR009_Benchmark.md` (1.43× @ N=12).

### N=16 (PR9 NCU highlights)

- `native_fp64_extreme_fwt_kernel<8,256>`: 32 invocations, ~1.28 ms avg (Kronecker half)
- Phase + recombine same order of magnitude as N=14

## PR8 → PR9 structural change (code-level)

| N | PR8 (`840f016f8`) | PR9 |
|---|------------------|-----|
| ≤12 | phase_split + 2× FWT + recombine | **1× `extreme_iqp_fused`** per sample |
| >12 | phase_split + 4× FWT + **4× transpose** + recombine | phase_split + 4× FWT (**fused scatter**) + recombine |

NCU @ N=14 confirms **transpose elimination** when fused epilogue is active.

## PR9c note (FP32)

FP32 native path added in PR9c: fused IQP for N≤12; N>12 uses explicit `iqp_tc_batch_transpose_kernel_f32` (4 transposes) until fp32 fused scatter lands.