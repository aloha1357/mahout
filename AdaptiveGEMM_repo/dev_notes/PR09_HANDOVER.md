# PR9 Handover — Fused Native IQP

**Date:** 2026-06-11  
**Code branch:** `pr9-native-fused-iqp` (base: PR8 `840f016f8`)  
**Tip commit:** `29a84d0d7` (PR9a + PR9b)
**Environment:** WSL2 Ubuntu, RTX 4060 Laptop, `.venv_wsl`

---

## Status: PR9a+PR9b landed; benchmark + PR9c next

| Item | Status |
|------|--------|
| Fused N≤12 kernel | ✅ `native_fp64_extreme_iqp_fused_kernel` (Phase + extreme FWT) |
| N>12 Kronecker 2-step | ✅ extreme FWT + fused-transpose epilogue (`transpose_batch`) |
| FP32 QML path | ⏳ PR9c |
| pytest (47 fast) | ✅ pass |
| E2E benchmark vs PR8 | ⏳ NCU + `reports/PR009_Benchmark.md` |

---

## PR9a change

`launch_iqp_encode_native` for **N = 6..12** now launches `native_fp64_extreme_iqp_fused_kernel`: Phase → **`native_fp64_extreme_fwt_transform`** on real/imag → normalize. This fuses PR8's fastest kernel, **not** the PR7 TC naive SMEM butterfly. Removes PR8's phase_split + dual extreme FWT + recombine (3+ kernels, 4 temp buffers).

**N > 12** (PR9b): 2-step Kronecker matching PR7 — `execute_implicit_hadamard_fp64(..., transpose_batch=true, batch_rows=dim*)` with fused scatter epilogue on `native_fp64_extreme_fwt_kernel`. Removed 4 explicit `iqp_tc_launch_transpose` calls.

---

## Next: benchmark + PR9c

1. NCU: compare GM bytes / kernel launches PR8 vs PR9 @ N=14,16.
2. If scatter epilogue is bandwidth-bound, tile the fused-transpose write (Ozaki 64×64 style).
3. `reports/PR009_Benchmark.md` on `main`.
4. PR9c: FP32 QML + optional default switch.