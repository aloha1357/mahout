# PR9 Handover — Fused Native IQP

**Date:** 2026-06-11  
**Code branch:** `pr9-native-fused-iqp` (base: PR8 `840f016f8`)  
**Tip commit:** `ac3d6f13d` (PR9a)
**Environment:** WSL2 Ubuntu, RTX 4060 Laptop, `.venv_wsl`

---

## Status: PR9a landed; PR9b next

| Item | Status |
|------|--------|
| Fused N≤12 kernel | ✅ `native_fp64_extreme_iqp_fused_kernel` (Phase + extreme FWT) |
| N>12 Kronecker 2-step | ⏳ PR9b |
| FP32 QML path | ⏳ PR9c |
| pytest (47 fast) | ✅ pass |
| E2E benchmark vs PR8 | ⏳ pending |

---

## PR9a change

`launch_iqp_encode_native` for **N = 6..12** now launches `native_fp64_extreme_iqp_fused_kernel`: Phase → **`native_fp64_extreme_fwt_transform`** on real/imag → normalize. This fuses PR8's fastest kernel, **not** the PR7 TC naive SMEM butterfly. Removes PR8's phase_split + dual extreme FWT + recombine (3+ kernels, 4 temp buffers).

**N > 12** unchanged (4-step Kronecker + explicit transpose) until PR9b.

---

## Next: PR9b (2-step Kronecker)

1. Add `execute_implicit_hadamard_fp64_fused_transpose` to `ImplicitHadamardNative.cu` (transpose epilogue like Ozaki fused batch).
2. Replace 4-step + `iqp_tc_launch_transpose` in `iqp_native_run_kronecker_fp64`.
3. Target: match PR7 launch count (2 Hadamard calls per leg).

See `PR09_Fused_Native_Plan.md` §9.2.