# PR011 and Current State Handover

## Environment Repair Status (2026-07-10)

The local WSL GPU environment is only partially restored.

- Ubuntu access works, but in Codex/sandbox sessions it must be launched via
  `C:\Users\aloha\AppData\Local\Microsoft\WindowsApps\ubuntu.exe`
- `.venv_wsl` was rebuilt and `torch 2.9.0+cu128` imports successfully
- GPU pytest is still blocked by `_qdp` loading a conflicting CUDA runtime
  before `torch`

See:

- `dev_notes/WSL_GPU_ENVIRONMENT_REPAIR_HANDOVER_2026_07_10.md`

Do not assume the environment is fully healthy until `python -m pytest
testing/qdp/test_bindings.py -q` passes again inside WSL.

## Current Status
We have completed the Native FP32 Fast Walsh-Hadamard Transform (FWT) pipeline implementation and benchmarked it against the highly optimized `Dao-AILab/fast-hadamard-transform` (Dao FHT).
The results and analysis are documented in `PR011_NATIVE_FP32_PIPELINE.md`.

## Accomplished
1. Benchmarked our naive FP32 global-memory butterfly implementation against `Dao FHT`.
2. Verified numeric correctness (differences are zero).
3. Identified severe memory bandwidth bottlenecks in our implementation due to global-memory accesses for every butterfly stage.
4. Explored extreme optimization plans for Dao FHT logic integration.

## Files Added/Modified
- `PR011_NATIVE_FP32_PIPELINE.md`: Contains the comprehensive benchmarking and bottleneck analysis.
- `benchmark_dao_fht.py`: Script used for benchmarking against Dao FHT.
- Various test and debug scripts: `test_tc.py`, `test_large_n.py`, `test_t1.py` etc.
- NCU profile reports within `qdp/qdp-kernels/`.

## PR2 Rework (2026-06-10) — Completed

- Branch `pr2-implicit-fwt-rework` cleanly re-cut from PR1 tip (`mahout_fork/pr1-phase-kernel-opt`).
- Removed naive O(4^N) IQP fallback; all sizes use implicit FWT (`iqp.cu`).
- Added `testing/qdp/test_implicit_fwt.py` — 12/12 passed (FP64, rtol/atol 1e-12).
- Pre-commit passed (ruff, ty, license, cargo clippy) after WSL env fix (`setup_wsl_env.sh`).
- Pushed to `mahout_fork/pr2-implicit-fwt-rework`.

See `STANDARD_PR_WORKFLOW_HANDOVER.md` for the corrected step order.

## Next Steps

- PR3: branch from PR2 tip; shared-memory FWT optimization.
- Run PR2 benchmark on `internal-dev-notes` / `main`; record speedup vs Pre-PR1 baseline.
