# WSL GPU Environment Repair Handover (2026-07-10)

## Purpose

This note records the current WSL GPU test breakage, the verified root cause,
the exact directories and branches involved, and the repair plan for the next
person who needs to restore the local QDP GPU test environment.

This document belongs on `internal-dev-notes`.

## Current status

The WSL Python environment was rebuilt and is now partially healthy:

- Ubuntu is reachable from this machine.
- `.venv_wsl` exists at repo root.
- `python`, `uv`, and `torch` import successfully inside WSL.
- `torch.__version__ == 2.9.0+cu128`
- `torch.cuda.is_available() == True`

However, GPU pytest is still broken for QDP binding tests because `_qdp` and
PyTorch do not currently share a compatible CUDA runtime load order.

## Exact failure

This fails:

```bash
python -m pytest testing/qdp/test_bindings.py -q
```

Observed error:

```text
ImportError: .../torch/lib/libc10_cuda.so: undefined symbol:
cudaGetDriverEntryPointByVersion, version libcudart.so.12
```

## Verified root cause

The issue is not "PyTorch is missing" anymore. The issue is dynamic library
resolution order.

What we verified:

1. `import torch` alone succeeds in WSL.
2. `import pytest` then `import torch` also succeeds.
3. `import _qdp` first, then `import torch`, fails.
4. `pytest` fails because `testing/conftest.py` imports `_qdp` at startup.
5. The `_qdp` native extension has an ELF `RPATH` that prefers
   `/usr/local/cuda/lib64`.

That means:

- `_qdp` pulls in the system CUDA runtime first.
- `torch` later tries to use its wheel-compatible CUDA runtime.
- both land in the same process and symbol resolution breaks.

## Important files

Start here:

- `testing/conftest.py`
- `testing/qdp/test_bindings.py`
- `testing/qdp/qdp_test_utils.py`
- `qdp/qdp-python/Cargo.toml`
- `qdp/qdp-python/build.rs` if linkage behavior needs inspection
- the built extension:
  `.venv_wsl/lib/python3.12/site-packages/_qdp/_qdp.cpython-312-x86_64-linux-gnu.so`

## Important branches

- `internal-dev-notes`
  - documentation and handover branch
  - this file should stay here
- `pr2-batch-throughput-opt`
  - current PR branch that needs validation
- `dev_iqp_tc_ci_fix`
  - local branch with a separate test-related commit
  - do not lose it without checking whether the change is still needed

## Repo root and WSL path

Windows repo root:

```text
D:\D_backup\2025\tum\26S\apache_mout
```

WSL repo root:

```text
/mnt/d/D_backup/2025/tum/26S/apache_mout
```

## WSL entrypoint

In this Codex/sandbox setup, `wsl.exe` may run under the wrong Windows user and
report no default distro even though Ubuntu exists for the interactive user.

Use this entrypoint instead:

```powershell
C:\Users\aloha\AppData\Local\Microsoft\WindowsApps\ubuntu.exe
```

Sanity check:

```powershell
& 'C:\Users\aloha\AppData\Local\Microsoft\WindowsApps\ubuntu.exe' run bash -lc 'uname -a'
```

## Known-good environment bootstrap

Run repo commands through Ubuntu and set env inline:

```bash
cd /mnt/d/D_backup/2025/tum/26S/apache_mout
export UV_PROJECT_ENVIRONMENT="$PWD/.venv_wsl"
export VIRTUAL_ENV="$PWD/.venv_wsl"
export LIBTORCH_USE_PYTORCH=1
export PATH="$PWD/.venv_wsl/bin:/usr/local/cuda/bin:$PATH"
NV_LIBS=$(find "$PWD/.venv_wsl/lib/python3.12/site-packages/nvidia" -maxdepth 2 -type d -name lib | paste -sd: -)
TORCH_LIB="$PWD/.venv_wsl/lib/python3.12/site-packages/torch/lib"
export LD_LIBRARY_PATH="$NV_LIBS:$TORCH_LIB:${LD_LIBRARY_PATH:-}"
```

Quick health check:

```bash
.venv_wsl/bin/python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
PY
```

Expected:

```text
2.9.0+cu128
True
12.8
```

## Commands used to prove the current bug

### 1. PyTorch alone is healthy

```bash
.venv_wsl/bin/python - <<'PY'
import torch
print("torch ok", torch.__version__, torch.cuda.is_available(), torch.version.cuda)
PY
```

### 2. `_qdp` before `torch` reproduces the conflict

```bash
.venv_wsl/bin/python - <<'PY'
import _qdp
import torch
PY
```

Expected failure:

```text
ImportError: ...libc10_cuda.so: undefined symbol:
cudaGetDriverEntryPointByVersion, version libcudart.so.12
```

### 3. Inspect `_qdp` linkage

```bash
ldd .venv_wsl/lib/python3.12/site-packages/_qdp/_qdp.cpython-312-x86_64-linux-gnu.so
readelf -d .venv_wsl/lib/python3.12/site-packages/_qdp/_qdp.cpython-312-x86_64-linux-gnu.so | grep -E 'RPATH|RUNPATH|NEEDED'
```

Observed key result:

```text
RPATH: [/usr/local/cuda/lib64:.../qdp/target/.../out]
NEEDED: libcudart.so.12
```

## Repair strategy

There are two layers here. The first restores testability. The second fixes the
runtime behavior correctly.

### Layer 1: restore GPU pytest without guessing

Goal:

- stop `pytest` from loading `_qdp` before `torch`
- confirm that tests collect and start with a stable runtime order

Likely first change:

- make `testing/conftest.py` lazy
- do not import `_qdp` at module import time
- move `_qdp` availability checks into fixtures or hook functions that do not
  poison the process before test modules import `torch`

Validation:

```bash
python -m pytest testing/qdp/test_bindings.py -q
```

If this works, we regain the local GPU test loop even before fixing the binary.

### Layer 2: fix `_qdp` runtime linkage

Goal:

- stop the `_qdp` extension from forcing `/usr/local/cuda/lib64` ahead of the
  CUDA wheel libraries used by PyTorch

What to inspect:

- maturin build output
- cargo/rust build scripts
- any `RPATH`/`RUNPATH` flags
- whether `_qdp` should rely on PyTorch CUDA libs instead of system CUDA first

Possible directions:

1. remove or relax `_qdp` `RPATH`
2. use `RUNPATH` instead of hard `RPATH` if appropriate
3. align the extension with the same CUDA runtime source expected by torch
4. patch the built `.so` only as a diagnostic step, not as the final workflow

Validation:

```bash
.venv_wsl/bin/python - <<'PY'
import _qdp
import torch
print("both ok")
PY
python -m pytest testing/qdp/test_bindings.py -q
```

## Recommended work order

1. Stay on `internal-dev-notes` while documenting.
2. Switch to the code branch only when making the actual fix.
3. Reproduce with the commands above before changing anything.
4. Try the lazy-import `testing/conftest.py` fix first.
5. If the bug remains, inspect and change `_qdp` linkage.
6. Only after GPU pytest is stable, return to PR validation work.

## Commands to switch branches cleanly

```bash
git checkout internal-dev-notes
git status --short

git checkout pr2-batch-throughput-opt
git status --short
```

If local diagnostics are still untracked, do not carry them into the PR branch.
Either delete them or keep them on `internal-dev-notes` only if they are worth
turning into permanent docs or helper scripts.

## Temporary local diagnostics currently present

These were created during repair and should not be pushed blindly:

- `tmp_wsl_postsync_check.sh`
- `tmp_wsl_cuda_link_check.sh`
- `tmp_wsl_run_bindings.sh`
- `tmp_wsl_pytest_torch_probe.sh`
- `tmp_wsl_module_probe.py`
- `tmp_wsl_qdp_then_torch.py`
- `tmp_wsl_readelf_qdp.sh`

Keep only what becomes a deliberate maintainer tool. Delete the rest.

## Handover summary

The environment is no longer "missing Ubuntu" or "missing torch". The current
problem is narrower and more actionable:

- WSL access works via the Ubuntu WindowsApps alias
- `.venv_wsl` is rebuilt
- PyTorch CUDA works by itself
- `_qdp` currently forces a conflicting CUDA runtime into the process
- `pytest` triggers the failure early through `testing/conftest.py`

The next person should start from the commands in this file, not from a fresh
reinstall.
