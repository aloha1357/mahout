#!/usr/bin/env bash
# Source this before running pre-commit, maturin, or pytest in WSL.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export UV_PROJECT_ENVIRONMENT="${REPO}/.venv_wsl"
export VIRTUAL_ENV="${REPO}/.venv_wsl"

NV_LIBS="$(find "${REPO}/.venv_wsl/lib/python3.12/site-packages/nvidia" -name lib -type d 2>/dev/null | paste -sd:)"
TORCH_LIB="${REPO}/.venv_wsl/lib/python3.12/site-packages/torch/lib"
export LD_LIBRARY_PATH="${NV_LIBS}:${TORCH_LIB}:${LD_LIBRARY_PATH:-}"
export LIBTORCH_USE_PYTORCH=1
export PATH="/usr/local/cuda/bin:${REPO}/.venv_wsl/bin:${PATH}"

if [[ ! -e "${REPO}/.venv/pyvenv.cfg" ]]; then
  ln -sfn .venv_wsl "${REPO}/.venv"
fi