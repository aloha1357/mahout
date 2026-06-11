#!/usr/bin/env bash
set -euo pipefail
export PATH="/home/aloha/.cargo/bin:/usr/local/cuda/bin:${PATH}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "${ROOT}/.venv_wsl/bin/activate"
cd "${ROOT}"
pytest testing/qdp/test_iqp_native_path.py -v --tb=short