#!/usr/bin/env bash
set -euo pipefail
export PATH="/home/aloha/.cargo/bin:/usr/local/cuda/bin:${PATH}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "${ROOT}/.venv_wsl/bin/activate"
cd "${ROOT}/qdp/qdp-python/benchmark"
for N in 12 14 16; do
  echo "========== E2E N=${N} =========="
  python benchmark_e2e.py --qubits "${N}" --samples 32 --encoding-method iqp-z \
    --frameworks mahout-arrow mahout-tc mahout-native
done