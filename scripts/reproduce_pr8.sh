#!/usr/bin/env bash
set -euo pipefail
export PATH="/home/aloha/.cargo/bin:/usr/local/cuda/bin:${PATH}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "${ROOT}/.venv_wsl/bin/activate"
echo "=== [1/4] Build ==="
cd "${ROOT}/qdp/qdp-python" && maturin develop --release
cd "${ROOT}"
echo "=== [2/4] Unit tests ==="
pytest testing/qdp/test_iqp_native_path.py testing/qdp/test_iqp_tc_path.py -v --tb=short
echo "=== [3/4] E2E pytest ==="
pytest testing/qdp/test_iqp_native_e2e.py -v --tb=short -m "not slow"
echo "=== [4/4] Benchmark A/B ==="
cd "${ROOT}/qdp/qdp-python/benchmark"
for N in 12 14 16; do
  echo "--- N=${N} ---"
  python benchmark_e2e.py --qubits "${N}" --samples 32 --encoding-method iqp-z \
    --frameworks mahout-arrow mahout-tc mahout-native
done
echo "=== PR8 reproduction complete ==="
