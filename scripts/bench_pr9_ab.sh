#!/usr/bin/env bash
set -euo pipefail
export PATH="/home/aloha/.cargo/bin:/usr/local/cuda/bin:/usr/bin:/bin:${PATH}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "${ROOT}/.venv_wsl/bin/activate"
cd "${ROOT}/qdp/qdp-python/benchmark"

LABEL="${1:-PR9}"
echo "=== Benchmark A/B: ${LABEL} ==="
echo "git: $(cd "${ROOT}" && git log --oneline -1)"

for N in 12 14 16; do
  echo ""
  echo "########## N=${N} ##########"
  python benchmark_e2e.py \
    --qubits "${N}" \
    --samples 32 \
    --encoding-method iqp-z \
    --frameworks mahout-arrow mahout-tc mahout-native 2>&1 \
    | grep -E "^(  |Mahout-|E2E|Samples:|={3,}|encode_batch|Total Time|Arrow read|VERIFY|\[Mahout|Max Amplitude|>> SUCCESS)"
done

echo ""
echo "=== fp32/fp64 encode GPU timing ==="
cd "${ROOT}"
python scripts/benchmark_iqp_native.py --label "${LABEL}"

echo ""
echo "=== pytest (fast) ==="
pytest testing/qdp/test_iqp_native_path.py testing/qdp/test_iqp_tc_path.py \
  testing/qdp/test_iqp_native_e2e.py testing/qdp/test_iqp_native_fp32.py \
  -m "not slow" -q --tb=no 2>&1 | tail -5
