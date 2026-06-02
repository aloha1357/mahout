#!/usr/bin/env bash
set -euo pipefail
export PATH="/usr/local/cuda/bin:${PATH:-}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

eval "$(bash scripts/check_gpu_compat.sh)"

ARCH=$(bash scripts/detect_gpu_arch.sh)
echo "=== Detected GPU arch: $ARCH (T4_SIMT_MODE=${T4_SIMT_MODE}) ==="
nvcc --version | head -3

mkdir -p build
COMMON_SRC=(
  tests/bench_kernels.cu
  src/iqp.cu
  src/iqp_tc.cu
  src/amplitude.cu
  src/angle.cu
  src/basis.cu
  src/phase.cu
  src/validation.cu
)

if [[ "${T4_SIMT_MODE}" == "1" ]]; then
  echo "=== SIMT-compat build (no Ozaki MMA object code) ==="
  nvcc -O3 -std=c++17 -arch="${ARCH}" -I src \
    -o build/bench_kernels \
    "${COMMON_SRC[@]}" \
    src/ImplicitHadamardOzaki_stub.cu \
    -lcublas
else
  nvcc -O3 -std=c++17 -arch="${ARCH}" -I src \
    -o build/bench_kernels \
    "${COMMON_SRC[@]}" \
    src/ImplicitHadamardOzaki.cu \
    src/AdaptiveOzaki.cu \
    -lcublas
fi

echo "=== Build OK: build/bench_kernels ==="
if [[ "${T4_SIMT_MODE}" == "1" ]]; then
  ./build/bench_kernels 12 128
else
  ./build/bench_kernels 14 1
fi