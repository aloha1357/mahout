#!/usr/bin/env bash
set -euo pipefail
export PATH="/usr/local/cuda/bin:${PATH:-}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

bash scripts/check_gpu_compat.sh || exit 1

ARCH=$(bash scripts/detect_gpu_arch.sh)
echo "=== Detected GPU arch: $ARCH ==="
nvcc --version | head -3

mkdir -p build
SRC_FILES=(
  tests/bench_kernels.cu
  src/iqp.cu
  src/iqp_tc.cu
  src/ImplicitHadamardOzaki.cu
  src/AdaptiveOzaki.cu
  src/amplitude.cu
  src/angle.cu
  src/basis.cu
  src/phase.cu
  src/validation.cu
)

nvcc -O3 -std=c++17 -arch="${ARCH}" -I src \
  -o build/bench_kernels \
  "${SRC_FILES[@]}" \
  -lcublas

echo "=== Build OK: build/bench_kernels ==="
./build/bench_kernels 14 1