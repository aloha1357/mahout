#!/usr/bin/env bash
set -euo pipefail
export PATH="/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

echo "=== WSL build (repo: $REPO_ROOT) ==="
nvcc --version | head -3
ncu --version | head -3

mkdir -p qdp/qdp-kernels/build_wsl
SRC="qdp/qdp-kernels/src"
nvcc -O3 -std=c++17 -arch=sm_89 -I "$SRC" \
  -o qdp/qdp-kernels/build_wsl/bench_kernels \
  qdp/qdp-kernels/tests/bench_kernels.cu \
  "$SRC/iqp.cu" \
  "$SRC/iqp_tc.cu" \
  "$SRC/ImplicitHadamardOzaki.cu" \
  "$SRC/AdaptiveOzaki.cu" \
  "$SRC/amplitude.cu" \
  "$SRC/angle.cu" \
  "$SRC/basis.cu" \
  "$SRC/phase.cu" \
  "$SRC/validation.cu"

echo "=== Bench (OZAKI_NCU_PROFILE=1, N=14 batch=1) ==="
export OZAKI_NCU_PROFILE=1
./qdp/qdp-kernels/build_wsl/bench_kernels 14 1

METRICS="sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_active.avg.pct_of_peak_sustained_active,smsp__warps_active.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum,launch__shared_mem_per_block,launch__registers_per_thread"
OUTDIR="qdp/qdp-kernels/reports"
mkdir -p "$OUTDIR"

echo "=== NCU: grid Ozaki kernel ==="
export OZAKI_NCU_PROFILE=1
ncu --metrics "$METRICS" \
  --kernel-name implicit_hadamard_ozaki_grid_kernel_implicit \
  --csv \
  ./qdp/qdp-kernels/build_wsl/bench_kernels 14 1 \
  >"$OUTDIR/ncu_ozaki_grid_wsl.csv" 2>"$OUTDIR/ncu_ozaki_grid_wsl.log" || true

echo "=== NCU log tail ==="
tail -20 "$OUTDIR/ncu_ozaki_grid_wsl.log" 2>/dev/null || true
echo "=== NCU CSV (first data rows) ==="
grep -E '^"[0-9]+"' "$OUTDIR/ncu_ozaki_grid_wsl.csv" 2>/dev/null | head -8 || cat "$OUTDIR/ncu_ozaki_grid_wsl.csv" 2>/dev/null | tail -15