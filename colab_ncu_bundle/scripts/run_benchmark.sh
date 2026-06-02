#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
eval "$(bash scripts/check_gpu_compat.sh)"
BIN=build/bench_kernels
OUT=reports/benchmark_results.txt
mkdir -p reports

{
  echo "PR007 Benchmark Results"
  echo "Date: $(date -Iseconds 2>/dev/null || date)"
  echo "GPU: $(nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader 2>/dev/null | head -1)"
  echo "T4_SIMT_MODE=${T4_SIMT_MODE}"
  echo ""
  if [[ "${T4_SIMT_MODE}" == "1" ]]; then
    for n in 10 12; do
      echo "--- N=$n batch=128 (SIMT TC path, CC<8 compat) ---"
      "$BIN" "$n" 128 2>&1
      echo ""
    done
    echo "--- Skipped on this GPU: N=14,16 (require Ozaki Tensor Core, CC>=8) ---"
  else
    for n in 14 16; do
      echo "--- N=$n batch=128 (TC path timing in bench output) ---"
      "$BIN" "$n" 128 2>&1
      echo ""
    done
    echo "--- N=14 batch=1 (NCU workload) ---"
    OZAKI_NCU_PROFILE=1 "$BIN" 14 1 2>&1
    echo ""
    echo "--- N=14 batch=1 (production persistent Ozaki path) ---"
    unset OZAKI_NCU_PROFILE
    "$BIN" 14 1 2>&1
  fi
} | tee "$OUT"

echo "Wrote $OUT"
