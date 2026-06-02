#!/usr/bin/env bash
set -uo pipefail
export PATH="/usr/local/cuda/bin:${PATH:-}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
eval "$(bash scripts/check_gpu_compat.sh)"

BIN=build/bench_kernels
REPORTS=reports
mkdir -p "$REPORTS"

if ! command -v ncu >/dev/null 2>&1; then
  echo "ERROR: ncu not found."
  exit 1
fi

METRICS_BASE="sm__throughput.avg.pct_of_peak_sustained_elapsed,smsp__warps_active.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum,launch__shared_mem_per_block,launch__registers_per_thread"
METRICS_TC="${METRICS_BASE},sm__pipe_tensor_active.avg.pct_of_peak_sustained_active"

run_ncu() {
  local tag=$1
  local kernel=$2
  local metrics=$3
  local extra_env=${4:-}
  local bench_args=${5:-14 1}
  local csv="$REPORTS/ncu_${tag}.csv"
  local log="$REPORTS/ncu_${tag}.log"
  echo ""
  echo "=== NCU: $tag (kernel: $kernel) ==="
  (
    unset OZAKI_NCU_PROFILE
    eval "$extra_env"
    ncu --metrics "$metrics" --kernel-name "$kernel" --csv \
      "$BIN" $bench_args
  ) >"$csv" 2>"$log" || true
  if grep -q "LaunchFailed" "$csv" "$log" 2>/dev/null; then
    echo "  -> LaunchFailed"
  elif grep -qE '^"[0-9]+"' "$csv" 2>/dev/null; then
    echo "  -> OK"
  else
    echo "  -> check $csv"
  fi
}

echo "=== Environment ===" | tee "$REPORTS/ncu_run_manifest.txt"
{
  date -Iseconds 2>/dev/null || date
  echo "T4_SIMT_MODE=${T4_SIMT_MODE}"
  nvidia-smi 2>/dev/null | head -12
  nvcc --version 2>/dev/null | head -2
  ncu --version 2>/dev/null | head -2
} | tee -a "$REPORTS/ncu_run_manifest.txt"

run_ncu "fwt_baseline" "fwt_butterfly_batch_kernel" "$METRICS_BASE" "" "12 1"

if [[ "${T4_SIMT_MODE}" == "1" ]]; then
  run_ncu "simt_tc_fused" "iqp_phase_fwt_normalize_tc_kernel" "$METRICS_BASE" "" "12 1"
  for tag in phase_split modulo_precompute ozaki_grid ozaki_persistent; do
    echo "SKIP ncu_${tag} (Ozaki / N>12 path, CC>=8 only)" >"$REPORTS/ncu_${tag}.csv"
    echo "SKIPPED: CC<8 SIMT mode" >"$REPORTS/ncu_${tag}.log"
  done
else
  run_ncu "phase_split" "iqp_phase_split_kernel" "$METRICS_BASE" "" "14 1"
  run_ncu "modulo_precompute" "precompute_modulo_kernel_p26_implicit" "$METRICS_BASE" "" "14 1"
  run_ncu "ozaki_grid" "implicit_hadamard_ozaki_grid_kernel_implicit" "$METRICS_TC" "export OZAKI_NCU_PROFILE=1" "14 1"
  run_ncu "ozaki_persistent" "implicit_hadamard_ozaki_persistent_kernel_implicit" "$METRICS_TC" "" "14 1"
fi

python3 scripts/generate_ncu_summary.py