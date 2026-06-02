#!/usr/bin/env bash
# Profile all PR007 kernels; writes reports/ncu_*.csv and logs.
set -uo pipefail
export PATH="/usr/local/cuda/bin:${PATH:-}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
BIN=build/bench_kernels
REPORTS=reports
mkdir -p "$REPORTS"

if ! command -v ncu >/dev/null 2>&1; then
  echo "ERROR: ncu not found. Install CUDA / Nsight Compute or add to PATH."
  exit 1
fi

METRICS_BASE="sm__throughput.avg.pct_of_peak_sustained_elapsed,smsp__warps_active.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum,launch__shared_mem_per_block,launch__registers_per_thread"
METRICS_TC="${METRICS_BASE},sm__pipe_tensor_active.avg.pct_of_peak_sustained_active"

run_ncu() {
  local tag=$1
  local kernel=$2
  local metrics=$3
  local extra_env=${4:-}
  local csv="$REPORTS/ncu_${tag}.csv"
  local log="$REPORTS/ncu_${tag}.log"
  echo ""
  echo "=== NCU: $tag (kernel-name: $kernel) ==="
  (
    unset OZAKI_NCU_PROFILE
    eval "$extra_env"
    ncu --metrics "$metrics" \
      --kernel-name "$kernel" \
      --csv \
      "$BIN" 14 1
  ) >"$csv" 2>"$log" || true
  if grep -q "LaunchFailed" "$csv" 2>/dev/null || grep -q "LaunchFailed" "$log" 2>/dev/null; then
    echo "  -> LaunchFailed (see $log)"
  elif grep -qE '^"[0-9]+"' "$csv" 2>/dev/null && ! grep -q ',\"nan\"' "$csv" 2>/dev/null | head -1; then
    echo "  -> OK (see $csv)"
  else
    echo "  -> Check $csv / $log"
  fi
}

echo "=== Environment ===" | tee "$REPORTS/ncu_run_manifest.txt"
{
  date -Iseconds 2>/dev/null || date
  nvidia-smi 2>/dev/null | head -12
  nvcc --version 2>/dev/null | head -2
  ncu --version 2>/dev/null | head -2
} | tee -a "$REPORTS/ncu_run_manifest.txt"

# 1) Baseline FWT (global butterfly, 14 stages)
run_ncu "fwt_baseline" "fwt_butterfly_batch_kernel" "$METRICS_BASE" ""

# 2) TC pipeline side kernels
run_ncu "phase_split" "iqp_phase_split_kernel" "$METRICS_BASE" ""
run_ncu "modulo_precompute" "precompute_modulo_kernel_p26_implicit" "$METRICS_BASE" ""

# 3) Ozaki grid (profiling path)
run_ncu "ozaki_grid" "implicit_hadamard_ozaki_grid_kernel_implicit" "$METRICS_TC" "export OZAKI_NCU_PROFILE=1"

# 4) Ozaki persistent (production path, no env)
run_ncu "ozaki_persistent" "implicit_hadamard_ozaki_persistent_kernel_implicit" "$METRICS_TC" ""

echo ""
echo "=== All NCU runs finished. Generating summary ==="
python3 scripts/generate_ncu_summary.py