#!/usr/bin/env bash
# Compare PR8 vs PR9 native encode: kernel launch counts (nsys) + NCU basic metrics.
set -euo pipefail
export PATH="/usr/local/cuda/bin:/home/aloha/.cargo/bin:/usr/bin:/bin:${PATH}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${ROOT}/qdp/qdp-kernels/reports/pr9_ncu"
mkdir -p "${OUT}"
source "${ROOT}/.venv_wsl/bin/activate"

PROFILE_PY_SRC="${ROOT}/scripts/ncu_profile_native_encode.py"
cp "${PROFILE_PY_SRC}" "${OUT}/ncu_profile_native_encode.py"
PROFILE_PY="${OUT}/ncu_profile_native_encode.py"
export MAHOUT_ROOT="${ROOT}"
PR8_REF="${1:-840f016f8}"
PR9_REF="${2:-HEAD}"

run_nsys() {
  local label="$1"
  local n="$2"
  local rep="${OUT}/${label}_N${n}"
  echo "=== nsys ${label} N=${n} ==="
  nsys profile --force-overwrite=true -o "${rep}" \
    python "${PROFILE_PY}" --qubits "${n}" --samples 32 --rounds 10
  nsys stats --force-overwrite=true --report cuda_gpu_kern_sum \
    "${rep}.nsys-rep" > "${rep}_kern_sum.txt" 2>&1 || true
  grep -E "native_fp64|iqp_phase|transpose|recombine|extreme_iqp|Total" \
    "${rep}_kern_sum.txt" | head -40 || true
}

run_ncu() {
  local label="$1"
  local n="$2"
  local rep="${OUT}/${label}_N${n}_ncu"
  echo "=== ncu ${label} N=${n} ==="
  ncu --target-processes all --set basic \
    --kernel-name-base demangled \
    --kernel-name "regex:native_fp64|iqp_phase|transpose|recombine" \
    -o "${rep}" \
    python "${PROFILE_PY}" --qubits "${n}" --samples 32 --rounds 5 \
    2>&1 | tee "${rep}.log" || true
  if [[ -f "${rep}.ncu-rep" ]]; then
    ncu --import "${rep}.ncu-rep" --page details --print-summary per-kernel \
      2>&1 | tee "${rep}_summary.txt" || true
  fi
}

profile_ref() {
  local ref="$1"
  local label="$2"
  local wt="${OUT}/worktree_${label}"
  echo ""
  echo "############################################"
  echo "# Profiling ${label} @ ${ref}"
  echo "############################################"
  rm -rf "${wt}"
  git worktree add --detach "${wt}" "${ref}" >/dev/null
  cd "${wt}/qdp/qdp-python"
  maturin develop --release >/dev/null
  for N in 12 14 16; do
    run_nsys "${label}" "${N}"
  done
  run_ncu "${label}" 14
  run_ncu "${label}" 16
}

cd "${ROOT}"
profile_ref "${PR8_REF}" "PR8"
profile_ref "${PR9_REF}" "PR9"
cd "${ROOT}/qdp/qdp-python" && maturin develop --release >/dev/null

echo ""
echo "=== Launch-count diff (grep native/transpose) ==="
for N in 12 14 16; do
  echo "--- N=${N} ---"
  echo "PR8:"
  grep -E "native_fp64|iqp_phase|transpose|recombine|extreme_iqp" \
    "${OUT}/PR8_N${N}_kern_sum.txt" 2>/dev/null | head -15 || echo "(no data)"
  echo "PR9:"
  grep -E "native_fp64|iqp_phase|transpose|recombine|extreme_iqp" \
    "${OUT}/PR9_N${N}_kern_sum.txt" 2>/dev/null | head -15 || echo "(no data)"
done

echo ""
echo "Reports written to ${OUT}/"
