#!/usr/bin/env bash
# Exit 0 if GPU supports INT8 mma.m16n8k32 (CC >= 8.0). Else exit 1 and print guidance.
set -euo pipefail
MIN_MAJOR="${MIN_CC_MAJOR:-8}"

cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
if [[ -z "${cap:-}" ]]; then
  echo "WARN: cannot read compute_cap from nvidia-smi"
  exit 0
fi

major=${cap%%.*}
minor=${cap##*.}
name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)

echo "GPU: ${name:-unknown} | compute capability ${major}.${minor} (sm_${major}${minor})"

if [[ "$major" -lt "$MIN_MAJOR" ]]; then
  echo ""
  echo "ERROR: This bundle uses inline Tensor Core MMA m16n8k32 (INT8 Ozaki kernels)."
  echo "       Minimum compute capability: ${MIN_MAJOR}.0 (e.g. A100, L4, RTX 3060+)."
  echo "       Your GPU is ${major}.${minor} (e.g. Colab T4) — not supported for full build."
  echo ""
  echo "Colab fix: Runtime -> Change runtime type -> Hardware accelerator -> GPU"
  echo "           Choose A100 or L4 (Colab Pro / paid runtime may be required)."
  echo ""
  echo "Partial option on T4: run side-kernel NCU only:"
  echo "  ./scripts/run_side_kernels_ncu.sh"
  exit 1
fi
exit 0