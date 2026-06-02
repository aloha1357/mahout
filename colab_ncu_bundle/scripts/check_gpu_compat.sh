#!/usr/bin/env bash
# Prints shell snippet to eval: export T4_SIMT_MODE=0|1
set -euo pipefail
MIN_MAJOR="${MIN_CC_MAJOR:-8}"
T4_SIMT_MODE=0

cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
if [[ -n "${FORCE_T4_SIMT_MODE:-}" ]]; then
  echo "FORCE_T4_SIMT_MODE — SIMT-compat build" >&2
  echo "export T4_SIMT_MODE=1"
  exit 0
fi

if [[ -z "${cap:-}" ]]; then
  echo "WARN: cannot read compute_cap; full build" >&2
  echo "export T4_SIMT_MODE=0"
  exit 0
fi

major=${cap%%.*}
minor=${cap##*.}
name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo "GPU: ${name:-unknown} | CC ${major}.${minor} (sm_${major}${minor})" >&2

if [[ "$major" -lt "$MIN_MAJOR" ]]; then
  T4_SIMT_MODE=1
  echo "" >&2
  echo "SIMT-compat mode (same as original repo: N<=12 uses SIMT, no Ozaki MMA)." >&2
  echo "PR007 N=14/16 Ozaki path skipped on this GPU; use A100/L4 for full matrix." >&2
  echo "Set PR007_STRICT_GPU=1 to abort instead of compat mode." >&2
  echo "" >&2
  if [[ "${PR007_STRICT_GPU:-0}" == "1" ]]; then
    echo "ERROR: PR007_STRICT_GPU=1" >&2
    exit 1
  fi
fi

echo "export T4_SIMT_MODE=${T4_SIMT_MODE}"
