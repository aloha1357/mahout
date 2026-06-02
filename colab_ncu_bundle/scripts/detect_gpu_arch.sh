#!/usr/bin/env bash
# Prints -arch=sm_XX for nvcc based on first GPU (Colab T4=75, L4=89, A100=80, etc.)
set -euo pipefail
cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
if [[ -z "${cap:-}" ]]; then
  echo "sm_75"
  exit 0
fi
major=${cap%%.*}
minor=${cap##*.}
echo "sm_${major}${minor}"