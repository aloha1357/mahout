#!/usr/bin/env bash
# One-shot: build + benchmark + all NCU + summary
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
chmod +x scripts/*.sh
bash scripts/build.sh
bash scripts/run_benchmark.sh
bash scripts/run_all_ncu.sh
echo ""
echo "=== DONE ==="
echo "Read: reports/ncu_summary.txt"
echo "      reports/ncu_summary.md"
echo "      reports/benchmark_results.txt"