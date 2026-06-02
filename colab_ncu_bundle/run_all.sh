#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
chmod +x scripts/*.sh
bash scripts/build.sh
bash scripts/run_benchmark.sh
bash scripts/run_all_ncu.sh
echo ""
echo "=== DONE ==="
cat reports/ncu_summary.txt