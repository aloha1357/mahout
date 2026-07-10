#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${REPO}/setup_wsl_env.sh"
cd "${REPO}"

python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available())"
python -m pre_commit --version
python -m pytest --version
python -c "import ruff, ty; print('ruff/ty: ok')"
