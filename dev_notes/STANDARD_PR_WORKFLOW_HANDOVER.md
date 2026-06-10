# Standard PR Workflow Handover

This document is the **mandatory execution order** for PR1 through PR6.
Previous handovers listed the right ideas in the wrong sequence, which caused:

- PR2 branches cut from `main` with docs/benchmarks mixed into code commits
- Skipped environment setup (missing PyTorch / CUDA libs → pre-commit or tests ignored)
- Pushing before pre-commit passed
- Documentation committed on PR code branches instead of `internal-dev-notes`

**Follow the phases below in order. Do not skip or reorder steps.**

---

## Branch roles (two repos)

| Branch | Repo | Contents |
|--------|------|----------|
| `pr1-phase-kernel-opt` … `pr6-*` | `mahout` fork (`mahout_fork`) | **Code + unit tests only** |
| `internal-dev-notes` | `apache_mout` fork (`origin`) | PR drafts, benchmark records, this handover |
| `main` | `apache_mout` fork (`origin`) | Archived reports synced from `internal-dev-notes` |

PR stack (each branch inherits the **tip** of the previous PR):

```
main/upstream ──► PR1 ──► PR2 ──► PR3 ──► PR4 ──► PR5 ──► PR6
```

| PR | Branch | Base (must branch from) |
|----|--------|-------------------------|
| PR1 | `pr1-phase-kernel-opt` | `upstream/main` (or agreed point) |
| PR2 | `pr2-implicit-fwt-rework` | **PR1 tip** (`git merge-base` = PR1 HEAD) |
| PR3 | `pr3-shared-memory-fwt` | **PR2 tip** |
| PR4 | `pr4-kronecker-fwt` | **PR3 tip** |
| PR5 | `pr5-implicit-hadamard-engine` | **PR4 tip** |
| PR6 | `pr6-tensor-core-acceleration` | **PR5 tip** |

**Verify inheritance before any work:**

```bash
git fetch mahout_fork pr1-phase-kernel-opt
git log --oneline pr1-phase-kernel-opt..HEAD          # should list only THIS PR's commits
git diff pr1-phase-kernel-opt..HEAD --stat            # should list only THIS PR's files
```

If the diff includes `dev_notes/`, `reports/`, `HANDOVER_*.md`, or unrelated kernel files, **reset and re-branch from the correct PR tip**.

---

## Phase 0: Complete the previous PR first

Before starting PRn (n ≥ 2), PR(n−1) must already be:

1. Merged or pushed to `mahout_fork` with a known tip commit
2. Free of agent handover comments (`// PRn Optimization`, etc.) in `.cu` / `.rs`
3. Passing pre-commit and unit tests on the PR branch itself

**Do not start PR2 until PR1 is pushed and stable.**

---

## Phase 1: Create / reset the PR branch (code only)

```bash
git fetch mahout_fork pr1-phase-kernel-opt    # example: starting PR2
git checkout -B pr2-implicit-fwt-rework mahout_fork/pr1-phase-kernel-opt
# implement changes …
git add qdp/ testing/                         # code paths only
git commit -m "feat(qdp): …"
```

**Rules:**

- Branch from the **previous PR tip**, not from `main` and not from a stale experimental branch.
- One PR = kernel / Rust / Python binding changes + `testing/` tests.
- **Never** commit `.md` reports, `dev_notes/`, or benchmark scripts on the PR branch.
- Remove dead code and verbose debug prints; keep ASF-appropriate algorithm comments only.

---

## Phase 2: Environment setup (mandatory — do not skip)

Missing packages are not optional. If `uv sync`, `import torch`, or `maturin develop` fails, **fix the environment before continuing**.

### WSL (recommended for CUDA + pre-commit)

```bash
cd /mnt/d/.../apache_mout
export UV_PROJECT_ENVIRONMENT=.venv_wsl
uv sync --group dev --python 3.12

# .venv symlink for clippy hook (pre-commit expects .venv/bin/activate)
ln -sfn .venv_wsl .venv

# Source env helper (sets LD_LIBRARY_PATH for nvidia wheels + torch)
source setup_wsl_env.sh
```

**Quick health check:**

```bash
bash check_env.sh
# Expect: torch 2.9.0+cu*, cuda: True, pre-commit + pytest available
```

### Common fixes (learned from PR2 rework)

| Symptom | Fix |
|---------|-----|
| `libcudnn.so.9` / `libnvshmem_host.so.3` not found | `uv pip install --reinstall nvidia-cudnn-cu12 nvidia-nvshmem-cu12`; `source setup_wsl_env.sh` |
| `tch` expects PyTorch 2.9.0 | `uv pip install torch==2.9.0` (must match `pyproject.toml` `<=2.9.0`) |
| `cargo clippy` cannot find `.venv/bin/activate` | `ln -sfn .venv_wsl .venv` |
| `uv sync` IO error on `.venv/Scripts` (Windows mount) | Use `UV_PROJECT_ENVIRONMENT=.venv_wsl`; symlink `.venv` → `.venv_wsl` |
| Kernel changes not visible in tests | `uv run --active maturin develop --manifest-path qdp/qdp-python/Cargo.toml --release` |

**Do not push while the environment is broken.**

---

## Phase 3: Build kernels & run unit tests

```bash
source setup_wsl_env.sh
export PATH=/usr/local/cuda/bin:$PATH

uv run --active maturin develop \
  --manifest-path qdp/qdp-python/Cargo.toml --release

python -m pytest testing/qdp/test_<feature>.py -v
```

**Correctness bar:** `torch.testing.assert_close(..., rtol=1e-12, atol=1e-12)` against PyTorch reference.

PR2 example (completed 2026-06-10): `testing/qdp/test_implicit_fwt.py` — 12/12 passed after clean re-branch from PR1.

---

## Phase 4: Pre-commit (mandatory gate before push)

Run on **changed files only** (avoids CRLF churn on WSL+Windows mount):

```bash
source setup_wsl_env.sh
pre-commit run --files \
  qdp/qdp-kernels/src/iqp.cu \
  testing/qdp/test_implicit_fwt.py
```

All hooks must pass: ruff, ty, license headers, `cargo fmt`, `cargo clippy`.

**If a hook fails:** install the missing tool/package and re-run. Do not push with `--no-verify`.

---

## Phase 5: Push PR branch to fork

```bash
git push mahout_fork <pr-branch-name> --force-with-lease
```

Confirm the remote branch contains **only** code commits inherited from the previous PR.

---

## Phase 6: Documentation & benchmarks (`internal-dev-notes` only)

After the PR branch is pushed and validated:

```bash
git checkout internal-dev-notes
# edit dev_notes/PR0XX_*.md, reports/, benchmark results
git add dev_notes/ reports/
git commit -m "docs: PR2 benchmark results and workflow notes"
git push origin internal-dev-notes
```

Optionally sync to `main` archive:

```bash
git checkout main
git checkout internal-dev-notes -- dev_notes/STANDARD_PR_WORKFLOW_HANDOVER.md reports/PR002_Benchmark.md
git commit -m "docs: sync PR2 records from internal-dev-notes"
git push origin main
```

Benchmark scripts (`benchmark_linux.py`, etc.) live on `main` or `internal-dev-notes`, **not** on PR code branches.

---

## Phase 7: Upstream sync (only when explicitly rebasing the stack)

Rebasing PR branches onto `upstream/main` is a **separate maintenance step**, not the first step when starting PR2+.

```bash
git fetch upstream main
# Only when the team agrees to rebase the whole stack:
git checkout pr1-phase-kernel-opt && git rebase upstream/main
# Then re-stack PR2 from new PR1 tip, PR3 from new PR2 tip, etc.
```

For day-to-day PR2–PR6 work, **inherit from the previous PR tip**, not from a fresh `main` checkout with docs mixed in.

---

## Anti-patterns (do not repeat)

| Wrong | Right |
|-------|-------|
| `git checkout -b pr2 main` with docs commits | `git checkout -B pr2 pr1-tip` code only |
| Push when pre-commit fails | Fix env → re-run pre-commit → then push |
| Ignore `uv sync` / torch import errors | Run `check_env.sh`; reinstall nvidia wheels |
| `// PR2 Optimization` in `.cu` | PR description only; algorithm comments in code |
| Benchmark `.md` on PR branch | `internal-dev-notes` → sync to `main` |
| Delete prior PR unit tests | Accumulate tests PR1 → PR6 |

---

## Quick checklist (copy per PR)

- [ ] Previous PR pushed and tip commit recorded
- [ ] New branch from previous PR tip (diff shows only this PR's files)
- [ ] `source setup_wsl_env.sh` + `check_env.sh` passes
- [ ] `maturin develop --release` succeeds
- [ ] `pytest testing/qdp/…` passes
- [ ] `pre-commit run --files <changed>` passes
- [ ] `git push mahout_fork <branch> --force-with-lease`
- [ ] Benchmark + docs committed on `internal-dev-notes`
- [ ] (Optional) sync docs to `origin/main`