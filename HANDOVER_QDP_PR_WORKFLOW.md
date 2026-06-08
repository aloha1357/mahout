# QDP PR Workflow: CONTRIBUTING, Pre-commit, Benchmark & Inheritance

This document is the **mandatory handover checklist** for QDP experimental PRs (PR1–PR6).
Every PR must follow the same process before pushing to a fork or opening a review on
`apache/mahout`.

Related docs:

- [CONTRIBUTING.md](CONTRIBUTING.md) — repo-wide workflow and testing
- [qdp/DEVELOPMENT.md](qdp/DEVELOPMENT.md) — build, pytest, benchmarks
- [HANDOVER_PR1_PR6.md](HANDOVER_PR1_PR6.md) — technical background per PR
- [.github/PULL_REQUEST_TEMPLATE](.github/PULL_REQUEST_TEMPLATE) — PR description format

---

## 1. PR stack & inheritance (PR1 → PR2 → PR3 → …)

Experimental branches are **stacked**, not independent rebases on `main`.

| PR | Typical branch | Base (compare against) | Must inherit from previous PR |
|----|----------------|----------------------|-------------------------------|
| PR1 | `pr1-phase-kernel-opt` | `main` (or agreed upstream point) | — |
| PR2 | `pr2-implicit-fwt` / `pr2-batch-throughput-opt` | **PR1 end commit** | PR1 kernel changes, tests, clean comment style |
| PR3 | `pr3-shared-memory-fwt` | **PR2 end commit** | PR1 + PR2 |
| PR4 | `pr4-kronecker-fwt` | **PR3 end commit** | PR1 + PR2 + PR3 |
| PR5 | `pr5-implicit-hadamard-engine` | **PR4 end commit** | … |
| PR6 | `pr6-tensor-core-acceleration` | **PR5 end commit** | … |

**Inheritance rules (apply to every handoff):**

1. **Do not reintroduce removed patterns** — e.g. PR1 eliminated `// PR1 Optimization`
   comments in `.cu` files; PR2+ must keep kernel comments minimal and ASF-appropriate
   (math/algorithm only, no PR numbers or agent handover text).
2. **Keep accumulated unit tests** — add new tests; do not delete prior PR tests unless
   behaviour intentionally changes.
3. **Keep benchmark scripts generic** — use names like `benchmark_phase.py`, not
   `benchmark_pr1.py`; no references to internal handover files in source.
4. **Branch from the previous PR’s tip** when starting the next experiment, not from
   stale `main`, unless deliberately rebasing the whole stack.

---

## 2. Code & comment standards

| Allowed | Not allowed in committed code |
|---------|-------------------------------|
| ASF license headers | `// PRn Optimization: …` |
| Short algorithm comments (φ, norm, bit masks) | Agent/session handover notes in `.cu` / `.rs` |
| Neutral test docstrings | `GEMINI.md`, `HANDOVER_*.md` references in source |
| PR Why/How in **PR description** only | PyTorch or Naive CPU as benchmark baseline |

---

## 3. Pre-commit gate (required before every fork push)

Pre-commit must pass on **changed files** before pushing. This matches
[CONTRIBUTING.md](CONTRIBUTING.md) and CI (`.github/workflows/pre-commit.yml`).

### 3.1 Run locally (recommended: Linux / WSL with CUDA)

```bash
cd mahout
source .venv/bin/activate   # or .venv_wsl on WSL — see below
uv sync --group dev

# Only touch files you changed (avoid --all-files on WSL+Windows mount — CRLF churn)
uv run pre-commit run --files \
  qdp/qdp-kernels/src/phase.cu \
  testing/qdp/test_bindings.py \
  qdp/qdp-python/benchmark/benchmark_phase.py
```

**WSL note:** the `clippy` hook expects `.venv/bin/activate`. If using `.venv_wsl`:

```bash
mkdir -p .venv && ln -sfn ../.venv_wsl/bin .venv/bin
```

**CUDA rebuild** (when kernels change):

```bash
export PATH=/usr/local/cuda/bin:$PATH
uv run maturin develop --manifest-path qdp/qdp-python/Cargo.toml
```

### 3.2 Hooks that must pass

| Hook | Scope |
|------|--------|
| trailing whitespace / EOF | all changed files |
| ruff lint + format | Python |
| ty check | Python |
| license headers | Python, Rust, CUDA |
| `cargo fmt` | Rust (if Rust changed) |
| `cargo clippy` | `qdp/Cargo.toml` (CI uses `-D warnings`) |

### 3.3 Push policy

1. Push to **your fork** first (`git push <your-fork> <branch>`).
2. Open or update the upstream PR only when pre-commit, tests, and benchmark report are done.
3. Do **not** request review until the checklist in §6 is complete.

---

## 4. Unit tests (required for every PR)

Per [CONTRIBUTING.md](CONTRIBUTING.md), behaviour changes need tests under `testing/`.

### 4.1 Where to add tests

- Primary: `testing/qdp/test_bindings.py`
- GPU tests: `@requires_qdp`, `@pytest.mark.gpu`
- Match existing style: deterministic expected values, normalization checks, error paths

### 4.2 Minimum coverage per encoding/kernel PR

| Check | Example |
|-------|---------|
| Basic correctness | known phases → expected amplitudes |
| Single- & multi-qubit | N=1, N=2, N=4 |
| Normalization | unit norm for arbitrary inputs |
| Batch path | 2D tensor, batch_size > 1 |
| Large-N smoke | same N as benchmark (e.g. N=14) |
| Error handling | wrong length, non-finite values |

### 4.3 Run tests (WSL + GPU)

```bash
source .venv/bin/activate
export PATH=/usr/local/cuda/bin:$PATH
uv run pytest testing/qdp/test_bindings.py -k "phase_encode" -v
# or the relevant -k filter for the PR
```

All selected tests must **pass** before push.

---

## 5. Benchmark methodology (strict GPU vs base)

**Never** use Naive CPU or PyTorch as the primary speedup baseline. Compare **optimized GPU
vs unoptimized GPU on the same hardware** via git checkout (git archaeology).

### 5.1 Procedure (every PR)

```text
1. Identify BASE commit  — parent of the PR’s core change (or previous PR tip).
2. git checkout <BASE> && rebuild extension (maturin develop).
3. Run benchmark with --label baseline → record numbers.
4. git checkout <PR-TIP> && rebuild extension.
5. Run benchmark with --label optimized → record numbers.
6. Update reports/PR00X_Benchmark.md (ms, GPU model, N, batch, iterations).
```

### 5.2 Script & config

- Script: `qdp/qdp-python/benchmark/benchmark_phase.py` (or PR-specific script without
  `PRn` in the filename).
- Default config (PR1 reference): **N=14**, **batch=128**, **iterations=5**.
- Document GPU model (e.g. RTX 4060), date, and commit hashes in the report.

### 5.3 Report table format (`reports/PR00X_Benchmark.md`)

Use **milliseconds (ms)**. Example:

| Implementation | Per sample | Total (batch) | Notes |
|--------------|------------|---------------|-------|
| GPU (baseline checkout) | 1.26 ms | 161.91 ms | Strict checkout before PRn |
| GPU (this PR) | 1.16 ms | 147.94 ms | ~9.4% gain, same GPU |

Optional: one-line CPU reference for scale only — **not** as speedup denominator.

### 5.4 PR description

Copy the benchmark table into the PR body (`.github/PULL_REQUEST_TEMPLATE` Why/How sections).
Do not rely on stale numbers from other GPUs (e.g. RTX 4090) unless re-measured.

---

## 6. Per-PR completion checklist

Copy into each PR description before requesting review:

```markdown
## Checklist

- [ ] Branch inherits previous PR (if PR2+: based on PR{n-1} tip, not stale main)
- [ ] No PR-number / agent comments in `.cu`, `.rs`, or Python source
- [ ] Unit tests added/updated in `testing/qdp/`
- [ ] `pytest` passed locally for new/changed tests (GPU)
- [ ] Strict GPU baseline vs GPU optimized benchmark recorded in `reports/PR00X_Benchmark.md`
- [ ] `pre-commit run --files <changed>` passed (incl. clippy)
- [ ] Pushed to fork; upstream PR opened only when above is green
- [ ] Documentation updated if user-visible (e.g. `qdp/qdp-python/benchmark/README.md`)
```

---

## 7. PR1 → PR2 handoff (current)

**PR1 (`pr1-phase-kernel-opt`) delivers:**

- `phase.cu` / `iqp.cu`: branchless bit tests, host-side `norm_factor` (no PR comments)
- `testing/qdp/test_bindings.py`: `test_phase_encode_*` suite (incl. N=14, batch=128)
- `qdp/qdp-python/benchmark/benchmark_phase.py`
- `reports/PR001_Benchmark.md`: GPU vs GPU numbers (verify on target hardware)

**PR2 must:**

1. Branch from PR1 tip after PR1 checklist is complete.
2. Keep PR1 kernel + test + comment conventions.
3. Add PR2 tests and `reports/PR002_Benchmark.md` using the same §4–§5 process.
4. Pass pre-commit before any fork push.

---

## 8. Environment quick reference

| Environment | venv | CUDA |
|-------------|------|------|
| Linux / WSL | `.venv` or `.venv_wsl` + symlink §3.1 | `export PATH=/usr/local/cuda/bin:$PATH` |
| Windows native | `.venv` (Scripts) | pre-commit clippy may fail — prefer WSL for QDP |

Do **not** run `pre-commit run --all-files` on a WSL mount of a Windows working copy;
it can rewrite line endings across the entire tree.