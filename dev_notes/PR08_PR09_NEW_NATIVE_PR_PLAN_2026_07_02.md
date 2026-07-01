# PR8 / PR9 Native Fused IQP — New PR Plan

**Date:** 2026-07-02  
**Context:** after PR #1387 reviewer remediation push  
**Current PR #1387 tip:** `6497bbbe7` (`fix(qdp): complete TC path reviewer remediation`)  
**Purpose:** plan a new, independent PR for Native/Fused IQP work.

---

## Decision

PR8 / PR9 should be treated as a **new PR line**, not as a continuation of
apache/mahout#1387.

Reviewer feedback on #1387 objected to merging scaffolding into `main`. The
response for #1387 was to convert that branch into a complete TC-path remediation.
The Native/Fused work is a separate feature path and should be presented as a new
explicit opt-in implementation.

---

## Scope Separation

### #1387 scope

- TC path correctness and reviewer remediation.
- Remove unused `AdaptiveOzakiEngine`.
- Fix TC launcher error handling.
- Add deterministic TC tests and CPU formula oracle.
- Preserve existing IQP semantics.

### New Native/Fused PR scope

- Add Native/Fused IQP encode path.
- Keep it explicit opt-in.
- Do not replace default IQP path.
- Do not change `encode_batch_tc` behavior.
- Provide correctness tests, smoke tests, and benchmark evidence.

---

## Candidate Source Branches

Inspect these branches before creating the clean PR branch:

- `pr8-native-hadamard-benchmark`
- `pr9-native-fused-iqp`
- local report worktree history, if needed:
  - `qdp/qdp-kernels/reports/pr9_ncu/worktree_PR8` was a local artifact and
    must not be copied into the PR branch.

Use these branches only as source material. Do not push them directly upstream.

---

## Clean Base Decision

Preferred base:

```bash
git fetch upstream
git checkout -b qdp-native-fused-iqp upstream/main
```

If the Native/Fused code needs #1387's TC remediation to compile or coexist,
base from the PR #1387 branch instead:

```bash
git checkout -b qdp-native-fused-iqp pr2-batch-throughput-opt
```

If based on #1387, the new PR body must explicitly say:

```text
This PR is functionally independent, but currently based on #1387 to avoid
conflicts with the TC remediation branch.
```

---

## Inventory Plan

Run a file inventory before applying changes:

```bash
git diff --name-status upstream/main..pr8-native-hadamard-benchmark
git diff --name-status upstream/main..pr9-native-fused-iqp
```

Classify every file into one of these buckets.

### Keep

- Production CUDA kernels, e.g. Native/Fused Hadamard/IQP implementation.
- Rust FFI/API changes required by the new explicit path.
- Python API exposure only if needed.
- Focused tests under `testing/qdp/`.
- Minimal benchmark script if it is small, maintained, and useful for reviewers.

### Drop

- `qdp/qdp-kernels/reports/`
- `terminals/`
- `dev_notes/`
- `HANDOVER_*.md`
- raw NCU logs
- generated `.arrow` / `.parquet`
- scratch files
- temporary reproduction worktrees
- agent-only comments or handover strings in production files

---

## Recommended PR Shape

Default recommendation: create **one clean PR** for the final usable Native/Fused
path, rather than separate scaffold PRs.

Use two PRs only if PR8 is independently useful and reviewable without PR9:

1. Native Hadamard path with explicit API and correctness tests.
2. Fused IQP native path built on top of that.

Given the reviewer feedback on #1387, avoid opening another scaffold-only PR.

---

## Correctness Plan

Use deterministic tests. Do not rely on unseeded random input.

Recommended tests:

- Native path vs existing trusted IQP/FWT path for feasible N.
- Native path normalization checks.
- Zero-phase invariant checks.
- Large-N smoke tests that verify:
  - launch succeeds
  - output is finite
  - no CUDA error is surfaced

Do not claim full-state N=27 numerical error unless that exact test is run.
N=27 materializes `2^27` amplitudes and is not appropriate for an automated
CPU-oracle correctness gate.

Suggested tolerances:

| Path | Reference | Tolerance |
|------|-----------|-----------|
| Native FP64 | existing FP64 FWT / IQP path | `1e-8` or tighter if measured |
| Native FP32 | FP64 reference | document drift, likely `1e-4` class |
| Smoke-only large N | invariants only | no full max-error claim |

---

## Validation Gate

Before opening the PR:

```bash
cargo build -p qdp-kernels
cargo test -p qdp-kernels -q
cargo clippy -p qdp-kernels -- -D warnings
pytest testing/qdp/ -q
ruff check qdp/
ty check qdp/
```

Run Native/Fused-specific tests explicitly, for example:

```bash
pytest testing/qdp/test_iqp_native_path.py -v
pytest testing/qdp/test_iqp_native_e2e.py -v -m "not slow"
```

Exact file names may change during cleanup.

---

## Benchmark Plan

Benchmark evidence belongs in the PR body or internal notes, not as raw generated
artifacts on the code branch.

Record:

- GPU model
- CUDA version
- WSL/Linux environment
- batch size
- qubit counts
- baseline path
- Native/Fused path
- runtime and speedup
- numerical verification result

Recommended table:

| N | Baseline | Native/Fused | Speedup | Verification |
|---|----------|--------------|---------|--------------|
| 12 | existing FWT/TC | TBD | TBD | TBD |
| 14 | existing FWT/TC | TBD | TBD | TBD |
| 16 | existing FWT/TC | TBD | TBD | TBD |
| 18 | existing FWT/TC | TBD | TBD | TBD |

---

## PR Body Draft

Title:

```text
[QDP] Add native fused IQP encoding path
```

Core wording:

```markdown
This PR adds a new explicit opt-in Native/Fused IQP encoding path.

It does not replace the default IQP path and does not change the TC path
semantics introduced/remediated in #1387.

The goal is to provide a separate Native/Fused implementation with focused
correctness tests and benchmark evidence.
```

Validation section should list exact commands and results.

---

## Immediate Next Steps

1. Inventory `pr8-native-hadamard-benchmark` and `pr9-native-fused-iqp`.
2. Produce a keep/drop table.
3. Pick base: `upstream/main` if possible, otherwise #1387 tip.
4. Create clean branch `qdp-native-fused-iqp`.
5. Apply only production code and focused tests.
6. Run full validation gate.
7. Push to `mahout_fork/qdp-native-fused-iqp`.
8. Open a new upstream PR, not a continuation of #1387.

