# Tensor Core IQP Performance Spec

**Date:** 2026-07-02
**Branch context:** `internal-dev-notes` after PR #1387 remediation push
**Purpose:** define where the current Tensor Core IQP path is actually faster,
what remains unverified on the current stack, and what work must be completed
before making new performance claims.

Repository workflow references:

- `CONTRIBUTING.md`
- `qdp/DEVELOPMENT.md`
- `dev_notes/STANDARD_PR_WORKFLOW_HANDOVER.md`

---

## 1. Current evidence

The existing notes support two different performance statements, and they must
not be mixed.

### 1.1 Encode-only, direct kernel timing

Canonical source: `benchmark_pr6.py`, release build, batch `1024`.

Recorded handover numbers:

| N | FWT `encode()` | TC `encode_batch_tc()` | Interpretation |
|---|----------------|------------------------|----------------|
| 12 | ~14 ms | ~8 ms | TC faster, about `1.68x` |
| 14 | ~69 ms | ~207 ms | TC slower, about `0.33x` |

Interpretation:

- For `N <= 12`, the fused shared-memory TC path is a real encode-side win.
- For `N > 12`, the current TC path is **not** an encode-side win.
- The documented reason is global-memory and orchestration overhead in the
  Kronecker / Implicit Hadamard route, including transpose and allocation
  costs.

### 1.2 End-to-end (E2E) pipeline timing

Canonical source in prior notes: PR7 E2E benchmark, `iqp-z`, `32` samples,
`Mahout-TC` vs `Mahout-Arrow FWT`.

Recorded handover numbers:

| N | Mahout-TC | Mahout-Arrow FWT | Interpretation |
|---|-----------|------------------|----------------|
| 12 | ~0.05 s | ~0.83 s | TC path much faster |
| 14 | ~0.05 s | ~1.1 s | TC path much faster |
| 16 | ~0.057 s | ~0.827 s | TC path much faster |

Interpretation:

- The experimental PR7 stack showed a strong **pipeline-level** win for the TC
  route.
- This is a valid historical result, but it is **not the same benchmark** as
  encode-only timing.
- The current upstream-style `benchmark_e2e.py` in this worktree does not
  expose `mahout-tc`, so this E2E result is not currently reproducible from the
  stock script alone.

---

## 2. What TC is actually faster at

The current TC path is faster in these two cases only:

1. **Direct encode, `N <= 12`:**
   the fused shared-memory Tensor Core kernel beats the default FWT encode path.
2. **Experimental PR7 E2E IQP pipeline:**
   the full `Mahout-TC` path beat the `Mahout-Arrow FWT` path for the tested
   `iqp-z` workloads.

The current TC path is **not established as faster** for:

- direct encode at `N > 12`
- generic default `encode()` replacement
- current upstream `benchmark_e2e.py` on this branch, because the comparison
  path is not wired in the current script

---

## 3. Working hypothesis for the E2E win

Until the PR7 E2E benchmark is restored and rerun on the current stack, the
following should be treated as working hypotheses rather than final claims:

- The E2E win is not explained by `N > 12` encode-only kernel speed, because the
  historical notes explicitly show that encode-only TC can be slower there.
- The E2E win likely comes from the overall `Mahout-TC` pipeline shape used in
  PR7 rather than from a single large-N encode kernel advantage.
- Any future PR body must separate:
  - encode-only kernel speed
  - E2E pipeline speed
  - correctness validation

---

## 4. Required follow-up work

### 4.1 Restore canonical benchmarks

We need two benchmark paths, both reproducible from the current repo state or a
cleanly documented helper branch:

1. **Encode-only canonical benchmark**
   - Source: `benchmark_pr6.py` or an equivalent maintained replacement.
   - Workload: IQP full-ZZ, release build, batch `1024`.
   - Comparison: `encode(..., "iqp")` vs `encode_batch_tc(..., "iqp")`.

2. **PR7-style E2E benchmark**
   - Restore `mahout-tc` comparator support in an E2E harness.
   - Workload: `iqp-z`, `32` samples minimum, same environment notes as PR7.
   - Comparison: `Mahout-TC` vs `Mahout-Arrow FWT`.

### 4.2 Attribute the E2E win

After the E2E harness is restored, add one attribution pass:

1. Measure `N = 12, 14, 16`.
2. Record total E2E time and direct encode time separately.
3. Confirm whether the E2E advantage persists when encode-only `N > 12` remains
   slower.
4. Document which part of the pipeline dominates:
   - encode
   - data movement / format conversion
   - loader / batch orchestration
   - downstream handoff

### 4.3 Set benchmark reporting rules

Future benchmark tables must label themselves as one of:

- `encode-only`
- `latency`
- `throughput`
- `E2E`

No chart or PR body should compare numbers across those categories.

---

## 5. Acceptance criteria

This section is the release gate for future TC performance claims and for any PR
that intends to present new TC benchmark evidence.

### 5.0 PR preconditions

Before opening or updating a PR with TC performance claims, all repository and
QDP workflow prerequisites must already be satisfied.

Required setup and workflow:

- Use the repo environment described in `CONTRIBUTING.md` and
  `qdp/DEVELOPMENT.md`.
- Install and run `pre-commit`.
- Keep code changes on the PR branch and keep docs / benchmark notes on
  `internal-dev-notes` unless they are explicitly meant for upstream docs.
- Do not push a PR branch while pre-commit or required tests are failing.

Required pre-PR checks:

- `pre-commit run --files <changed files>` passes for the actual changed files.
- `pre-commit run --all-files` is recommended before opening the PR when the
  branch touches shared benchmark or test infrastructure.
- Changed-feature tests under `testing/` pass.
- Relevant Rust tests pass.
- Relevant Python/QDP tests pass.

### 5.1 Evidence gate

Accept a new TC performance claim only if all of the following are true:

- The benchmark script is present in the repo or referenced by exact branch and
  path.
- The environment is recorded:
  - GPU model
  - CUDA version
  - WSL/Linux context
  - release vs debug build
- The compared APIs are stated explicitly.
- Numerical verification is reported next to timing results.

### 5.2 Encode-only claim gate

Accept the claim "`encode_batch_tc` is faster" only if:

- the benchmark is encode-only
- the batch size is stated
- `encode()` and `encode_batch_tc()` are compared on the same input shape
- results are reported separately for `N <= 12` and `N > 12`

Current expectation from prior evidence:

- `N <= 12`: TC may win
- `N > 12`: TC should not be claimed faster unless rerun data proves it

### 5.3 E2E claim gate

Accept the claim "`Mahout-TC` is faster end-to-end" only if:

- the benchmark is an E2E harness
- `mahout-tc` and `mahout-arrow` are both present in the same run
- workload parameters are recorded
- correctness verification passes
- the result is reproduced on the current code path, not only quoted from
  historical notes

### 5.4 PR-body gate

Before adding a performance chart to any PR:

- confirm the chart category
- confirm the comparator APIs
- confirm the benchmark script is reproducible from the branch or documented
  helper branch
- confirm the numbers match the textual claim in the PR body

### 5.5 Minimum validation gate

Any PR that changes TC code, TC tests, or TC benchmark harnesses must satisfy a
minimum validation gate before push.

Repository-wide gate from `CONTRIBUTING.md`:

- `make pre-commit`
- `make tests`, or the equivalent targeted subsets below when full-suite runtime
  is impractical during iteration

Targeted QDP gate for TC-path work:

- `cargo build -p qdp-kernels`
- `cargo test -p qdp-kernels -q`
- `cargo clippy -p qdp-kernels -- -D warnings`
- `pytest testing/qdp/test_iqp_tc_path.py -v`
- `pytest testing/qdp/test_bindings.py -q`
- `pytest testing/qdp/ -q`
- `ruff check qdp/`
- `ty check qdp/`

If the PR changes the benchmark harness:

- the relevant benchmark script must execute successfully in the documented
  environment
- benchmark output must include a correctness check or explicitly reference a
  paired correctness test run

### 5.6 Benchmark acceptance gate

Accept a new TC benchmark report only if all of the following are true:

- Pre-commit checks passed before the report is presented as PR evidence.
- The benchmark was run on the same code that is being proposed in the PR.
- The commands needed to reproduce the benchmark are written down exactly.
- The measured path and baseline path are named precisely.
- The associated TC correctness tests passed on the same branch.
- The report states whether it is:
  - historical evidence
  - freshly rerun evidence on the current branch

### 5.7 PR readiness gate

Treat a TC PR as ready for review only if:

- repository workflow requirements from `CONTRIBUTING.md` are satisfied
- branch hygiene requirements from `dev_notes/STANDARD_PR_WORKFLOW_HANDOVER.md`
  are satisfied
- required targeted QDP tests are green
- benchmark evidence, if included, is benchmark-category-correct and reproducible
- the PR body follows the repository template and does not overclaim beyond the
  measured evidence

---

## 6. Immediate implementation tasks

1. Re-run the repository workflow for the active TC branch before any new
   benchmark-driven PR update:
   - `pre-commit run --files <changed files>`
   - relevant `cargo` and `pytest` commands
2. Recover or reintroduce a maintained `benchmark_pr6.py` equivalent for the
   current stack.
3. Restore PR7-style `mahout-tc` support in an E2E benchmark harness.
4. Rerun both benchmark families on the current remediation baseline.
5. Write one benchmark report that separates:
   - small-N encode-only
   - large-N encode-only
   - E2E
6. Use that report as the performance baseline for the future Native/Fused PR.

---

## 7. Non-goals

This spec does not claim:

- that TC is universally faster than the default IQP path
- that large-N encode-only TC is already optimized
- that PR #1387 is a speedup PR
- that Native/Fused FP32 work is part of the TC remediation scope
