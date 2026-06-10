# Standard PR Workflow Handover

This document strictly defines the lifecycle for implementing, validating, and submitting every Pull Request (PR2 through PR6) to ensure branch cleanliness, prevent merge conflicts, and accurately track performance.

## Phase 1: PR Branch Preparation & Conflict Resolution
1. **Sync Upstream:** Before starting or resuming a PR, always fetch the latest changes from the official `upstream/main` repository to capture recently merged PRs.
   ```bash
   git fetch upstream main
   ```
2. **Rebase & Resolve:** Rebase the current PR branch onto `upstream/main` to resolve any conflicts caused by previous PRs.
   ```bash
   git rebase upstream/main
   # If conflicts occur, resolve them, git add the files, and run:
   # git rebase --continue
   ```

## Phase 2: Implementation & Unit Testing
1. **Code Only:** Implement the required C++/CUDA/Rust changes directly on the PR branch. Remove obsolete/naive commented-out code.
2. **Unit Tests:** Add or update strict unit tests under `testing/qdp/` targeting the new logic. 
3. **Validation:** Ensure the outputs perfectly match the PyTorch exact theoretical baseline (`torch.testing.assert_close(..., rtol=1e-12, atol=1e-12)`). The PR branch **must** pass all tests via `pytest`.

## Phase 3: Pre-Commit Quality Checks
1. **Formatting & Linting:** Run the strict pre-commit hooks over all modified files in the PR branch. **This step is mandatory before pushing.**
   ```bash
   uv run pre-commit run --all-files
   ```
2. **Fix Issues:** Address any styling, trailing whitespaces, or type checking errors flagged by `ruff`, `ty`, or EOF fixers.

## Phase 4: Push PR Branch
1. **Push Clean Code:** Force push (if rebased) the PR branch to your fork. The PR branch must **only** contain code (`.cu`, `.py`, `.rs`) and unit test files.
   ```bash
   git push --force origin <pr-branch-name>
   ```
   *Note: Never include benchmark scripts or markdown performance reports in the PR branch.*

## Phase 5: Benchmarking & Documentation (Main Branch Only)
1. **Switch to Main:** Checkout the `main` branch.
2. **Run Benchmarks:** Write a temporary benchmark script (e.g., `benchmark_prX.py`) to measure the performance (latency, throughput, memory) against the original Pre-PR1 baseline.
3. **Document Speedups:** Record the optimization results and speedup multipliers directly into the corresponding planning Markdown file in the `main` branch (e.g., `AdaptiveGEMM_repo/dev_notes/PR02_Batch_Throughput_Optimization.md`).
4. **Commit to Main:** Add the markdown files and commit them to the `main` branch to preserve the project's historical performance records.
