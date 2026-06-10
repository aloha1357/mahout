# Phase 8: DevOps & CI/CD Pipeline Handover

## Overview
This document records the DevOps and CI/CD troubleshooting steps taken to ensure that the massive PR for AdaptiveGEMM passes all rigorous checks in the `apache_mout` repository. Future developers and CI/CD engineers should reference this document to understand the strict formatting, typing, and linting constraints enforced by the project.

## Key CI/CD Hurdles & Solutions

### 1. Pre-Commit Hooks (Formatters & Linters)
The repository enforces a strict set of pre-commit hooks, including `end-of-file-fixer` and `trailing-whitespace`.
- **Issue**: Our `.md`, `.txt`, `.csv`, `.cu`, and `.h` files were failing these basic checks due to missing trailing newlines or trailing whitespace.
- **Solution**: Executed `pre-commit run --all-files` locally to let the hooks automatically format everything.
- **Takeaway**: Always run `pre-commit run --all-files` *before* pushing to remote. Never assume your IDE formatters cover all project-specific hook requirements.

### 2. Python Linting (Ruff)
- **Issue**: `E722 bare except` and `E701 multiple statements on one line`.
- **Solution**: Refactored `except:` to `except Exception:` and split one-liners into proper blocks (e.g., `if not metric_name:\n    continue`).
- **Issue**: `PGH003` required specific error codes for `# type: ignore`.
- **Solution**: Changed `# type: ignore` to targeted ignores like `# type: ignore[unresolved-attribute]`.

### 3. Strict Static Type Checking (`ty` - Astral)
The most significant blocker was the highly strict `ty` static analysis tool.
- **Issue (Unresolved Modules)**: `ty` does not resolve third-party packages or C++ extensions by default unless they are allowlisted or stubbed.
- **Solution**: Updated `pyproject.toml`'s `[tool.ty.analysis]` section to include `allowed-unresolved-imports`. Added `numpy`, `pytest`, `qiskit`, `boto3`, `cirq`, `braket`, and `adaptive_gemm_py`.
- **Issue (Dynamic Attributes in Braket/Qiskit)**: `ty` strictly evaluates Abstract Syntax Trees (AST) and fails on dynamic dispatch (e.g., `circuit.x()`, `backend.shots = ...`) commonly used in quantum libraries via `__getattr__`.
- **Solution (Structural Typing & Reflection)**:
  - Suppressing with `# type: ignore` is brittle with `ty`.
  - Used explicit structural downcasting to `typing.Any` (e.g., `circuit: Any = Circuit()`).
  - Used safe reflection like `getattr(result, "measurement_counts")` and `setattr(backend, "shots", shots)`.
- **Takeaway**: Do not play "whack-a-mole" with `# type: ignore`. Use `getattr/setattr` for reflection or explicitly declare objects as `Any` when dealing with heavily metaprogrammed external dependencies (like AWS Braket's Circuit builders).

### 4. Git Hygiene & Build Artifacts
- **Issue**: `git add .` accidentally staged untracked `build/` directories and `.exe` binaries, which would severely bloat the repository and break CI.
- **Solution**: Used `git rm -r --cached` and `git commit --amend` to remove these artifacts. Ensure `.gitignore` properly tracks standard build directories (e.g., `build_wsl/`).

## Actionable Advice for Future CI/CD Operations
1. **Never bypass hooks**: If a hook fails, don't use `--no-verify`. Fix the root cause.
2. **Handle Dynamic Typing Elegantly**: When introducing new quantum backends, expect `ty` to fail on dynamic methods. Wrap them in `Any` or use `getattr` proactively.
3. **Keep Commits Atomic**: As we transition to merging this massive project, we must break the work down into atomic PRs (Infrastructure -> Implicit FWT -> Memory Optimization -> Runtime Dispatch). See `PR_arch.md`.
