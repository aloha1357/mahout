# Benchmark Handover — `apache_mout` Experimental Repo

This repo is **`origin` → `https://github.com/aloha1357/apache_mout.git`**.
All QDP CUDA experiments and benchmark reports live here—not on `upstream` (apache/mahout)
until promoted.

Related docs:

- [HANDOVER_PR1_PR6.md](HANDOVER_PR1_PR6.md) — performance story and N≤12 / N>12 behaviour
- [HANDOVER_QDP_PR_WORKFLOW.md](HANDOVER_QDP_PR_WORKFLOW.md) — pre-commit, GPU vs GPU methodology
- [qdp/qdp-python/benchmark/README.md](qdp/qdp-python/benchmark/README.md) — upstream benchmark README

---

## 1. Remotes (do not confuse)

| Remote | URL | Use for |
|--------|-----|---------|
| **`origin`** | `aloha1357/apache_mout` | **Experiments, benchmarks, reports** |
| `mahout_fork` | `aloha1357/mahout` | Upstream PR branches only |
| `upstream` | `apache/mahout` | Clean baseline / eventual merge target |

```bash
git remote -v
# origin → apache_mout  ← work here
```

---

## 2. Branches on `apache_mout`

| Branch | Code stack | Benchmark-critical files |
|--------|------------|---------------------------|
| `main` (local) | Docs + `reports/PR00X_*.md` | Handover, all benchmark reports |
| `pr6-tensor-core-acceleration` | Full integrated CUDA stack | `benchmark_pr6.py`, `encode_batch_tc` |
| `pr5-implicit-hadamard-engine` | Ozaki engine | — |
| `pr4-kronecker-fwt` / `origin/pr4-runtime` | Kronecker FWT | — |
| `pr3-shared-memory-fwt` | Fused N≤12 | — |
| `pr1-phase-kernel-opt` | Phase/IQP kernel opt | — |
| `origin/main` (remote) | Research layout only | **Missing** handover reports—use local `main` |

**Rule:** kernel code on experiment branches; **reports + handover on `main`**.

---

## 3. Files to keep (checklist)

### 3.1 Always on `main` (documentation)

| File | Purpose |
|------|---------|
| `HANDOVER_PR1_PR6.md` | Final scores narrative (1.68× N≤12, 0.33× N>12) |
| `HANDOVER_QDP_PR_WORKFLOW.md` | How to test, pre-commit, inheritance |
| `BENCHMARK_HANDOVER_APACHE_MOUT.md` | **This file** — file manifest + commands |
| `reports/PR001_Benchmark.md` … `reports/PR006_Benchmark.md` | Recorded numbers per experiment |
| `reports/Comparison_Report.md` | Cross-PR summary (if present) |

### 3.2 Benchmark scripts (under `qdp/qdp-python/benchmark/`)

| Script | Keep? | What it measures | Handover reference? |
|--------|-------|------------------|---------------------|
| **`benchmark_pr6.py`** | **Yes — canonical** | IQP `encode()` vs `encode_batch_tc()`, batch=1024 | **`reports/PR006_Benchmark.md`** |
| `benchmark_latency.py` | Yes | RAM → GPU, supports `iqp` / `iqp-z` | Supplementary |
| `benchmark_throughput.py` | Yes | vectors/sec DataLoader style | Supplementary |
| `benchmark_e2e.py` | Yes | Disk → GPU → forward (cold start) | Not in handover table |
| `utils.py` | Yes | Shared data generation | Required by all above |
| `encoding_benchmarks/qdp_pipeline/svhn_iqp.py` | Optional | Real ML IQP (SVHN) | ML scenario only |
| `benchmark_pytorch_ref.py` | Optional | PyTorch reference | Not strict GPU baseline |

### 3.3 Branch-only code (copy when switching)

These exist only on experiment branches—**re-checkout from the branch** after switching:

```bash
# From any branch, pull canonical IQP kernel benchmark onto main:
git checkout pr6-tensor-core-acceleration -- qdp/qdp-python/benchmark/benchmark_pr6.py

# Pull latest PR006 report if you were on another branch:
git checkout pr6-tensor-core-acceleration -- reports/PR006_Benchmark.md

# Pull all handover + reports onto a code branch (e.g. before benchmarking):
git checkout main -- HANDOVER_PR1_PR6.md HANDOVER_QDP_PR_WORKFLOW.md BENCHMARK_HANDOVER_APACHE_MOUT.md reports/
```

### 3.4 Stashed / WIP (not on `main` yet)

| Item | Location | Notes |
|------|----------|-------|
| E2E IQP + `mahout-tc` support | `git stash list` → `e2e-iqp-tc-pr6` | Applied on `pr6-tensor-core-acceleration` only |

```bash
git checkout pr6-tensor-core-acceleration
git stash apply stash^{/e2e-iqp-tc-pr6}   # if needed
```

---

## 4. How to run benchmarks (WSL + RTX 4060)

### 4.1 One-time setup

```bash
cd /path/to/apache_mout
make benchmark
# or manually:
source .venv_wsl/bin/activate
export PATH=/usr/local/cuda/bin:$PATH
maturin develop --release --manifest-path qdp/qdp-python/Cargo.toml
```

**Use `--release` for handover numbers.** Debug builds skew N≤12 TC comparison.

### 4.2 Canonical score (matches handover table)

```bash
cd qdp/qdp-python/benchmark
python benchmark_pr6.py
```

| Setting | Value |
|---------|-------|
| GPU | RTX 4060 |
| Batch | 1024 |
| Precision | float64 |
| Compare | `encode(data, N, "iqp")` vs `encode_batch_tc(data, N)` |

Record results in `reports/PR006_Benchmark.md`.

### 4.3 Supplementary benchmarks

```bash
# IQP RAM → GPU (no disk)
python benchmark_latency.py --qubits 12 --batches 50 --batch-size 64 \
  --frameworks mahout --encoding-method iqp

# E2E cold start (amplitude / angle / basis on stock script)
python benchmark_e2e.py --qubits 12 --samples 64 \
  --frameworks mahout-arrow --encoding-method amplitude
```

E2E numbers **do not** match `benchmark_pr6.py` — different pipeline (disk IO, batch FWT path).

### 4.4 Strict GPU vs GPU (before/after change)

```bash
git checkout <baseline-commit>
maturin develop --release --manifest-path qdp/qdp-python/Cargo.toml
python qdp/qdp-python/benchmark/benchmark_pr6.py | tee before.txt

git checkout <optimized-commit>
maturin develop --release --manifest-path qdp/qdp-python/Cargo.toml
python qdp/qdp-python/benchmark/benchmark_pr6.py | tee after.txt
```

---

## 5. Checkout workflow (typical)

```bash
# 1. Work on integrated code
git checkout pr6-tensor-core-acceleration

# 2. Bring docs + reports from main
git checkout main -- HANDOVER_PR1_PR6.md HANDOVER_QDP_PR_WORKFLOW.md \
  BENCHMARK_HANDOVER_APACHE_MOUT.md reports/

# 3. Ensure canonical benchmark script is present
git checkout pr6-tensor-core-acceleration -- qdp/qdp-python/benchmark/benchmark_pr6.py

# 4. Build release + run
export PATH=/usr/local/cuda/bin:$PATH
maturin develop --release --manifest-path qdp/qdp-python/Cargo.toml
python qdp/qdp-python/benchmark/benchmark_pr6.py

# 5. Promote numbers back to main
git checkout main
# edit reports/PR006_Benchmark.md, commit, push origin main
```

---

## 6. What the handover scores mean (quick reference)

From `reports/PR006_Benchmark.md` on RTX 4060, batch=1024, **release build**:

| N | FWT (`encode`) | TC (`encode_batch_tc`) | Speedup |
|---|----------------|------------------------|---------|
| 12 | ~14 ms | ~8 ms | **~1.68×** |
| 14 | ~69 ms | ~207 ms | **~0.33×** |

- **N ≤ 12:** TC fused shared-memory wins.
- **N > 12:** TC (Implicit Hadamard) slower—malloc + transpose overhead.

This is **kernel-only** timing—not E2E, not PennyLane.

---

## 7. Push targets

| Content | Push to |
|---------|---------|
| Handover + `reports/` | `origin main` (`apache_mout`) |
| Experiment code | `origin pr6-tensor-core-acceleration` (create on remote if missing) |
| Upstream PR | `mahout_fork` only when ready for review |

```bash
git push origin main
git push origin pr6-tensor-core-acceleration
```
