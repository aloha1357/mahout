# Tensor Core IQP path remediation

This PR updates the Tensor Core IQP path from the earlier scaffold into a merge-ready implementation addressing the review feedback on #1387.

The main goals are:

- remove unused experimental `AdaptiveOzakiEngine` code
- fix Tensor Core launcher error propagation
- validate `encode_batch_tc` directly with deterministic correctness tests
- keep Tensor Core behavior explicit opt-in and separate from the default IQP path

## What changed

### Dead code removal

The previous branch carried an unused `AdaptiveOzakiEngine` implementation and an unused `launch_adaptive_ozaki_gemm` FFI surface. That code was not part of the IQP Tensor Core path and made the PR broader than necessary.

This PR removes that path entirely:

- deleted `qdp/qdp-kernels/src/AdaptiveOzaki.cu`
- deleted `qdp/qdp-kernels/src/AdaptiveOzaki.h`
- removed `AdaptiveOzaki.cu` from `build.rs`
- removed `launch_adaptive_ozaki_gemm` FFI declaration and non-CUDA stub from `lib.rs`
- extracted the minimal shared Ozaki config into `ozaki_config.h`

### CUDA error handling

The `N <= 12` Tensor Core path now checks CUDA failures instead of silently returning success:

- captures and checks `cudaFuncSetAttribute`
- launches the shared-memory TC kernel
- calls `cudaStreamSynchronize(stream)`
- returns any synchronization error
- otherwise returns the final `cudaGetLastError()`

### Test coverage

The test suite now exercises `encode_batch_tc` directly instead of only the default `encode` path.

| Test area | Coverage |
|---|---|
| Deterministic inputs | seeds `42` and `137` |
| Independent oracle | `torch_iqp_encode_ref` CPU formula reference |
| Small-N correctness | `N = 6, 8, 10, 12`, `max_err < 1e-9` |
| Large-N correctness | `N = 14, 16`, `max_err < 1e-5` |
| Existing TC tests | preserved and moved to deterministic inputs |
| Binding smoke tests | unchanged behavior verified |

## Reviewer feedback addressed

| Review point | Resolution |
|---|---|
| TC API returns invalid IQP state | Direct TC path tests validate against an independent CPU formula oracle |
| Full-ZZ TC batches read wrong sample data | TC path accounts for the full IQP parameter count |
| Test did not exercise TC path | Tests now call `encode_batch_tc` directly |
| TC launcher hid CUDA failures | `cudaFuncSetAttribute` is checked; stream is synchronized before final error check |
| Random-only tests / tolerance unclear | Deterministic seeds and documented `1e-9` / `1e-5` tolerances |
| Scaffolding should not merge to `main` | Branch is now a complete TC-path remediation, not scaffold-only |

## Path separation

This PR does **not** introduce the later Native/Fused FP32 Hadamard work. That will be proposed separately as a new explicit opt-in path.

| Path | Status in this PR |
|---|---|
| Default IQP `encode` path | preserved |
| Tensor Core `encode_batch_tc` path | remediated and directly tested |
| Native/Fused FP32-FP64 path | separate future PR |

## Benchmark characterization

This PR is a correctness/remediation PR, not a speedup PR. The benchmark below is included to characterize the remediated `encode_batch_tc` path against the default FWT encode path on the pushed remediation commit.

**Commit:** `6497bbbe7`  
**Workload:** IQP full-ZZ, deterministic seed `42`, WSL/CUDA local environment  
**Metric:** median direct encode latency, 10 timed iterations after warmup

![PR1387 TC benchmark](https://raw.githubusercontent.com/aloha1357/apache_mout/internal-dev-notes/dev_notes/pr1387_benchmark/pr1387_tc_benchmark.png)

| N | batch | FWT median ms | TC median ms | TC speedup | max abs err vs FWT | TC norm max err |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 8 | 1.109 | 1.088 | 1.02x | 0.000e+00 | 1.110e-16 |
| 10 | 8 | 1.106 | 1.102 | 1.00x | 0.000e+00 | 1.110e-16 |
| 12 | 8 | 1.713 | 1.265 | 1.35x | 0.000e+00 | 0.000e+00 |
| 14 | 8 | 1.967 | 2.134 | 0.92x | 1.185e-09 | 7.914e-10 |
| 16 | 8 | 5.570 | 17.097 | 0.33x | 3.142e-17 | 1.110e-16 |
| 18 | 4 | 13.838 | 62.159 | 0.22x | 2.296e-17 | 0.000e+00 |

The current direct TC encode path is numerically consistent with the default FWT path on these measured cases. Larger-N direct encode is not presented as a latency win in this PR; Native/Fused work is planned as a separate performance PR.

## Validation

Validated locally in the repository WSL environment:

| Command | Result |
|---|---|
| `cargo build -p qdp-kernels` | passed |
| `cargo test -p qdp-kernels -q` | passed |
| `cargo clippy -p qdp-kernels -- -D warnings` | passed |
| `pytest testing/qdp/test_iqp_tc_path.py -v` | 25 passed |
| `pytest testing/qdp/test_iqp_tc_path.py -q` | passed |
| `pytest testing/qdp/test_bindings.py -q` | 80 passed, 2 skipped |
| `pytest testing/qdp/ -q` | 165 passed, 2 skipped |
| `ruff check qdp/` | passed |
| `ty check qdp/` | passed |

## Notes on large N

The implementation is intended to support larger Tensor Core IQP workloads, but the automated CPU-oracle correctness tests intentionally stop at `N = 16`.

For `N = 27`, a full state has `2^27 = 134,217,728` amplitudes, so a full CPU-oracle max-error test is not appropriate for this PR gate. Larger-N coverage should use smoke/invariant tests and benchmark-specific validation rather than materializing a full CPU reference in CI.
