# PR3: Memory Model Optimization & Operator Fusion

## Scope
Reduce DRAM bandwidth bottlenecks via aggressive shared memory usage and kernel fusion.

## Target Changes
- Shared memory persistent execution models
- Warp shuffle integration for cross-thread data exchanges
- Fused angle/amplitude encoding layers

## Review Focus
- Nsight Compute (NCU) analysis for L2/DRAM traffic reduction
- Occupancy impact from shared memory pressure

## Merge Blockers
- Performance regressions
- Shared memory overallocation on smaller GPUs (e.g., T4)
