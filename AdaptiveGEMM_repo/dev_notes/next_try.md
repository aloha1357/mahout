# DEVELOPMENT_GUIDE.md

# FWHT on INT8 Tensor Cores

## TDD-Driven CUDA/HPC Development Guide

Version: 0.1

Audience:

* CUDA Kernel Engineers
* HPC Engineers
* Performance Architects
* GPU Compiler/Runtime Engineers
* Researchers investigating Tensor Core acceleration of Fast Walsh-Hadamard Transform (FWHT/FWT)

---

# 1. Executive Summary

This project investigates whether a Fast Walsh-Hadamard Transform (FWHT)

[
y = H_n x
]

can be mapped efficiently onto INT8 Tensor Cores.

The primary engineering challenge is NOT arithmetic throughput.

The primary challenge is:

```text
Communication
>
Computation
```

For sufficiently large transforms:

```text
registers
↓
shared memory
↓
cross-CTA communication
↓
L2
↓
HBM/DRAM
```

dominates runtime.

Therefore:

```text
Goal:
Maximize locality

NOT

Maximize Tensor Core utilization
```

This guide intentionally follows:

```text
Correctness
→ Profiling
→ Roofline
→ Optimization
→ Tensorization
```

rather than:

```text
Tensor Core first
```

which is usually the wrong optimization order.

---

# 2. Engineering Principles

## Principle 1

Never optimize what is not measured.

Every optimization must be justified by:

* Nsight Compute
* Roofline
* Microbenchmark

---

## Principle 2

Communication dominates large FWHT.

Assume:

```text
Large N

Memory Traffic
>
Tensor Core Throughput
```

until proven otherwise.

---

## Principle 3

Warp shuffle is the baseline.

Tensor Core implementation must outperform:

```cpp
__shfl_xor_sync()
```

otherwise it is rejected.

---

## Principle 4

Kernel fusion before tensorization.

Preferred order:

```text
Stage Fusion
↓
Shared Memory Optimization
↓
Communication Reduction
↓
Tensor Core Mapping
```

---

# 3. Success Criteria

Project is successful if:

```text
Correctness:
    bit-exact (int32 reference)

Performance:
    >1.5x over optimized SIMT baseline

Scalability:
    scales beyond one SM

Occupancy:
    >25%

Tensor Core Utilization:
    >50%

Memory Throughput:
    not DRAM bound
```

---

# 4. Repository Structure

```text
fwht-tc/

├── docs/
│   ├── DEVELOPMENT_GUIDE.md
│   ├── ARCHITECTURE.md
│   └── ROOFLINE.md
│
├── reference/
│   ├── fwht_cpu.cpp
│   └── fwht_cpu_int32.cpp
│
├── tests/
│   ├── test_correctness.cu
│   ├── test_overflow.cu
│   └── test_quantization.cu
│
├── kernels/
│   ├── fwht_warp.cu
│   ├── fwht_smem.cu
│   ├── fwht_fused.cu
│   ├── fwht_tensorcore.cu
│   └── fwht_hierarchical.cu
│
├── benchmarks/
│   ├── benchmark_warp.cu
│   ├── benchmark_tensorcore.cu
│   └── benchmark_scaling.cu
│
├── profiling/
│   ├── ncu_reports/
│   └── scripts/
│
└── experiments/
    ├── exp01_local_fwht
    ├── exp02_fusion
    ├── exp03_tensorcore
    └── exp04_hierarchical
```

---

# 5. Development Roadmap

Do NOT jump directly to Tensor Cores.

Mandatory sequence:

```text
TDD-0
Reference

TDD-1
Warp FWHT

TDD-2
Shared Memory FWHT

TDD-3
Stage Fusion

TDD-4
Hierarchical FWHT

TDD-5
Tensor Core Microkernel

TDD-6
Hybrid FWHT

TDD-7
Communication Avoiding FWHT
```

---

# TDD-0

## CPU Reference

Implement:

```cpp
void fwht_reference(
    int32_t* x,
    int n);
```

Requirements:

* deterministic
* no overflow
* int32 accumulation

Test:

```text
N=2
N=4
N=8
N=16
...
N=65536
```

Verification:

```cpp
gpu_result == cpu_result
```

---

# TDD-1

## Warp-Level FWHT

Target:

```text
N <= 32
```

Implementation:

```cpp
__shfl_xor_sync()
```

Example:

```cpp
for(int d=1; d<32; d<<=1)
{
    int y =
      __shfl_xor_sync(
         0xffffffff,
         x,
         d);

    if((lane & d)==0)
        x += y;
    else
        x = y - x;
}
```

Goals:

* no shared memory
* no global sync

Expected:

```text
Tensor Core not needed
```

---

## Test

Random:

```text
10000 vectors
```

Compare against:

```text
CPU reference
```

Tolerance:

```text
exact
```

---

# TDD-2

## Shared Memory FWHT

Target:

```text
32 < N <= 1024
```

Design:

```text
register
↓
shared memory
↓
register
```

No intermediate global store.

---

## Metrics

Monitor:

```text
Shared Load Efficiency
Shared Store Efficiency
Bank Conflicts
```

Using:

```text
Nsight Compute
```

Key counters:

```text
smsp__sass_average_branch_targets_threads_uniform.pct

l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum
```

---

## Stop Condition

Reject design if:

```text
Bank Conflict > 2x
```

relative to baseline.

---

# TDD-3

## Stage Fusion

This phase usually produces the largest speedup.

Bad:

```text
kernel
kernel
kernel
kernel
```

Good:

```text
kernel
{
    stage1
    stage2
    stage3
}
```

---

## Objective

Reduce:

```text
Global Memory Traffic
```

Measure:

```text
dram bytes
```

Expected:

```text
>30% reduction
```

---

# TDD-4

## Hierarchical FWHT

This is where scalability starts.

Problem:

```text
N > CTA size
```

Cross-SM communication appears.

---

## Decomposition

Example:

```text
FWHT(4096)

↓

FWHT(64)
FWHT(64)
FWHT(64)
...
```

Local transforms first.

Then merge.

---

## Philosophy

Borrow from FFT.

Avoid:

```text
Global Synchronization
Every Stage
```

Prefer:

```text
Local Transform
↓
Merge
↓
Local Transform
```

---

## Test

Scaling:

```text
1k
2k
4k
8k
16k
32k
64k
```

Plot:

```text
runtime
vs
N log N
```

---

# TDD-5

## Tensor Core Microkernel

Only start here.

---

## Goal

Map:

```text
[a+b]
[a-b]
```

into matrix form.

Small Hadamard block:

```text
H2
H4
H8
H16
```

---

## Candidate Tiles

Ampere:

```text
m16n8k32
```

Ada:

```text
m16n8k32
```

Hopper:

```text
m16n8k32
m16n16k16
```

depending on path.

---

## Restriction

Tensor Core only for:

```text
local dense block
```

Never tensorize entire FWHT initially.

---

## Validation

Compare:

```text
warp shuffle
```

vs

```text
mma.sync
```

for:

```text
16-point
32-point
64-point
```

---

## Success

Tensor Core path must achieve:

```text
>1.2x
```

speedup.

Otherwise discard.

---

# TDD-6

## Hybrid Architecture

Recommended production architecture.

---

Structure:

```text
Large FWHT

├─ Tensor Core
│    local block
│
└─ SIMT
     merge stages
```

---

Reason

Tensor Core excels at:

```text
dense compute
```

FWHT merge stages are:

```text
sparse communication
```

SIMT is better.

---

# TDD-7

## Communication-Avoiding FWHT

Final optimization phase.

---

Question:

Not:

```text
How do we use Tensor Cores?
```

Instead:

```text
How do we reduce communication?
```

---

Techniques:

### Persistent Kernel

Avoid:

```text
launch
sync
launch
sync
```

Prefer:

```cpp
while(stage<n)
{
   ...
}
```

---

### Cooperative Groups

Use:

```cpp
grid_group
```

when supported.

Tradeoff:

```text
Occupancy ↓
Synchronization ↓
```

---

### Recursive Decomposition

Preferred for very large transforms.

---

# Quantization Strategy

Tensor Core path:

```text
Input:
INT8

Accumulate:
INT32
```

---

Worst Case

Each stage doubles range.

For:

```text
N=1024
```

Maximum:

```text
127 * 1024
```

---

Policy

Keep:

```text
INT32
```

for entire transform.

Only requantize:

```text
Final Output
```

---

Reject

```text
INT8 every stage
```

unless accuracy study proves acceptable.

---

# Occupancy Checklist

For every kernel:

Record:

```text
Registers / Thread

Shared Memory / CTA

Warps / SM

Occupancy
```

---

Reject if:

```text
Occupancy < 25%
```

without measurable gain.

---

# Nsight Compute Checklist

Every optimization requires:

```text
SM Occupancy

Tensor Core Utilization

DRAM Throughput

L2 Hit Rate

Shared Memory Efficiency

Bank Conflicts

Warp Stall Reasons
```

---

Most Important

```text
sm__throughput

dram__throughput

smsp__warps_active.avg

smsp__warp_issue_stalled*
```

---

# Decision Tree

## Tensor Core Slower?

Check:

```text
Data Reordering
```

---

If:

```text
Reordering Cost
>
Compute Savings
```

Return to:

```text
SIMT
```

---

## Occupancy Collapse?

Check:

```text
Registers
```

and

```text
Accumulator Count
```

---

## DRAM Bound?

Do NOT optimize Tensor Core.

Optimize:

```text
Fusion
Hierarchy
Communication
```

instead.

---

# Production Recommendation

For modern GPUs:

```text
Warp Shuffle
+
Shared Memory
+
Stage Fusion
+
Hierarchical Decomposition
+
Selective Tensor Core Usage
```

is expected to outperform:

```text
Full Tensor-Core FWHT
```

for most practical transform sizes.

The engineering objective is therefore:

```text
Minimize Communication

not

Maximize Tensor Core Usage
```

This principle should guide every benchmark, optimization, and architectural decision in the project.
