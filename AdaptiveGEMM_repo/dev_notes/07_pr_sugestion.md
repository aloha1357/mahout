**現在真正需要做的是 NCU（Nsight Compute）把 bottleneck 與 memory boundary 定量化**

而且重點是：

> 不要一開始就把 NCU 當成「看全部 metrics」。

那會直接資訊爆炸。

你現在應該是：

```text
Question-driven profiling
```

先定義問題，再收集 metrics。

---

# 你現在真正想回答的問題

我幫你整理成 4 個 research questions。

---

## Q1. 為什麼 TC runtime 幾乎是平的？

你之前：

```text
N14 ≈ 80us
N16 ≈ 76us
```

非常奇怪。

你想知道：

```text
launch overhead dominated ?
occupancy fixed ?
memory bound ?
```

### NCU 要看

#### Compute utilization

```text
sm__throughput.avg.pct_of_peak_sustained_elapsed
```

看：

```text
SM 有沒有真的忙
```

---

#### Tensor Core utilization

```text
smsp__inst_executed_pipe_tensor.sum
```

看：

```text
Tensor core 指令到底有沒有吃滿
```

---

#### Occupancy

```text
sm__warps_active.avg.pct_of_peak_sustained_active
```

看：

```text
是不是 occupancy 卡住
```

---

#### Kernel duration

直接看：

```text
GPU time
```

---

如果看到：

```text
SM util 低
tensor util 低
runtime 幾乎固定
```

那通常：

```text
launch overhead dominated
```

---

## Q2. 你到底是 compute bound 還 memory bound？

這是 paper 很重要的一句。

### NCU 要看

#### DRAM throughput

```text
dram__throughput.avg.pct_of_peak_sustained_elapsed
```

---

#### L2 throughput

```text
lts__throughput.avg.pct_of_peak_sustained_elapsed
```

---

#### Shared memory throughput

```text
l1tex__data_pipe_lsu_wavefronts_mem_shared.sum
```

因為你 heavily shared-memory fusion。

---

判讀：

### Case A

```text
dram 90%
sm 20%
```

=

```text
memory bound
```

---

### Case B

```text
sm 90%
dram 20%
```

=

```text
compute bound
```

---

### Case C

```text
tensor pipe high
dram moderate
```

=

你 mapping 很成功。

---

## Q3. N=28 為什麼炸？

這超重要。

現在懷疑：

```text
misaligned address
```

其實是：

```text
memory pressure boundary
```

而不是 indexing bug。

### NCU 要看

#### Shared memory usage

```text
launch__shared_mem_per_block
```

---

#### Register pressure

```text
launch__registers_per_thread
```

---

#### Local memory spill

```text
smsp__sass_average_data_bytes_per_sector_mem_local
```

看：

```text
register spill
```

---

#### Achieved occupancy

看：

```text
是不是 resource exhaustion
```

---

如果：

```text
shared ↑
register ↑
occupancy ↓
```

然後：

```text
28 fail
```

你就能合理說：

```text
resource saturation
```

---

## Q4. 為什麼 Tensor Core 比 baseline scaling 完全不同？

這其實是最漂亮的一張圖。

要看：

### Roofline-ish behavior

你不用正式 roofline。

只要：

#### FLOP-ish proxy

```text
sm__throughput
```

#### memory proxy

```text
dram__throughput
```

#### runtime

---

做：

| N  | Runtime | SM util | DRAM util |
| -- | ------: | ------: | --------: |
| 14 |         |         |           |
| 16 |         |         |           |
| 20 |         |         |           |
| 24 |         |         |           |
| 26 |         |         |           |
| 27 |         |         |           |

---

你會看到 scaling pattern。

這超像 paper figure。

---

# 現在推薦 profiling 條件

不要全 sweep。

先：

```text
N=14
N=20
N=26
N=27
N=28(borderline)
```

理由：

```text
small
mid
near boundary
boundary
fail
```

---

每次：

### 跑 1 kernel

不要整 pipeline。

先 profile：

```text
ozaki_h_n2
```

或你的 hottest kernel。

因為整 pipeline 太亂。

---

### release build

一定：

```text
-O3
```

不是 debug。

---

### warmup

至少：

```cpp
10 warmups
```

後：

```text
100 iterations
```

---

### 固定 clock（可選）

如果你想嚴格：

用：

```text
nvidia-smi -lgc
```

固定 GPU clock。

避免 boost fluctuation。

---

# 我建議你現在紀錄欄位

做 CSV：

```text
N
kernel_name
runtime_us

sm_util_pct
tensor_inst
dram_util_pct
l2_util_pct
shared_mem_per_block
register_per_thread
occupancy_pct

correctness
status(pass/fail)
```

---

這樣你之後：

* README
* Apache discussion
* paper
* PhD proposal

全部都能直接 reuse。

---

一句話總結現在 NCU 的目標：

> **不是證明它快，而是解釋它為什麼快，以及為什麼 N=28 壞掉。**

這會讓你的結果從 benchmark 變 research。
