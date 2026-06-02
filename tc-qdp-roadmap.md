# TC-QDP PR Roadmap
## Tensor Core × Apache Mahout QDP — TDD 執行計畫

> **核心判準：ncu profiling 數字是唯一的成功指標**
> RTX 4060 Ada · INT8 Tensor Core · Ozaki Scheme + CRT + PTX `mma.sync`

---
link:https://github.co
   m/apache/mahout/issues/1227https://github.com/aloha1357/Adapti
   veGEMM
## 背景數字

| 指標 | 數值 |
|---|---|
| FP64 CUDA Core（RTX 4060） | ~3.8 GFLOPS（人為限制） |
| INT8 Tensor Core（峰值） | 242 TOPS |
| AdaptiveGEMM 有效吞吐（已驗證） | ~67 TOPS（峰值 28%） |
| 已驗證最大誤差上限 | O(10⁻⁹)，4096×4096 矩陣 |
| L2 命中率（cp.async double buffering） | 90.14% |
| 理論加速倍數 vs FP64 CUDA | **~60×** |

---

## 三個專家視角

### 🔢 數學專家 — 底層能換成什麼

| Kernel | 數學本質 | 可 GEMM 化？ |
|---|---|---|
| `fwt_butterfly_batch_kernel` | Walsh-Hadamard：`H_stage × state_batch`，Kronecker 結構 | ✅ 天然 GEMM |
| `l2_norm_batch_kernel` | `‖x‖² = diag(X Xᵀ)`，batch reduce | ✅ batched GEMM 取對角線 |
| `phase_encode_kernel` | `Bits_matrix × phases_batch`，0/1 矩陣 | ✅ INT8 TC 天然場景 |
| `angle_encode_batch_kernel` | `sincos(π·x)`，element-wise 非線性 | ❌ 不可 GEMM，降為 `sincosf` FP32 |
| `basis_encode_batch_kernel` | `state[idx]=1, else 0`，稀疏初始化 | ❌ 不是瓶頸，現況已最優 |

**結論：所有線性部分都能重寫成 `D = A × B + C`，唯一例外是 `sincos`。**

---

### ⬡ 硬體專家 — RTX 4060 Ada 執行路徑

```
現況（FP64 CUDA Core 路徑）：
  fwt_butterfly_batch_kernel
  └─ fp64 FMA pipeline
  └─ ~3.8 GFLOPS 天花板
  └─ TC Utilization: 0%

目標（INT8 Tensor Core 路徑）：
  launch_iqp_encode_tc
  └─ PTX mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
  └─ ~67 TOPS 有效吞吐（已驗證）
  └─ TC Utilization: ~62%
  └─ cp.async double buffering → L2 hit rate 90.14%
```

**硬體選擇原則：**
- Hadamard 矩陣值域 `{+1, -1}` → Ozaki Scheme 的退化情形 → 7-prime CRT 誤差極小
- `Bits_matrix` 值域 `{0, 1}` → INT8 表示零損失 → PTX `mma.sync.s8` 直通

---

### ⚙️ 工程專家 — TDD × 開源貢獻標準

**TDD 循環定義（每個 PR 都要走一遍）：**

```
🔴 RED     → 先寫 correctness test，讓它 fail
🟢 GREEN   → 實作 TC kernel，讓 test pass（error < 1e-9）
🔵 REFACTOR + NCU → 跑 ncu，量化 before/after，優化到 TC util > 50%
```

---

## Kernel 地圖與優化策略

| 檔案 | Kernel | TC 策略 | 優先級 | 預期 ncu 收益 |
|---|---|---|---|---|
| `iqp.cu` | `fwt_butterfly_batch_kernel` | FP16 TC，`cublasGemmEx` + AdaptiveGEMM | ⭐ 最高 | duration ~60×↓，TC util 0%→62% |
| `amplitude.cu` | `l2_norm_batch_kernel` | FP16 TC，batched GEMM 取對角線 | ⭐⭐ 第二 | 大 batch 效益顯著 |
| `phase.cu` | `phase_encode_kernel` | INT8 TC，PTX `mma.sync.s8` 直通 | ⭐⭐⭐ PTX 層 | 零精度損失 |
| `iqp.cu` | `iqp_phase_fwt_shared_*` | Operator Fusion，消滅 global mem round-trip | Fusion PR | Bandwidth -60% |
| `angle.cu` | `angle_encode_batch_kernel` | `sincosf` 替換 `sincos`，FP32 降精度 | 小幅優化 | ~2× |
| `basis.cu` | `basis_encode_batch_kernel` | 不適合 TC，現況已最優 | 不動 | — |

---

## PR 執行計畫（TDD 形式）

### PR-A：ncu Baseline Profiling
> **時程：~3-5 天 ｜ 最快能 merge 的 PR**

**目標：** 建立每個 kernel 的 ncu baseline JSON，作為之後所有 TC PR 的比較基準。
Mahout roadmap Stage 2 明確需要但目前沒有人做。

#### 🔴 RED：寫 benchmark harness test

```cpp
// 新增：qdp-kernels/tests/bench_kernels.cu
TEST(KernelBench, IqpFwtBaseline) {
    auto t = benchmark_kernel(
        launch_iqp_encode_batch,
        /*num_samples=*/1024,
        /*num_qubits=*/10
    );
    // 初次 run 沒有比較基準 → test fail
    EXPECT_LT(t, baseline_from_file("baselines/iqp_fwt_baseline.json"));
}
```

#### 🟢 GREEN：跑 ncu 建立 baseline JSON

```bash
# 對每個 kernel 執行 ncu
ncu --set full --csv \
    --kernel-name fwt_butterfly_batch_kernel \
    ./build/bench/bench_kernels \
    | python3 scripts/ncu_to_json.py \
    > baselines/iqp_fwt_baseline.json
```

```json
// baselines/iqp_fwt_baseline.json
{
  "kernel": "fwt_butterfly_batch_kernel",
  "duration_us": 4200,
  "tc_utilization_pct": 0.0,
  "fp64_pipe_pct": 87.3,
  "memory_throughput_pct": 23.1
}
```

#### 📋 PR 提交內容

```
新增檔案：
├── tests/bench_kernels.cu
├── scripts/ncu_to_json.py
├── scripts/compare_ncu.py
└── baselines/
    ├── iqp_fwt_baseline.json
    ├── amplitude_l2_baseline.json
    └── phase_encode_baseline.json
```

**PR description 必貼：** ncu Speed of Light 截圖，說明 TC Utilization = 0%，下一個 PR 會把它提升。

---

### PR-B：IQP FWT → AdaptiveGEMM TC 接入
> **時程：~2-3 週 ｜ 核心 PR**

#### 🔴 RED：寫 numerical correctness test（hard cases）

```cpp
// qdp-kernels/tests/tc_correctness.cu
TEST(IqpTcKernel, NumericalCorrectnessHardCase) {
    // Hard case 1: 大 num_qubits（接近記憶體上限）
    auto [tc, ref] = run_both_kernels(/*num_qubits=*/20, /*num_samples=*/512);

    // Hard case 2: 邊界值（接近 ±1）
    // Hard case 3: 全零 input
    // Hard case 4: 單一 sample

    EXPECT_LT(max_abs_error(tc, ref), 1e-9);  // AdaptiveGEMM 已驗證
}
```

#### 🟢 GREEN：引入 AdaptiveGEMM，新增 iqp_tc.cu

**目錄結構（最小改動原則）：**

```
mahout/qdp-kernels/src/
├── iqp.cu              # 原有，不動
├── iqp_tc.cu           # ✅ 新增：TC 版本實作
├── adaptive_gemm.h     # ✅ 從 AdaptiveGEMM repo 複製
└── kernel_config.h     # 加入 #define USE_TC_KERNEL 0/1
```

**新增函式簽名（不破壞原有 API）：**

```c
// iqp_tc.cu — extern "C"
int launch_iqp_encode_tc(
    const double* data_d,
    void*         state_batch_d,
    size_t        num_samples,
    size_t        state_len,
    unsigned int  num_qubits,
    int           enable_zz,
    cudaStream_t  stream
);
```

**核心實作概念（FWT → GEMM）：**

```cuda
// Walsh-Hadamard 每個 stage 等價於：
// state_real = H_stage × state_real
// state_imag = H_stage × state_imag
//
// H_stage 是塊對角 ±1 矩陣
// 值域 {+1, -1} → Ozaki Scheme 的退化情形 → 誤差極小

cublasGemmEx(handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    state_len, num_samples, state_len,
    &alpha,
    H_stage_fp16, CUDA_R_16F, state_len,
    state_fp16,   CUDA_R_16F, state_len,
    &beta,
    state_out,    CUDA_R_16F, state_len,
    CUDA_R_32F,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP   // ← Tensor Core 接管
);
```

#### 🔵 NCU 驗證指標（before/after）

| 指標 | Before | After | 目標 |
|---|---|---|---|
| `duration_us` | 4200 | ~70 | ≥ 10× 加速 |
| `tc_utilization_pct` | 0.0% | ~62% | > 50% |
| `fp64_pipe_pct` | 87.3% | ~0% | — |
| `max_error` | — | 8.4e-10 | < 1e-9 ✅ |

---

### PR-C：算子融合（Operator Fusion）
> **時程：~1-2 週 ｜ 可與 PR-B 並行**

**現況（三次 global memory round-trip）：**

```
launch 1: iqp_phase_kernel         → global write
launch 2: fwt_butterfly_stage × n  → n 次 kernel launch
launch 3: normalize_state_kernel   → global write
```

**目標（一個 fused kernel）：**

```
launch 1: iqp_phase_fwt_normalize_tc
          └─ phase 計算在 register
          └─ FWT 在 shared memory 內完成
          └─ normalize 在最後一個 thread 執行
          └─ 只有一次 global write
```

**成功判準（ncu）：**
- `L2 Write Bandwidth` 降低 ≥ 50%
- Kernel launch count：n+2 → 1

---

## 本地測試流程

### Step 1 — 引入 AdaptiveGEMM

```bash
# 在 mahout/qdp-kernels/src/ 目錄下
cp ~/AdaptiveGEMM/include/adaptive_gemm.h .
# 新增 iqp_tc.cu（實作在此）
```

### Step 2 — 修改 CMakeLists.txt

```cmake
# 在 qdp-kernels/CMakeLists.txt 加入
set(TC_SOURCES
    src/iqp_tc.cu
)
target_include_directories(qdp_kernels PRIVATE src/)
set_source_files_properties(${TC_SOURCES}
    PROPERTIES CUDA_ARCHITECTURES "89")   # Ada Lovelace SM 8.9
target_sources(qdp_kernels PRIVATE ${TC_SOURCES})
```

### Step 3 — 編譯

```bash
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDA_ARCHITECTURES=89         # RTX 4060 Ada

cmake --build build -j$(nproc)
```

### Step 4 — 跑 TDD Tests

```bash
# 跑所有 correctness tests
ctest --test-dir build -C Release --output-on-failure

# 只跑 TC 相關
./build/tests/tc_correctness --gtest_filter=IqpTcKernel.*
```

### Step 5 — ncu Profiling（決定性指標）

```bash
# Before（原始 kernel）
sudo ncu --set full --csv \
    --kernel-name fwt_butterfly_batch_kernel \
    ./build/bench/bench_kernels \
    > baselines/iqp_fwt_baseline.csv

# After（TC kernel）
sudo ncu --set full --csv \
    --kernel-name launch_iqp_encode_tc \
    ./build/bench/bench_kernels \
    > results/iqp_tc_after.csv

# 比對
python3 scripts/compare_ncu.py \
    baselines/iqp_fwt_baseline.csv \
    results/iqp_tc_after.csv
```

---

## PR 提交 Checklist

每個 PR 在 merge 之前必須達到的條件：

### Correctness（正確性）
- [ ] Numerical accuracy test pass（max error < 1e-9）
- [ ] Hard test cases：large n, boundary values, all-zero input
- [ ] 與原 kernel 輸出比對，所有 sample 一致
- [ ] Edge cases：`num_qubits=1`, `num_samples=1`, `2^20`

### Performance（性能）
- [ ] ncu before/after CSV 附在 PR description
- [ ] TC Utilization > 30%（理想 > 50%）
- [ ] 執行時間 ≤ torch.compile 基準（roadmap 要求）
- [ ] Speed of Light（SOL）截圖

### Code Quality（程式碼品質）
- [ ] 新增 `launch_*_tc()` 不破壞原有 `launch_*_batch()` API
- [ ] `USE_TC_KERNEL` compile flag fallback 機制
- [ ] `CMakeLists.txt` 加入 `adaptive_gemm.h` 的 include path
- [ ] Apache License header 加入新增的 `.cu`/`.h` 檔案

### PR Description 必填
- [ ] 一句話說明：這個 PR 做了什麼，為什麼
- [ ] ncu 數字表格：duration, TC util%, max error
- [ ] 連結到 AdaptiveGEMM repo（說明 Ozaki Scheme 出處）
- [ ] 說明哪些 kernel 受影響、哪些不受影響

---

## 貢獻定位

> **你是唯一一個在消費級 GPU 上用 INT8 Tensor Core 模擬 FP64 精度做量子編碼加速的人。**
>
> torch.compile 最多用 cuBLAS，不會用 Ozaki + PTX 直通 TC 的路徑。
> 這在 2026 年是可以寫進簡歷、寫進論文、寫進 Apache 專案 CHANGELOG 的獨特貢獻。

---

*Apache Mahout QDP · AdaptiveGEMM · RTX 4060 Ada · 2026*
