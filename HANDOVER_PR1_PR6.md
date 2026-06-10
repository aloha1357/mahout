# Handover Report: QDP CUDA Optimization (PR1 to PR6)

## 總結 (Executive Summary)
本文件總結了 QDP (Quantum Data Processing) 專案中針對 IQP (Instantaneous Quantum Polynomial) 編碼所進行的一系列漸進式 CUDA 核心效能優化與張量核心 (Tensor Core) 加速實驗。
這個優化旅程從 PR1 進行到 PR6，探討了減少記憶體瓶頸、算子融合 (Operator Fusion)、以及克羅內克積分解 (Kronecker Product Decomposition) 配合張量核心的各種策略。

本文件旨在為未來的維護者與開發者提供技術交接，解釋為何特定的優化在特定的 Qubit 數量 (N) 下有極佳的表現，以及為何在大規模 Qubit 時張量核心策略遇到了效能瓶頸。

---

## 為什麼 N <= 12 的 Tensor Core 加速表現很好，而 N > 12 卻暴跌？

在 PR6 的架構中，我們根據 Qubits (N) 的大小採取了兩種截然不同的計算路徑：

### 1. N <= 12 (表現優異：1.68x 加速)
當 N <= 12 時，量子態向量 (State Vector) 最大包含 $2^{12} = 4096$ 個複數 (每個 16 bytes)，總共剛好是 **64 KB**。
*   **技術：** 這個大小可以完全放入 GPU 的「共享記憶體 (Shared Memory / L1 Cache)」。我們使用了 **算子融合 (Operator Fusion)**，把「相位計算」、「快速哈達瑪轉換 (FWT)」以及「正規化」合併進同一個 CUDA Kernel。
*   **結果：** 資料從全局記憶體 (DRAM) 讀取一次進入共享記憶體後，所有運算都在 GPU 晶片內部的極高速快取完成，最後再一次性寫回 DRAM。這徹底消除了傳統 FWT 的「記憶體頻寬瓶頸 (Memory Bound)」，因此速度直接提升高達 1.68 倍。

### 2. N > 12 (表現崩跌：0.33x 減速)
當 N > 12 時，狀態向量大小超過了共享記憶體的物理極限（約 100KB），我們必須改用另一套基於 **Tensor Core (矩陣乘法)** 的架構。
*   **技術：** 我們利用「克羅內克積分解 (Kronecker Product Decomposition)」硬將一維的 FWT 拆解成一系列的小矩陣乘法 (GEMM)，以利用硬體 Tensor Core 的能力。
*   **瓶頸原因：**
    1.  **動態記憶體分配 (Overhead)：** 為了運作這些小矩陣乘法，迴圈內部使用了同步的 `cudaMalloc` 與 `cudaFree` 來頻繁分配臨時暫存區。
    2.  **資料轉置 (Batch Transpose)：** Tensor Core 要求特定的記憶體排列 (Row-major / Col-major)，我們被迫穿插了大量全域記憶體的矩陣轉置 (Transpose) 核心，導致嚴重的 DRAM 讀寫延遲。
    3.  **高精度 Ozaki 演算法負擔：** 為了讓主要支援 INT8/FP16 的 Tensor Core 能算出精準的「雙精度 (Double / Float64)」結果，我們掛載了複雜的 **Adaptive Ozaki Engine** (將數字拆分進 7 個不同的質數場分開算)。這在超大矩陣中可以攤平開銷，但在 N=14 ($128 \times 128$ 矩陣) 這種相對小規模的 GEMM 中，複雜度的開銷遠大於張量核心算力的收益。

### 3. 為什麼 Baseline 在 N=4 這種極小規模時反而比較快？
當 Qubit 極少（如 N=4）時，總共只有 $2^4 = 16$ 個數字要處理。真正的計算時間可能不到 1 微秒 (μs)。
但每次從 CPU「啟動一個 CUDA Kernel (Kernel Launch Overhead)」以及「準備共享記憶體」的溝通成本高達幾十微秒。在這種「殺雞用牛刀」的情況下，Baseline 簡單粗暴的迴圈反而因為沒有太多的架構準備成本而勝出。這也是為何實務上都會設定一個 `FWT_MIN_QUBITS`，低於這個數量就不啟動 GPU 加速機制。

---

## PR1 到 PR6 實驗發展史與現狀

這六個 PR (Pull Requests) **原本並不存在於 `main` branch 中**。
`main` 分支是專案最乾淨的 Baseline。PR1 到 PR6 是團隊為了測試不同優化技術，而依序從 `main` 切出去的**實驗性分支 (Experimental Branches)**。現在我們將這些實驗結果與報告統整回 `main` 分支的 `reports/` 資料夾，以作為日後正式合併演算法的參考依據。

以下是 PR1 ~ PR6 的技術演進軌跡：

### [PR1] Phase Kernel Optimization (`pr1-phase-kernel-opt`)
*   **目標：** 優化最初版的 FWT 核心。
*   **技術：** 移除 Warp 發散 (Warp Divergence) 造成的等待時間，並將原本放在迴圈內的「正規化 (Normalization)」運算提升 (Hoist) 出來，減少多餘的除法。
*   **結果：** 將核心執行時間從 4200μs 大幅縮減到 27μs (155 倍驚人加速)，確立了極高標準的 Baseline。

### [PR2] Implicit Split Phase & Scaffolding (`pr2-implicit-split-phase`)
*   **目標：** 為未來接入 Tensor Core 做前置資料結構準備。
*   **技術：** 實作了 Bank-Conflict-Free 的共享記憶體矩陣轉置，並將複數 (Complex) 陣列隱式地拆分成純實部 (Real) 與純虛部 (Imaginary) 兩個陣列。

### [PR3] Fused Shared Memory FWT (`pr3-fused-shared-mem-fwt`)
*   **目標：** 針對小 Qubits (N <= 12) 的極致壓榨。
*   **技術：** 把 Phase、FWT、Normalize 融合進共享記憶體。完全避免了迴圈中對全局記憶體的反覆讀寫。
*   **結果：** 在 N=12 以內，對比未優化的 PyTorch Baseline 最高達 533 倍加速。這也是 PR6 裡 N <= 12 表現極佳的核心邏輯來源。

### [PR4] Matrix-Free Kronecker Product (`pr4-kronecker-decomposition`)
*   **目標：** 將 N > 12 的計算轉換成 Tensor Core 認識的形狀。
*   **技術：** 透過數學推導，將全局 FWT 轉換為「矩陣乘法 (GEMM)」與「批次轉置 (Batch Transpose)」的組合，以規避傳統 FWT 演算法在 Tensor Core 上的不相容性。

### [PR5] Adaptive Ozaki Engine (`pr5-adaptive-ozaki-integration`)
*   **目標：** 解決 Tensor Core 的精度問題。
*   **技術：** 由於 Tensor Core 擅長低精度，直接算會喪失量子計算所需的 Double Precision。我們引入了 Mixed-precision 的 Ozaki 算法與剩餘數系統 (Residue Number System)，用 INT8 算出 Float64 等級的精度。

### [PR6] Full Tensor Core Acceleration (`pr6-tensor-core-acceleration`)
*   **目標：** 集大成者。實作動態分支：N <= 12 走 PR3 的共享記憶體融合路徑；N > 12 走 PR4 + PR5 的張量核心 Ozaki 路徑。
*   **修復：** 解決了 `ImplicitHadamardOzaki.cu` 裡 `ldmatrix` 的 16-byte 對齊 (CUDA_ERROR_MISALIGNED_ADDRESS) 的嚴重崩潰問題。
*   **結論：** 證實了「共享記憶體融合」極其成功；但「張量核心分解」的 Overhead 在目前實作下仍然太高，不敵高度優化過後的 PR1 Baseline。未來的維護者需針對記憶體配置與轉置進行深度改造，才能讓 N > 12 的 Tensor Core 路線產生實際加速。
*   **PR6 待完成（與 upstream review 前一併做完）：**
    1. **剝離 hot path `cudaMalloc`** — `iqp_tc.cu` 與 Ozaki GEMM 改用 buffer pool / workspace 重用（見 workflow §9）。
    2. **E2E 驗證** — 在實驗用 `apache_mout`（本 fork）跑 `benchmark_e2e.py`；PR6 擴充 `--encoding-method iqp` / `iqp-z` 後量測 disk→GPU→forward 全鏈路（見 workflow §10）。`benchmark_latency.py` / `throughput` 現已支援 IQP，可先驗證 PR1–PR4 疊加效果。

---

## 標準開發流程（必讀）

所有 PR1–PR6 分支在 push、交接、或 request review 之前，必須遵守同一份流程：

**[HANDOVER_QDP_PR_WORKFLOW.md](HANDOVER_QDP_PR_WORKFLOW.md)**
**[BENCHMARK_HANDOVER_APACHE_MOUT.md](BENCHMARK_HANDOVER_APACHE_MOUT.md)** — benchmark 檔案清單、`origin` 遠端、checkout 後要帶的檔案

重點摘要：

1. **CONTRIBUTING.md** — 測試、pre-commit、PR template、文件更新。
2. **Pre-commit** — 變更檔案必須全數通過（含 `cargo clippy`）；通過後才 push 到 fork。
3. **Unit tests** — 每個 PR 在 `testing/qdp/` 補強測試；PR2 繼承 PR1 的測試，PR3 繼承 PR2，依此類推。
4. **Benchmark** — 嚴格 **GPU baseline vs GPU optimized**（`git checkout` 前一個 commit 量測），禁止用 Naive CPU / PyTorch 當主 baseline；結果寫入 `reports/PR00X_Benchmark.md`。
5. **程式碼風格** — kernel 內禁止 `// PRn Optimization` 等 agent handover 註解；PR 說明寫在 PR description，不寫進 `.cu`。
6. **分支繼承** — PR2 從 PR1 tip 開出，PR3 從 PR2 tip 開出，保留前序 PR 的乾淨實作與測試。

---

## 接下來的維護建議

1.  **確認各分支的報告已合併：** 所有的 Benchmark 報告 (PR001_Benchmark.md 到 PR006_Benchmark.md) 皆已存放於 `main` branch 的 `reports/` 之下；新數據須依 [HANDOVER_QDP_PR_WORKFLOW.md](HANDOVER_QDP_PR_WORKFLOW.md) §5 以 GPU vs GPU 重新量測。
2.  **建議的正式合併策略：** 強烈建議先將 **PR1 (優化 Baseline)** 與 **PR3 (N<=12 共享記憶體加速)** 整合進正式版的 `main` 中，因為它們能提供穩定且無副作用的顯著加速。
3.  **TC 路線的重構：** 關於 PR6 中 N > 12 的路徑，若要商用，必須將 `cudaMalloc` 從執行迴圈中剔除，並研究融合 (Fusion) 轉置步驟，否則其開銷永遠無法回本。
4.  **PR 交接順序：** 完成 PR1 checklist 後再開 PR2；每個 PR 通過 pre-commit 與 benchmark 後才 push fork，準備好再開 upstream review。
5.  **PR6 整合項：** 在 `pr6-tensor-core-acceleration` 完成 cudaMalloc 剝離（buffer pool）、擴充 `benchmark_e2e.py` 支援 IQP，並跑 E2E 驗證 PR1–PR4 疊加效果。詳見 [HANDOVER_QDP_PR_WORKFLOW.md](HANDOVER_QDP_PR_WORKFLOW.md) §9–§10。
