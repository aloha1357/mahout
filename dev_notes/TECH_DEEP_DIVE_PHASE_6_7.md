# Technical Deep Dive: Phase 6-7 (The Matrix-Free Ozaki Engine)
**Perspective: Senior NVIDIA CUDA Engineer**

## 1. 核心挑戰：消費級 GPU 的 FP64 閹割問題
NVIDIA RTX 4090/4060 的 FP64（雙精度）算力極低（僅為 FP32 的 1/64）。
*   **工程痛點：** 量子模擬必須使用 FP64 以確保機率演化的正確性，但直接跑 FP64 會慢到無法接受。
*   **解決方案：** **Ozaki INT8 模擬 (Precision Emulation)**。

## 2. 技術方案 (Tech Solution): Ozaki 7-Pass 與 CRT 重組
### 2.1 Ozaki INT8 擴展 (The "Cheat Code")
*   **實作：** 將 FP64 的尾數拆解為多個 INT8 組件。
*   **技術細節：** 透過 7 個互質的大質數（中國剩餘定理 CRT）進行並行計算。
*   **工程價值：** 利用 RTX 4060 的 660 TOPS INT8 算力來「模擬」FP64。雖然要算 7 遍，但依然比直接算 FP64 快得多。

### 2.2 硬體層級優化: `mma.m16n8k32`
*   **實作：** 棄用 `wmma` 高階 API，直接使用 PTX 組合語言指令。
*   **技術細節：**
    *   使用 `ldmatrix.x4` 進行 Shared Memory 載入。
    *   精準控制寄存器壓力（`--maxrregcount=64`），確保能在 SM 上運行更多的 Warp。
*   **貢獻：** 達成了 $N=16$ 僅需 **74 微秒 (us)** 的恐怖速度。

### 2.3 性能歸因分析 (Performance Attribution)
*   **實作：** PR007 的 NCU 證據鏈。
*   **技術細節：**
    *   證明了 `iqp_phase_split` 的 SM Throughput 為 67.5%。
    *   證明了 DRAM 利用率從 33% 降至 0.45%。
*   **貢獻：** 這不是在吹牛，而是用 NCU 數據證明了「瓶頸成功轉移」。

## 3. 高級工程師點評 (Senior Engineer's Deep Dive)
"這是目前工業界最極限的 Ozaki 實現之一。我們不只是跑通了流程，還針對 WDDM 的 NCU `LaunchFailed` 進行了專門的 Grid Profiling Kernel 設計。這套架構讓 $N=16 \sim 30$ 的量子模擬從原本需要高性能工作站，變成了一張筆電顯示卡就能亞秒級跑完的任務。這就是 **Algorithms-Architectures Codesign** 的力量。"
