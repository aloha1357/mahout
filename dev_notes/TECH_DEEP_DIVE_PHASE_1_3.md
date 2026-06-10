# Technical Deep Dive: Phase 1-3 (Foundations & The Memory Wall)
**Perspective: Senior NVIDIA CUDA Engineer**

## 1. 核心挑戰：指數級增長的內存牆 (The Exponential Memory Wall)
在量子模擬的早期開發中，我們面臨的是 **Hadamard 變換的物理極限**。傳統 $N$-qubit 的算子是一個 $2^N \times 2^N$ 的巨型矩陣。
*   **工程痛點：** 當 $N=16$ 時，矩陣大小為 $65,536^2 \times 8$ bytes $\approx$ 34.3 GB。這遠超消費級顯卡（RTX 4090/4060）的 VRAM 限制，導致 **OOM (Out of Memory)**。
*   **初版解決方案：** 嘗試實體配置矩陣並使用 cuBLAS。雖然簡單，但在 $N=14$ 以上就徹底崩潰。

## 2. 技術方案 (Tech Solution): SIMT 基準與初步融合
### 2.1 算子融合 (Operator Fusion)
*   **實作：** `iqp_phase_fwt_normalize_tc_kernel`。
*   **技術細節：** 針對 $N \le 10$ 的小規模運算，我們不調用任何矩陣運算，而是直接在 Thread 層級將 Phase 計算、FWHT 的蝴蝶變換 (Butterfly) 以及正規化 (Normalization) 融合成一個單一 Kernel。
*   **貢獻：** 減少了多次 Global Memory 讀寫，將效能從 4200us 降低至 27us（約 150 倍提升）。

### 2.2 共享記憶體蝴蝶變換 (Shared Memory FWT)
*   **技術細節：** 實作了 $O(N \log N)$ 的 Fast Walsh-Hadamard Transform。利用 `__shfl_xor_sync` 指令在 Warp 內進行數據交換，完全避免了 Global Memory 訪問。
*   **工程成果：** 在 $N=12$ 時比 PyTorch 快 60 倍，奠定了 **Hybrid Routing** 的基礎。

## 3. 高級工程師點評 (Senior Engineer's Deep Dive)
"Phase 1-3 的價值在於建立了 **Baseline**。我們發現了量子計算中著名的『負面結果』：**Tensor Core 對於小規模運算其實是負優化**。因為 Tensor Core 的啟動成本（拆解、還原）遠高於簡單的 Warp Shuffle。這讓我們意識到必須走 **Hybrid (混合架構)** 路線：小 $N$ 走 SIMT，大 $N$ 走 Tensor Core。"
