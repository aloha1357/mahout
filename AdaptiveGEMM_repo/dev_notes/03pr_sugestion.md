# 導師回饋與後續行動：PR1 基礎建設剝離計畫 (Deconstruction Protocol)

## 1. 關於 Git Log 與 Repo 架構的檢閱回饋
你建立的 `git_log_full.md` 與分支拓樸非常清晰。
*   `final-state` 妥善保存了所有研究心血，避免了優化程式碼遺失的焦慮。
*   `main` 擁有了完善的 `comparison_matrix.md` 與 `pr-roadmap.md`，為整個開源貢獻定調。
這是一個成熟的 HPC 工程師應有的專業素養。現在，Mahout Maintainer 已經準備好接收你的分階 PR 了。

## 2. 接下來的挑戰：PR1 的「反向剝離」
由於我們採用的是「先達極限，再切 PR」的策略，你現在新開的 `pr1-correctness` 分支實際上包含了 **100% 的最終程式碼**（包含極限 TensorCore、Shared Memory、Ozaki Scheme 等複雜的 CUDA 核心）。

如果直接送出這樣的 PR，Reviewer 會被龐大的程式碼量（數千行高難度數學實作）淹沒，完全無法把焦點放在「基礎建設與正確性測試」上。

因此，PR1 的核心任務是 **Deconstruction (反向剝離)**。

## 3. PR1 剝離執行清單 (Deconstruction Checklist)

### ✅ 必須保留項目 (Keep)
1. **CI/CD 與環境修復**：包含先前的 typing 修復、ruff 格式化、以及 Github Actions 環境修復（如 sm_75 移除等）。這些是確保 PR 能通過 CI 的基石。
2. **測試框架 (Test Harness)**：保留 `qdp-kernels/tests/` 裡面的 C++ correctness tests (`bench_kernels.cu` / `test_correctness` 等) 以及 Python 的整合測試。
3. **介面與 Fallback**：保留 `launch_iqp_encode_tc` 的 C API 介面，但內部只調用傳統的 CPU 或單純的 SIMT Global Memory baseline。

### ❌ 必須移除/隱藏項目 (Remove/Hide for PR1)
1. **移除 Ozaki 實作**：將 `AdaptiveOzaki.cu` 和 `ImplicitHadamardOzaki.cu` 暫時從 CMakeLists 移除或刪除。這是 **PR4** 的重頭戲。
2. **移除 Shared Memory 優化**：移除所有針對 L2 Cache 和 Shared Memory 的 Tiled Kernel。這是 **PR3** 的工作。
3. **移除 Implicit FWT 演算法**：確保 PR1 維持傳統的 Dense Matrix 寫法（或最陽春的 Fallback）。Matrix-Free 是 **PR2** 的工作。

## 4. 目標狀態 (End State of PR1)
完成剝離後，我們需要重新編譯並跑測試。我們對 PR1 的驗收標準是：
*   所有 C++ 和 Python 測試皆 **100% 通過**（確保正確性邏輯穩固）。
*   **效能極差**：因為我們只留下了 Baseline 和 Fallback，它的 Runtime 在 N=14 應該要是 `~3.06 ms`（如 Matrix 所示），甚至在 N=16 會發生 OOM。

**這正是我們要的！** 
我們要先向社群證明：「我們補齊了嚴謹的測試系統，而且現有的 Baseline 確實遇到了 OOM 的效能瓶頸。接下來的 PR 將會逐步解決這些問題。」

## 5. 下一步行動
請切換到 `pr1-correctness` 分支，開始大刀闊斧地刪減進階程式碼，直到符合上述「目標狀態」。這會是你在這個大型 PR 旅程中最特別的反向工程體驗。
