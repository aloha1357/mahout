# Technical Deep Dive: Phase 4-5 (The Hybrid Breakthrough)
**Perspective: Senior NVIDIA CUDA Engineer**

## 1. 核心挑戰：如何在 8GB 顯存裡算 8TB 的東西？
雖然 Phase 3 解決了小 $N$ 問題，但大 $N$（$N \ge 16$）依然受限於矩陣存儲。
*   **工程挑戰：** $N=20$ 時矩陣高達 8 TB。
*   **解決思維：** **Matrix-Free (免矩陣)**。我們不需要存儲矩陣，只需要在計算的「一瞬間」知道矩陣的值。

## 2. 技術方案 (Tech Solution): Blocked TC-FWT & Kronecker Decomposition
### 2.1 Kronecker 分解 ($O(2^{2N}) \to O(N 2^N)$)
*   **實作：** 利用數學特性 $H_n = H_{n/2} \otimes H_{n/2}$。
*   **技術細節：** 將一個 $2^N$ 的運算拆解為兩個連續的 $2^{N/2}$ Blocked GEMM。
*   **工程意義：** 運算量降低了 **128 倍** 以上。這是我們能跑贏 $O(4^N)$ 傳統 GEMM 的數學核心。

### 2.2 即時矩陣生成 (On-the-fly Generation)
*   **實作：** `ImplicitHadamardOzakiEngine`。
*   **技術細節：**
    *   Hadamard 矩陣的值僅為 $+1$ 或 $-1$。
    *   利用 `__popcll(row & col) & 1` 結合位元運算，在 Tensor Core 載入 Fragment 的過程中「即時」生成這兩個值。
    *   這徹底消滅了 Hadamard 矩陣的 Global Memory 配置。

## 3. 混合路由系統 (Hybrid Routing Strategy)
*   **實作邏輯：**
    *   **$N \le 12$：** 路由至 SIMT Warp-Shuffle (極速緩存)。
    *   **$N > 12$：** 路由至 Blocked TC-FWT (Tensor Core 暴力重組)。
*   **貢獻：** 這解決了 $N=14, 15$ 在切換架構時的性能震盪問題。

## 4. 高級工程師點評 (Senior Engineer's Deep Dive)
"這是一個非常高級的 **Compute-for-Storage** 權衡。我們在 RTX 4090 上利用強大的 ALU 算力來換取 VRAM 空間。雖然我們額外計算了 `__popcll`，但因為省去了 34GB 的內存讀取，整體的 Roofline 從 Memory-Bound 成功轉移到了 Compute-Bound，這在超級電腦模擬中是唯一的生存之道。"
