# Colab NCU Bundle — PR007 IQP TC-FWT

自包含套件：在 **Google Colab**（或任意 Linux + NVIDIA GPU + CUDA）上編譯、跑 benchmark、對 **所有 PR007 kernel** 做 NCU，並產生 **summary**。

## 資料夾內容

```
colab_ncu_bundle/
├── README.md                 ← 本說明
├── run_all.sh                ← 一鍵：編譯 + bench + 全部 NCU + summary
├── colab_pr007_ncu.ipynb     ← Colab 筆記本（可選）
├── src/                      ← kernel 原始碼 + headers
├── tests/bench_kernels.cu    ← benchmark 驅動
├── scripts/
│   ├── build.sh
│   ├── run_benchmark.sh
│   ├── run_all_ncu.sh        ← 5 種 NCU 情境
│   ├── generate_ncu_summary.py
│   └── detect_gpu_arch.sh
└── reports/                  ← 執行後產生（CSV / summary）
```

## 涵蓋的「所有情況」（NCU）

| # | 檔案 | Kernel | 環境 | 意義 |
|---|------|--------|------|------|
| 1 | `ncu_fwt_baseline.csv` | `fwt_butterfly_batch_kernel` | 預設 | Baseline 全域 FWT |
| 2 | `ncu_phase_split.csv` | `iqp_phase_split_kernel` | 預設 | TC：相位拆分 |
| 3 | `ncu_modulo_precompute.csv` | `precompute_modulo_kernel_p26_implicit` | 預設 | TC：Ozaki 模數預計算 |
| 4 | `ncu_ozaki_grid.csv` | `implicit_hadamard_ozaki_grid_kernel_implicit` | `OZAKI_NCU_PROFILE=1` | 可監測 grid 路徑 |
| 5 | `ncu_ozaki_persistent.csv` | `implicit_hadamard_ozaki_persistent_kernel_implicit` | 預設（無 env） | 正式 persistent 路徑 |

Workload 固定：**N=14, batch=1**（與 PR007 NCU 一致）。

Summary 輸出：

- `reports/ncu_summary.txt` — 純文字，每個 case 的 status + metrics
- `reports/ncu_summary.md` — 總表（OK / LaunchFailed 一眼看懂）
- `reports/benchmark_results.txt` — N=14/16 batch=128 時間 + 正確性
- `reports/ncu_run_manifest.txt` — GPU / driver / nvcc / ncu 版本

---

## 方法一：Google Colab（推薦）

### 1) 上傳 / 解壓

1. 下載或解壓 **`apache_mout_colab.zip`**（結構為 `apache_mout/colab_ncu_bundle/...`）。
2. 在 Colab 選 **Runtime → Change runtime type → GPU**（T4 / L4 / A100 皆可）。

解壓後目錄應為：

```
/content/apache_mout/colab_ncu_bundle/   ← Colab 預設上傳到 /content
├── run_all.sh
├── README.md
└── ...
```

### 2) 安裝 / 確認 NCU

在 Colab 第一個 cell：

```python
# 檢查 GPU
!nvidia-smi

# 多數 Colab 已有 CUDA；若 ncu 不存在可嘗試：
import os, shutil
ncu = shutil.which("ncu") or "/usr/local/cuda/bin/ncu"
print("ncu:", ncu)
if not os.path.exists(ncu):
    print("Install Nsight Compute or use Colab with full CUDA toolkit.")
```

若 `ncu` 不存在，可嘗試（視 Colab 映像而定）：

```bash
apt-get update -qq && apt-get install -y -qq cuda-nsight-compute-12-6 2>/dev/null || true
```

### 3) 一鍵執行（與 Colab 筆記本相同）

```python
%cd apache_mout/colab_ncu_bundle
!chmod +x run_all.sh scripts/*.sh
!./run_all.sh
```

若解壓在別處，先 `%cd` 到含 `run_all.sh` 的 `colab_ncu_bundle` 目錄即可。

### 4) 看結果

```python
from IPython.display import Markdown, display
display(Markdown(open("reports/ncu_summary.md").read()))
print(open("reports/ncu_summary.txt").read())
```

下載整包結果：

```python
!zip -r pr007_colab_results.zip reports/
from google.colab import files
files.download("pr007_colab_results.zip")
```

### 5) 使用內附筆記本

上傳後開啟 **`colab_pr007_ncu.ipynb`**，依序 Run All。

---

## 方法二：本機 WSL / Linux

```bash
cd colab_ncu_bundle
chmod +x run_all.sh scripts/*.sh
./run_all.sh
cat reports/ncu_summary.txt
```

---

## 方法三：分步執行

```bash
./scripts/build.sh           # 自動 -arch=sm_XX
./scripts/run_benchmark.sh   # N=14/16 效能 + 正確性
./scripts/run_all_ncu.sh     # 5 個 NCU CSV
# summary 已由 run_all_ncu 自動呼叫；可單獨：
python3 scripts/generate_ncu_summary.py
```

---

## 如何解讀 summary

| NCU status | 意思 |
|------------|------|
| **OK** | 有有效 metrics，可寫進 PR |
| **LaunchFailed** | NCU 無法 profile（Win/WSL 上 Ozaki 常見；Colab 上若仍失敗見下方） |
| **MISSING** | 未跑或 CSV 不存在 |
| **No metrics** | 有跑但 CSV 無數字 |

**Bottleneck 速讀（與 handover 一致）：**

- FWT baseline：DRAM 高、SM 低 → memory-leaning  
- phase_split：SM 高、DRAM 極低 → compute-bound  
- Ozaki MMA：看 **Tensor pipe active %**（若 NCU OK）

---

## Colab vs Windows / WSL / Docker

| 環境 | Side-kernel NCU | Ozaki MMA NCU |
|------|-----------------|---------------|
| Windows WDDM | 通常 OK | 常 LaunchFailed |
| WSL2 | 通常 OK | 常 LaunchFailed |
| Docker Desktop (GPU) | 類似 WSL | 常 LaunchFailed |
| **Colab / 裸機 Linux** | OK | **較有機會 OK** |

調 Windows TDR 60 秒 **不能** 修 LaunchFailed（那是 launch/replay 失敗，不是跑太久）。

---

## 架構注意（Colab GPU）

`scripts/detect_gpu_arch.sh` 會依 `nvidia-smi` 設 `-arch=sm_XX`：

- T4 → `sm_75`
- A100 → `sm_80`
- L4 / RTX 40xx → `sm_89`

若編譯失敗，手動改 `scripts/build.sh` 裡的 `-arch`。

---

## 與主 repo 的關係

此 bundle 自 `qdp/qdp-kernels` 複製而來，用於 **外部評測**；主線更新 kernel 後請重新複製 `src/` 與 `tests/bench_kernels.cu`。

相關文件：

- `AdaptiveGEMM_repo/reports/PR007_ncu_benchmark.md`
- `AdaptiveGEMM_repo/dev_notes/phase7_pr007_handover.md`