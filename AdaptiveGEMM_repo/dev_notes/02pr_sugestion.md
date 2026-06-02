對，而且你剛好已經走到一個很多研究型專案都會遇到的節點：

```text
Final Branch
已經完成
CI 已過
Correctness 已驗證
Performance 已驗證
```

但：

```text
還沒有可 Review 的歷史
```

這兩件事情不一樣。

---

# 你現在的工作其實不是寫 Code

而是：

```text
Git Archaeology
```

把歷史重新挖出來。

因為 Reviewer 不會看：

```text
最終版
```

Reviewer 會看：

```text
你怎麼走到最終版
```

---

# 我會讓 AI 做的事情

不是：

```text
幫我寫 PR
```

而是：

```text
幫我重建開發歷史
```

---

# Step 1

先做 Evolution Report

建立：

```text
reports/

00_baseline.md
01_operator_fusion.md
02_memory_model.md
03_shared_memory.md
04_implicit_fwt.md
05_persistent_kernel.md
06_tensorcore.md
07_ozaki.md
08_final.md
```

---

然後丟給 AI 的 Prompt：

```markdown
你是一位 NVIDIA CUDA HPC Reviewer。

以下是：

1. baseline benchmark
2. 本次 commit 的改動
3. benchmark 結果
4. NCU 結果

請幫我撰寫：

- Goal
- What Changed
- Why
- NCU Analysis
- Bottleneck
- Risks
- Next Step

格式請符合 research report。
```

---

這樣每個版本都會有：

```text
發生什麼事
為什麼快
快在哪裡
代價是什麼
```

---

# Step 2

建立 PR Mapping

建立：

```text
pr_plan/

pr1.md
pr2.md
pr3.md
pr4.md
```

---

例如

## pr1.md

```markdown
PR1

Infrastructure + Correctness

包含：

- test framework
- benchmark framework
- validation

來源：

reports/

00
01

風險：

低

依賴：

無
```

---

然後問 AI：

```markdown
根據這份 PR Scope

請扮演 Apache Maintainer

列出：

1. reviewer 最可能問的問題
2. 必須補的測試
3. CI 風險
4. merge blockers
```

---

這超有用。

因為很多時候：

```text
工程師視角
≠
maintainer視角
```

---

# Step 3

建立 Research Repo

這跟 Mahout 分開。

例如：

```text
fwt-tc-research/

docs/
reports/
benchmarks/
ncu/
kernels/
experiments/
```

---

然後把：

```text
TensorCore
Ozaki
INT8
Quantum
```

全部放這裡。

---

因為：

這些東西未必能 merge。

但有研究價值。

---

# Step 4

讓 AI 幫你切 Branch

這個其實是你下一步。

你現在有：

```text
final branch
```

但沒有：

```text
PR1 branch
PR2 branch
PR3 branch
PR4 branch
```

---

所以我會先建立：

```bash
git checkout final

git branch pr1-correctness
git branch pr2-implicit-fwt
git branch pr3-memory
git branch pr4-runtime
```

---

然後問 AI：

```markdown
這是 final branch 的 commit history

目標：

PR1 = correctness

PR2 = implicit fwt

PR3 = memory optimization

PR4 = runtime dispatch

請幫我規劃：

1. 哪些 commit 應該進哪個 PR
2. 哪些 commit 需要拆開
3. 哪些 commit 必須 squash
4. 哪些 commit 不適合 merge
```

---

這其實比寫 Code 更重要。

---

# 我現在最推薦的流程

## 本週

```text
✓ Action 綠燈
✓ Correctness

建立：

reports/
```

完成：

```text
00
01
02
03
04
```

---

## 下週

建立：

```text
research repo
```

整理：

```text
benchmarks
ncu
results
```

---

## 再下一步

開：

```text
RFC Issue
```

內容：

```text
Motivation

Current Results

Known Limitations

PR Breakdown

Questions
```

---

## 最後

開始真的切：

```text
PR1
PR2
PR3
PR4
```

---

其實你現在最大的資產已經不是：

```text
75us
```

而是：

```text
你有完整的最終版
+
Action 通過
+
Correctness 通過
```

這代表你已經站在「整理歷史」的階段了。

接下來 AI 最有價值的用途不是幫你寫 Kernel，而是幫你把：

```text
混在一起的最終成果
```

重建成：

```text
可以被 reviewer 理解的演化過程
```

這件事做好之後，PR 成功率通常會比再追 10% 效能高得多。
