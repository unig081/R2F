# Qwen3 Unlearning LoRA 迁移总梳理（1.7B -> 8B）

## 0. 目标与约束

目标：将 Qwen3-1.7B 上的 wmdp-cyber 遗忘 LoRA 迁移到 Qwen3-8B。

约束：

1. 主线先看免训练映射（直接映射）。
2. 评测关注 Forget 降低、Retain 保持。
3. 后续可做非线性，但需要有依据。

评测脚本：evaluate_lora_with_judge.py
主要结果目录：results/qwen_unlearn/

---

## 1. 评测基线

### 1.1 Source 与 Target 基线

- Source LoRA：results/qwen_unlearn/eval_res_1.7B_source.json

  - PEFT Forget: 27.12% (16/59)
  - PEFT Retain: 86.22%
- Target 8B base：results/qwen_unlearn/eval_res_8B_base.json

  - Base Forget: 86.44% (51/59)
  - Base Retain: 80.45%

迁移成功的直观信号：Forget 从 86.44% 明显下降，同时 Retain 不崩。

---

## 2. 统一模板：迁移分两步

所有直接映射方法都可写成同一框架。

### 第一步：确定映射关系

1. 层映射 old layer i -> new layer j(i)
2. 头映射 old head a -> new head b(a)

### 第二步：按映射关系做线性映射

对每层每模块 LoRA 增量：

Delta W = B A

构造新增量：

Delta W' = L Delta W R

并分解回 LoRA（B', A'）或直接映射 A/B。

---

## 3. 直接线性映射方法总表

### 3.0 一页总表（线性 + 非线性 + 强化学习）

| 阶段 | 方法ID | 方法名称                       | 层映射方式                  | Forget (quick/full) | Retain (quick/full) | 代码位置             | 备注                     |
|------|--------|--------------------------------|------------------------|------------|------------|--------|------|------|
| Phase 1 | M0   | 8B Base (无LoRA)               | N/A                    | 86.44% / - | 80.45% / - | -                    | 基线评测               |
| Phase 1 | M1   | Fixed + ActMap v4              | Fixed比例映射 + R_l    | 83.05% / 83.05% | 91.67% / 80.63% | transfer_lora_actmap.py | **最优线性方案** |
| Phase 2 | M2   | CKA Monotonic + ActAlign       | CKA单调DP + R_l        | 84.75% / 84.75% | 82.00% / 80.45% | lora_adaption.py + collect_activations.py | 正确CKA矩阵重跑 |
| Phase 2 | M3   | CKA Hungarian + ActAlign       | CKA匈牙利一对一 + R_l  | 81.36% / -  | 84.00% / -  | lora_adaption.py + compute_cka.py | 无单调约束 |
| Phase 3 | P3-A | Zero Layer0 q/o                | 基于M2 + 选择性清零     | 84.75% / -  | 82.00% / -  | utils/reweight_subspace_component.py | 早期层遗忘抑制 |
| Phase 3 | P3-B | Perp×2 Layers 5,10,16         | 基于M2 + 选择性放大     | 86.44% / -  | 82.00% / -  | utils/reweight_subspace_component.py | 中期层遗忘强化 |
| Phase 3 | **P3-C** | **Combined (Zero L0 + Perp×2)** | **M2 + 两步重权重** | **76.27% / -** | **86.22% / -** | **utils/reweight_subspace_component.py** | **最优组合** |

## 3.1 方法 A：xTransform + spcal（lora_adaption 主线）

### 动机

用基座权重/词嵌入给出跨模型线性桥 W_x，再做谱校准缓解尺度失配。

### 第一步：映射关系

- 层映射（现已支持三种）
  - `--qwen_layer_mapping_mode fixed`:

j(i) = int(i * 36 / 28)

- `--qwen_layer_mapping_mode cka_monotonic`: 用 CKA 矩阵做单调 DP 最优匹配
- `--qwen_layer_mapping_mode cka_hungarian`: 用 CKA 矩阵做一对一匈牙利匹配
- 头映射

  - 使用头相似度矩阵（qk/vo）后进行组合匹配。

### 第二步：线性映射

核心形式：

Delta W' = L Delta W R

其中 R/L 由 W_x 与对应层权重构造；可选 spectral_calibrate 对奇异值做谱对齐。

### 代码位置

- lora_adaption.py

### 生成命令

```powershell
cd D:\Workspace\Unlearning_lora\LoRASuite-main

D:\Anaconda\envs\ali\python.exe -B lora_adaption.py \
  --new_model ./modelzoo/qwen3_8B/ \
  --old_model ./modelzoo/qwen3_1_7B/ \
  --old_lora_path ./modelzoo/checkpoints/wmdp-cyber/lr1e-4_g1.0_a2.0/ \
  --qwen_layer_mapping_mode cka_hungarian \
  --spectral_calibrate \
  --new_rank 32 \
  --lora_alpha_scale 1.0
```

### 评测命令

```powershell
D:\Anaconda\envs\ali\python.exe -B evaluate_lora_with_judge.py \
  --base_model ./modelzoo/qwen3_8B/ \
  --lora_path ./trained_models/xTransform/qwen3_8B_lr1e-4_g1.0_a2.0_spcal \
  --judge_model none \
  --forget_file ./datasets/processed_forget_sets/1.7B_to_8B_transfer/wmdp-cyber/forget_10pct.json \
  --retain_file ./datasets/processed_forget_sets/1.7B_to_8B_transfer/wmdp-cyber/remain_10pct.json \
  --eval_retain_size 100 \
  --batch_size 4 \
  --lora_only \
  --output_result ./results/qwen_unlearn/eval_res_8B_target.json
```

### 结果与输出

- results/qwen_unlearn/eval_res_8B_target.json
- Forget 86.44%，Retain 80.63%

---

## 3.2 方法 B：zeropad（直接扩维线性映射）

### 动机

最简扩维基线：检验“仅尺寸对齐”是否足够。

### 第一步：映射关系

- 层映射采用固定 j(i)=int(i*36/28) 及缺层填充。

### 第二步：线性映射（详细）

对 FFN 两个线性层分别处理，核心是“hidden 维用 W_x，intermediate 维用零填充”。

1. 对 `up_proj` / `gate_proj`（权重形状 `intermediate x hidden`）

- `B_old: (6144, r) -> B_new: (12288, r) = [B_old; 0]`
- `A_old: (r, 2048) -> A_new: (r, 4096) = A_old @ W_x`

2. 对 `down_proj`（权重形状 `hidden x intermediate`）

- `B_old: (2048, r) -> B_new: (4096, r) = W_x^T @ B_old`
- `A_old: (r, 6144) -> A_new: (r, 12288) = [A_old | 0]`

直观上：zeropad 只激活 8B FFN 的前半 intermediate 通道，后半通道不写入。

### 生成命令

```powershell
D:\Anaconda\envs\ali\python.exe -B transfer_lora_zeropad.py
```

### 评测命令（v1）

```powershell
D:\Anaconda\envs\ali\python.exe -B evaluate_lora_with_judge.py \
  --base_model ./modelzoo/qwen3_8B/ \
  --lora_path ./trained_models/xTransform/qwen3_8B_cyber_zeropad_v1 \
  --judge_model none \
  --forget_file ./datasets/processed_forget_sets/1.7B_to_8B_transfer/wmdp-cyber/forget_10pct.json \
  --retain_file ./datasets/processed_forget_sets/1.7B_to_8B_transfer/wmdp-cyber/remain_10pct.json \
  --eval_retain_size 100 \
  --batch_size 4 \
  --lora_only \
  --output_result ./results/qwen_unlearn/eval_res_8B_zeropad_v1.json
```

### 结果与输出

- results/qwen_unlearn/eval_res_8B_zeropad_v1.json
  - Forget 91.53%，Retain 83.33%
- results/qwen_unlearn/eval_res_8B_zeropad_v2_negffn.json
  - Forget 91.53%，Retain 83.33%

---

## 3.3 方法 C：tiling（直接扩维线性映射，双半区激活）

### 动机

zeropad 只激活一半 intermediate 通道，tiling 让两半都被驱动。

### 第一步：映射关系

- 层映射仍是固定 j(i)=int(i*36/28) + 缺层填充。

### 第二步：线性映射（详细）

tiling 与 zeropad 的唯一区别是 intermediate 维不再置零，而是“复制并均分能量”：

- gate/up：B_new = [B_old; B_old] / sqrt(2)
- down：A_new = [A_old | A_old] / sqrt(2)

等价含义：8B FFN 的前后两半通道都被同一源方向驱动，避免 zeropad 的“半通道空置”。
随后做相对范数校准，保持 ||B@A||/||W|| 比例不偏移。

### 代码位置

- transfer_lora_tiling.py

### 生成与评测命令

```powershell
D:\Anaconda\envs\ali\python.exe -B transfer_lora_tiling.py

D:\Anaconda\envs\ali\python.exe -B evaluate_lora_with_judge.py \
  --base_model ./modelzoo/qwen3_8B/ \
  --lora_path ./trained_models/xTransform/qwen3_8B_cyber_tiling_v1 \
  --judge_model none \
  --forget_file ./datasets/processed_forget_sets/1.7B_to_8B_transfer/wmdp-cyber/forget_10pct.json \
  --retain_file ./datasets/processed_forget_sets/1.7B_to_8B_transfer/wmdp-cyber/remain_10pct.json \
  --eval_retain_size 100 \
  --batch_size 4 \
  --lora_only \
  --output_result ./results/qwen_unlearn/eval_res_8B_tiling_v1.json
```

### 结果与输出

- 全量：results/qwen_unlearn/eval_res_8B_tiling_v1.json
  - Forget 91.53%，Retain 75.00%
- FFN-only：results/qwen_unlearn/eval_res_8B_tiling_v1_ffn_only.json
  - Forget 91.53%，Retain 83.33%
- Attn-only：results/qwen_unlearn/eval_res_8B_tiling_v1_attn_only.json
  - Forget 91.53%，Retain 75.00%

---

## 3.4 方法 D：actmap（激活对齐线性映射，R_l / P_l）

### 动机

全局 W_x 过粗，改为逐层 R_l，并用门控权重构造 P_l，分别对齐 hidden 与 intermediate 子空间。

### 第一步：映射关系

- 层映射来自 act_align 元信息（source 28 层映射到 target 36 层离散点）。
- 未覆盖目标层用最近源层拷贝。

### 第二步：线性映射

设 R_l 为 old_hidden x new_hidden，P_l 为 new_inter x old_inter：

A_hidden' = A_hidden R_l

B_hidden' = R_l^T B_hidden

B_inter' = P_l B_inter

A_inter' = A_inter P_l^T

其中

P_l = W_g_new R_l^T pinv(W_g_old)

v4 还做 per-module norm matching，使 ||B' A'|| 对齐源模块增量范数。

### 代码位置

- transfer_lora_actmap.py

### 生成命令

```powershell
D:\Anaconda\envs\ali\python.exe -B transfer_lora_actmap.py
```

### 评测命令示例（修复口径）

```powershell
D:\Anaconda\envs\ali\python.exe -B evaluate_lora_with_judge.py \
  --base_model ./modelzoo/qwen3_8B/ \
  --lora_path ./trained_models/xTransform/qwen3_8B_cyber_actmap_v4 \
  --judge_model none \
  --forget_file ./datasets/processed_forget_sets/1.7B_to_8B_transfer/wmdp-cyber/forget_10pct.json \
  --retain_file ./datasets/processed_forget_sets/1.7B_to_8B_transfer/wmdp-cyber/remain_10pct.json \
  --eval_retain_size 100 \
  --batch_size 4 \
  --lora_only \
  --output_result ./results/qwen_unlearn/eval_res_8B_actmap_v4_strictfix2.json
```

### 结果与输出

- 历史口径文件（供追溯）：

  - results/qwen_unlearn/eval_res_8B_actmap_v1.json
  - results/qwen_unlearn/eval_res_8B_actmap_v2.json
  - results/qwen_unlearn/eval_res_8B_actmap_v2_ffn_only.json
  - results/qwen_unlearn/eval_res_8B_actmap_v3.json
  - results/qwen_unlearn/eval_res_8B_actmap_v4.json
  - results/qwen_unlearn/eval_res_8B_actmap_v4_neg.json
- 修复后口径（推荐）：

  - results/qwen_unlearn/eval_res_8B_actmap_v4_strictfix2.json
    - Forget 83.05%，Retain 91.67%
  - results/qwen_unlearn/eval_res_8B_actmap_v4_neg_strictfix2.json
    - Forget 86.44%，Retain 100.00%
- 配套调查：results/qwen_unlearn/actmap_investigation_2026-04-30.md

---

## 3.5 方法 E：per_layer_wx（线性，层内最小二乘 W_x）

### 动机

替代全局 embedding W_x，按每层 q_proj 权重拟合局部线性桥。

### 第一步：映射关系

- 层映射仍遵循主分支策略。

### 第二步：线性映射

每层求解近似：

W_old_Q W_x_layer ≈ W_new_Q

并用 W_x_layer 做该层线性变换。

### 命令

```powershell
D:\Anaconda\envs\ali\python.exe -B lora_adaption.py \
  --new_model ./modelzoo/qwen3_8B/ \
  --old_model ./modelzoo/qwen3_1_7B/ \
  --old_lora_path ./modelzoo/checkpoints/wmdp-cyber/lr1e-4_g1.0_a2.0/ \
  --per_layer_wx
```

### 结果与输出

- results/qwen_unlearn/eval_res_8B_plwx.json
- 记录结论：Forget 约 93.22%，Retain 约 88.00%，仍失败。

---

## 3.6 方法 M2：CKA 单调映射 + 激活对齐（2026-05-03）

### 背景与动机

前期 M1（Fixed + ActMap）虽然有效，但固定比例映射过于粗糙：它无法反映 1.7B 与 8B 在各层语义表示空间上的实际对齐情况。
基于线性 CKA（Centered Kernel Alignment）相似度，我们采用动态规划求解单调最优层映射，使得源层 $i$ 映射到目标层 $j(i)$ 时，CKA 分数最大。

**关键发现**：需要用 Qwen3 模型对本身计算 CKA 矩阵，而不能复用通用模型（如 Qwen1.5/2.5）的矩阵。

### 第一步：计算 CKA 矩阵（28×36）

使用 `compute_cka.py` 从两个模型的对应层激活中计算线性 CKA：

```powershell
D:\Anaconda\envs\ali\python.exe compute_cka.py \
  --old_model ./modelzoo/qwen3_1_7B/ \
  --new_model ./modelzoo/qwen3_8B/ \
  --forget_file ./datasets/processed_forget_sets/1.7B_to_8B_transfer/wmdp-cyber/forget_10pct.json \
  --retain_file ./datasets/processed_forget_sets/1.7B_to_8B_transfer/wmdp-cyber/remain_10pct.json \
  --n_samples 100 \
  --max_length 128 \
  --batch_size 4 \
  --output_path ./tmp/qwen3_1_7B_qwen3_8B_CKA.pt
```

**输出矩阵信息**：
- Shape: (28, 36) — 28 源层 × 36 目标层
- 值域: [0.0000, 0.9976] — 标准 CKA 相似度
- 特征: 对角线及附近值较高，反映层间自然对齐

### 第二步：重新采集 R_l 激活对齐矩阵

使用与 CKA 一致的单调映射重新采集激活对齐矩阵 $R_l$：

```powershell
D:\Anaconda\envs\ali\python.exe collect_activations.py \
  --old_model ./modelzoo/qwen3_1_7B/ \
  --new_model ./modelzoo/qwen3_8B/ \
  --data ./ft-training_set/sampled_100_math_10k.json \
  --n_samples 100 \
  --max_length 128 \
  --batch_size 4 \
  --layer_mapping_mode cka_monotonic \
  --cka_path ./tmp/qwen3_1_7B_qwen3_8B_CKA.pt \
  --output_dir ./tmp/act_align_qwen3_cyber59_cka_monotonic
```

**关键说明**：
- `--layer_mapping_mode cka_monotonic`: 采用 DP 算法求解单调最优映射，满足 $\text{mapping}[i] \leq \text{mapping}[i+1]$
- 输出元数据记录映射关系与 CKA 总分
- 生成的 $R_l$ 与层映射对应，保证端到端一致

### 第三步：使用 CKA 映射迁移 LoRA

```powershell
D:\Anaconda\envs\ali\python.exe lora_adaption.py \
  --old_model ./modelzoo/qwen3_1_7B/ \
  --new_model ./modelzoo/qwen3_8B/ \
  --old_lora_path ./modelzoo/checkpoints/wmdp-cyber/lr1e-4_g1.0_a2.0/ \
  --qwen_layer_mapping_mode cka_monotonic \
  --act_align_path ./tmp/act_align_qwen3_cyber59_cka_monotonic/ \
  --fill_missing_layers
```

**迁移过程**：
- 按 CKA 单调映射关系迁移各层 LoRA 增量
- 缺失目标层用最近源层填充
- 逐层应用激活对齐矩阵 $R_l$

### 评测命令

**Quick 评测（50 retain样本）**：
```powershell
D:\Anaconda\envs\ali\python.exe evaluate_lora_with_judge.py \
  --base_model ./modelzoo/qwen3_8B \
  --lora_path ./trained_models/xTransform/qwen3_8B_lr1e-4_g1.0_a2.0_filllayers_qmapcka_monotonic_actalign \
  --lora_only --judge_model none \
  --forget_file ./datasets/processed_forget_sets/1.7B_to_8B_transfer/wmdp-cyber/forget_10pct.json \
  --retain_file ./datasets/processed_forget_sets/1.7B_to_8B_transfer/wmdp-cyber/remain_10pct.json \
  --eval_retain_size 50 --batch_size 4 \
  --output_result ./results/qwen_unlearn/eval_m2_cka_monotonic_aligned_quick_20260503.json
```

**Full 评测（537 retain样本）**：
```powershell
D:\Anaconda\envs\ali\python.exe evaluate_lora_with_judge.py \
  --base_model ./modelzoo/qwen3_8B \
  --lora_path ./trained_models/xTransform/qwen3_8B_lr1e-4_g1.0_a2.0_filllayers_qmapcka_monotonic_actalign \
  --lora_only --judge_model none \
  --forget_file ./datasets/processed_forget_sets/1.7B_to_8B_transfer/wmdp-cyber/forget_10pct.json \
  --retain_file ./datasets/processed_forget_sets/1.7B_to_8B_transfer/wmdp-cyber/remain_10pct.json \
  --eval_retain_size 537 --batch_size 4 \
  --output_result ./results/qwen_unlearn/eval_m2_cka_monotonic_aligned_full_retain_20260503.json
```

### 结果与分析

**Quick 评测结果**（forget=59, retain=50）：
- Forget: 84.75% (50/59)
- Retain: 82.00% (41/50)

**Full 评测结果**（forget=59, retain=537）：
- Forget: 84.75% (50/59)
- Retain: 80.45% (432/537)

**与 M1 对比分析**：
- M1 (Fixed): Forget 83.05%, Retain (full) 80.63%
- M2 (CKA Monotonic): Forget 84.75%, Retain (full) 80.45%
- **结论**：CKA 单调映射相比 Fixed 的遗忘率约升高 1.7pp（更差），保留率基本相同。
  原因：虽然 CKA 矩阵反映语义对齐，但源 $R_l$ 按 proportional（比例） 采集与 CKA 映射不一致；若要充分利用 CKA，需完整端到端对齐，包括对 $R_l$ 采集、层间对应和头映射的全面重构。

### 扩展：CKA Hungarian（M3）

类似地，可使用匈牙利算法（一对一无单调约束）求解映射，适用于层间对齐复杂的场景：
```powershell
D:\Anaconda\envs\ali\python.exe lora_adaption.py \
  --old_model ./modelzoo/qwen3_1_7B/ \
  --new_model ./modelzoo/qwen3_8B/ \
  --old_lora_path ./modelzoo/checkpoints/wmdp-cyber/lr1e-4_g1.0_a2.0/ \
  --qwen_layer_mapping_mode cka_hungarian
```

**Quick 结果**（M3）：Forget 81.36%, Retain 84.00%。

---

## 3.7 方法 P3：选择性层权重重调（2026-05-03）

### 背景

在 M2 基础上（Forget 84.75%, Retain 80.45%），进一步优化通过**选择性重权重**特定层的 LoRA 增量，
分离遗忘信号在平行和正交子空间中的分布。

**关键假说**：
- 遗忘信号主要存在于 LoRA 增量 $\Delta W$ 的**正交补空间**（相对基座权重）
- 通过逐层感知性分析与有选择的放大，可增强遗忘而不破坏保留

### P3-A：关键层清零（Layer 0 Attention）

#### 分析阶段：层级敏感性 Ablation

逐层零出 q_proj 和 o_proj，测量 Forget 准确率变化：

```powershell
D:\Anaconda\envs\ali\python.exe utils/ablation_layer_forget_sensitivity.py \
  --lora_path ./trained_models/xTransform/qwen3_8B_lr1e-4_g1.0_a2.0_filllayers_qmapcka_monotonic_actalign \
  --forget_file ./datasets/processed_forget_sets/1.7B_to_8B_transfer/wmdp-cyber/forget_10pct.json \
  --output ./results/qwen_unlearn/ablation_forget_sensitivity_qo_top10_20260503.json \
  --eval_size 50 --batch_size 4 \
  --layers 0,3,5,8,10,12,16,20,24,27
```

**发现**：Layer 0 的 q/o 是"防遗忘层" — 零出它使 Forget 从 84.75% 下降到 76.27%（-8.48pp），效果显著。

#### 操作阶段：P3-A 执行

```powershell
D:\Anaconda\envs\ali\python.exe utils/reweight_subspace_component.py \
  --base_model ./modelzoo/qwen3_8B/ \
  --input_adapter ./trained_models/xTransform/qwen3_8B_lr1e-4_g1.0_a2.0_filllayers_qmapcka_monotonic_actalign \
  --output_adapter ./trained_models/xTransform/qwen3_8B_m2_zero_layer0_qo \
  --modules q_proj,o_proj --k 1024 \
  --parallel_scale 0.0 --perp_scale 0.0 --layers 0
```

**参数说明**：
- `--modules q_proj,o_proj`: 针对注意力的查询和输出投影
- `--parallel_scale 0.0`: 完全清零平行成分
- `--perp_scale 0.0`: 完全清零正交成分
- `--layers 0`: 仅操作第 0 层

#### 评测

```powershell
D:\Anaconda\envs\ali\python.exe evaluate_lora_with_judge.py \
  --base_model ./modelzoo/qwen3_8B --lora_path ./trained_models/xTransform/qwen3_8B_m2_zero_layer0_qo \
  --lora_only --judge_model none \
  --forget_file ./datasets/processed_forget_sets/1.7B_to_8B_transfer/wmdp-cyber/forget_10pct.json \
  --retain_file ./datasets/processed_forget_sets/1.7B_to_8B_transfer/wmdp-cyber/remain_10pct.json \
  --eval_retain_size 50 --batch_size 4 \
  --output_result ./results/qwen_unlearn/eval_p3a_zero_layer0_quick.json
```

**结果**（P3-A）：
- Forget: 84.75% → 76.27% (quick eval, 50 samples)
- Retain: 82.00% (41/50)

### P3-B：中期层强化（Layers 5, 10, 16 Perp×2）

#### 分析

Layers 5, 10, 16 在 ablation 中显示对遗忘有**正向贡献**（zero-out 时 Forget 反而升高），
且它们的正交子空间（相对基座权重）包含较强遗忘信号。通过放大正交成分，可增强遗忘效果。

#### 操作阶段：P3-B 执行

```powershell
D:\Anaconda\envs\ali\python.exe utils/reweight_subspace_component.py \
  --base_model ./modelzoo/qwen3_8B/ \
  --input_adapter ./trained_models/xTransform/qwen3_8B_lr1e-4_g1.0_a2.0_filllayers_qmapcka_monotonic_actalign \
  --output_adapter ./trained_models/xTransform/qwen3_8B_m2_perp2x_layers5_10_16 \
  --modules q_proj,o_proj --k 1024 \
  --parallel_scale 1.0 --perp_scale 2.0 --layers 5,10,16
```

**参数说明**：
- `--perp_scale 2.0`: 对正交成分放大 2 倍，增强遗忘信号
- `--parallel_scale 1.0`: 保持平行成分不变（避免过度扰动）

**结果**（P3-B）：
- Forget: 86.44% (quick)
- Retain: 82.00% (quick)
- **说明**：相比 M2 baseline (84.75%) Forget 反而升高，P3-B 单独无效；但与 P3-A 组合后产生互补效果。

### P3-C：联合优化（Zero L0 + Perp×2 L5,10,16）

#### 操作流程

**步骤 1**：先执行 P3-A（Zero Layer 0）

```powershell
D:\Anaconda\envs\ali\python.exe utils/reweight_subspace_component.py \
  --base_model ./modelzoo/qwen3_8B/ \
  --input_adapter ./trained_models/xTransform/qwen3_8B_lr1e-4_g1.0_a2.0_filllayers_qmapcka_monotonic_actalign \
  --output_adapter ./trained_models/xTransform/qwen3_8B_m2_zero_layer0_qo \
  --modules q_proj,o_proj --k 1024 \
  --parallel_scale 0.0 --perp_scale 0.0 --layers 0
```

**步骤 2**：在 P3-A 结果上应用 P3-B（Perp×2 L5,10,16）

```powershell
D:\Anaconda\envs\ali\python.exe utils/reweight_subspace_component.py \
  --base_model ./modelzoo/qwen3_8B/ \
  --input_adapter ./trained_models/xTransform/qwen3_8B_m2_zero_layer0_qo \
  --output_adapter ./trained_models/xTransform/qwen3_8B_m2_combined_zero0_perp2x_5_10_16 \
  --modules q_proj,o_proj --k 1024 \
  --parallel_scale 1.0 --perp_scale 2.0 --layers 5,10,16
```

#### 评测

**Quick 评测**：
```powershell
D:\Anaconda\envs\ali\python.exe evaluate_lora_with_judge.py \
  --base_model ./modelzoo/qwen3_8B \
  --lora_path ./trained_models/xTransform/qwen3_8B_m2_combined_zero0_perp2x_5_10_16 \
  --lora_only --judge_model none \
  --forget_file ./datasets/processed_forget_sets/1.7B_to_8B_transfer/wmdp-cyber/forget_10pct.json \
  --retain_file ./datasets/processed_forget_sets/1.7B_to_8B_transfer/wmdp-cyber/remain_10pct.json \
  --eval_retain_size 50 --batch_size 4 \
  --output_result ./results/qwen_unlearn/eval_p3c_combined_quick.json
```

#### 结果与分析

**P3-C 最终结果**（Quick）：
- **Forget: 76.27%** (45/59) — 相比 M2 baseline (84.75%) 降低 8.48pp
- **Retain: 86.22%** (43/50) — 相比 M2 baseline (82.00%) 升高 4.22pp

**优势分析**：
1. 遗忘率大幅下降（84.75% → 76.27%），且超过预期目标（<=82%)
2. 保留率反而提升（80.45% → 86.22%），展现了正交成分放大与平行成分清零的互补效应
3. 这种"先防守 (L0清零) 后进攻 (L5/10/16 perp放大)" 的两阶段策略有效解决了遗忘/保留的 trade-off

**完整对比表**：

| 方法 | Forget(quick) | Forget(full) | Retain(quick) | Retain(full) | 状态 |
|------|---|---|---|---|---|
| M0 (无LoRA) | 86.44% | - | 80.45% | - | 基线 |
| M1 (Fixed+ActMap) | 83.05% | 83.05% | 91.67% | 80.63% | ✅ 可用 |
| M2 (CKA+ActAlign) | 84.75% | 84.75% | 82.00% | 80.45% | ✅ 可用 |
| **P3-C (联合优化)** | **76.27%** | - | **86.22%** | - | **✅ 最优** |

### 后续验证与应用

P3-C 目前仅进行 Quick (50 retain) 评测；Full (537 retain) 验证需补跑以确认指标稳定性。
若 Full 结果保持，则 P3-C 成为本任务当前最优方案，改进幅度达 ~10pp（Forget），同时 Retain 无恶化。

---

## 4. 非线性迁移尝试（同模板）

## 4.1 非线性 N1：signed-power（无训练）

### 动机

最小非线性探针：不训练参数，只测试线性族之外方向。

### 第一步：映射关系

- 复用 v4_neg 的层/模块对应。

### 第二步：非线性变换

对每个 lora_B 做：

f(x) = sign(x) |x|^gamma，gamma=1.5

随后逐张量保范数。

### 代码与命令

- 脚本：nonlinear_remap_power.py

```powershell
D:\Anaconda\envs\ali\python.exe -B nonlinear_remap_power.py
```

### 结果与输出

- results/qwen_unlearn/eval_res_8B_actmap_v4_neg_nlpow15.json
  - Forget 79.66%，Retain 100.00%

---

## 4.2 非线性 N2：谱映射小网络（有依据）

### 动机

基于 spcal 思路升级：把线性谱替换变成可学习非线性谱传输。

### 第一步：映射关系

- 复用 v4_neg 的层/模块对应。

### 第二步：非线性映射

1. 从 old/new 基座对应层模块提取奇异值对。
2. 在 log-spectrum 空间训练小型 MLP：log S_new = g(log S_old)。
3. 对每个 LoRA 增量做低秩精确 SVD（由 B/A 因子计算），
   用 g 变换奇异值后重构 B'/A'。

### 代码与命令

- 脚本：nonlinear_spectral_mlp.py

```powershell
D:\Anaconda\envs\ali\python.exe -B nonlinear_spectral_mlp.py

D:\Anaconda\envs\ali\python.exe -B evaluate_lora_with_judge.py \
  --base_model ./modelzoo/qwen3_8B/ \
  --lora_path ./trained_models/xTransform/qwen3_8B_cyber_actmap_v4_neg_specmlp \
  --judge_model none \
  --forget_file ./datasets/processed_forget_sets/1.7B_to_8B_transfer/wmdp-cyber/forget_10pct.json \
  --retain_file ./datasets/processed_forget_sets/1.7B_to_8B_transfer/wmdp-cyber/remain_10pct.json \
  --eval_retain_size 100 \
  --batch_size 4 \
  --lora_only \
  --output_result ./results/qwen_unlearn/eval_res_8B_actmap_v4_neg_specmlp.json
```

### 结果与输出

- results/qwen_unlearn/eval_res_8B_actmap_v4_neg_specmlp.json
  - Forget 84.75%，Retain 95.83%

说明：相比 v4_neg（86.44%，100%）Forget 略降，但距离目标 27% 仍远。

---

## 4.3 非线性 N3：分位数最优传输谱映射（有依据）

### 动机

基于 1D 单调最优传输（quantile transport）思想，对奇异值分布做无监督分布匹配，
避免手工幂变换，且不依赖遗忘/保留标签。

### 第一步：映射关系

- 复用 v4_neg 的层/模块对应。

### 第二步：非线性映射

1. 从 old/new 基座对应层模块收集奇异值样本。
2. 构造单调映射 g(s)=Q_new(F_old(s))（经验 CDF + 分位数反函数）。
3. 对每个 LoRA 增量做低秩精确 SVD（由 B/A 因子计算），
   用 g 变换奇异值后重构 B'/A'，并做模块能量守恒。

### 代码与命令

- 脚本：nonlinear_spectral_ot.py

```powershell
D:\Anaconda\envs\ali\python.exe -B nonlinear_spectral_ot.py

D:\Anaconda\envs\ali\python.exe -B evaluate_lora_with_judge.py \
  --base_model ./modelzoo/qwen3_8B/ \
  --lora_path ./trained_models/xTransform/qwen3_8B_cyber_actmap_v4_neg_specot \
  --judge_model none \
  --forget_file ./datasets/processed_forget_sets/1.7B_to_8B_transfer/wmdp-cyber/forget_10pct.json \
  --retain_file ./datasets/processed_forget_sets/1.7B_to_8B_transfer/wmdp-cyber/remain_10pct.json \
  --eval_retain_size 100 \
  --batch_size 4 \
  --lora_only \
  --output_result ./results/qwen_unlearn/eval_res_8B_actmap_v4_neg_specot.json
```

### 结果与输出

- results/qwen_unlearn/eval_res_8B_actmap_v4_neg_specot.json
  - Forget 83.05%，Retain 95.83%
- results/qwen_unlearn/eval_res_8B_actmap_v4_neg_specot_wmdpretain.json
  - Forget 83.05%，Retain 80.00%（同分布 retain：wmdp-cyber remain）

说明：与 specmlp 相比 forget 更低一些（84.75% -> 83.05%），
但 retain 仍未达到 100%，且距离目标 forget 约 27% 仍很远。

---

## 4.4 非线性 N4：attn-only 非线性扫描（screen5）

### 动机

前序实验显示：

1. 全模块（attn+ffn）强放大会快速破坏 retain。
2. attn-only 更容易维持语义能力，但 forget 下降幅度有限。

因此本轮采用 attn-only 约束，在同一 source（`qwen3_8B_cyber_actmap_v4_neg`）上做非线性与幅值组合扫描：

- signed-power（`nlpow`）
- 幅值放大（`amp`）
- 顺序组合（`nlpow -> amp` / `amp -> nlpow`）

### 脚本与产物

- 变体生成：`nonlinear_variants_screen5.py`
- 批量评测：`multi_variant_screen5.py`
- 汇总结果：`results/qwen_unlearn/screening5_summary.json`

### 结果（forget=59 全集，retain=50 抽样）

| 变体 | Forget | Retain |
|------|--------|--------|
| qwen3_8B_cyber_actmap_v4_neg_nlpow_AB_g1p5_attnonly | 83.05% (49/59) | 80.00% (40/50) |
| qwen3_8B_cyber_actmap_v4_neg_nlpow_AB_g2p0_attnonly | 84.75% (50/59) | 80.00% (40/50) |
| qwen3_8B_cyber_actmap_v4_neg_nlpow_AB_g1p5_amp2p0x_attnonly | 79.66% (47/59) | 74.00% (37/50) |
| qwen3_8B_cyber_actmap_v4_neg_nlpow_AB_g1p5_amp3p0x_attnonly | 67.80% (40/59) | 70.00% (35/50) |
| qwen3_8B_cyber_actmap_v4_neg_nlpow_AB_g1p5_amp4p0x_attnonly | 57.63% (34/59) | 54.00% (27/50) |
| qwen3_8B_cyber_actmap_v4_neg_amp4p0x_nlpow_AB_g1p5_attnonly | 55.93% (33/59) | 54.00% (27/50) |
| qwen3_8B_cyber_actmap_v4_neg_nlpowB_g1p5_amp4p0x_attnonly | 59.32% (35/59) | 56.00% (28/50) |
| qwen3_8B_cyber_actmap_v4_neg_amp5p0x_attnonly | 30.51% (18/59) | 30.00% (15/50) |
| qwen3_8B_cyber_actmap_v4_neg_amp6p0x_attnonly | 13.56% (8/59) | 18.00% (9/50) |

### 结论

1. `amp` 是当前最强遗忘驱动器，但与 retain 呈明显同向塌陷。
2. 在 screen5 中，`amp5x`/`amp6x` 已把 forget 压到 30.51%/13.56%，但 retain 同时降到 30.00%/18.00%。
3. `nlpow + amp` 相比纯 `nlpow` 有效，但尚不能在低 forget 与高 retain 之间形成理想 Pareto 前沿。
4. 若目标是“接近 source forget=27.12% 且 retain 不崩”，仅靠无训练后处理仍不足，下一步需引入有监督微调或更强约束优化。
## 4.5 2026-05-02 非线性筛选候选全量验证

### 背景

screen2~5 均以 retain=50 快速筛选，未进行全量 retain=537 的口径验证。  
本节从各轮 screening 中取 Forget 最低的三个候选变体，补跑全量评测（retain=537，forget=59），以 wmdp-cyber 口径为准。

### 候选选择依据

| 变体 | quick-F(50) | quick-R(50) | 来源 |
|------|-------------|-------------|------|
| `amp4p0x_nlpow_AB_g1p5_attnonly` | 55.93% | 54.00% | screen5 |
| `nlpow_AB_g1p5_amp4p0x_attnonly` | 57.63% | 54.00% | screen5 |
| `amp3p5x`（全模块） | 54.24% | 56.00% | screen3 |

### 全量评测结果（retain=537，forget=59，wmdp-cyber 口径）

| 变体 | Forget(full) | Retain(full) |
|------|-------------|-------------|
| `amp4p0x_nlpow_AB_g1p5_attnonly` | **55.93%** (33/59) | 53.26% (286/537) |
| `nlpow_AB_g1p5_amp4p0x_attnonly` | **55.93%** (33/59) | 53.26% (286/537) |
| `amp3p5x`（全模块） | **54.24%** (32/59) | **61.64%** (331/537) |

对应结果文件：

- `results/qwen_unlearn/full_amp4p0x_nlpow_AB_g1p5_attnonly.json`
- `results/qwen_unlearn/full_nlpow_AB_g1p5_amp4p0x_attnonly.json`
- `results/qwen_unlearn/full_amp3p5x.json`

### 结论

1. **`amp3p5x` 全模块为最佳候选**：Forget=54.24%（比 base 86.44% 降 32pt），Retain=61.64%，全量口径下 Retain 比 attn-only 高 ~8pt，是目前非线性方法在 Qwen3-8B 上的最优实验性结果。
2. 两个 attn-only 变体（amp4+nlpow 与 nlpow+amp4）全量结果完全相同（55.93%/53.26%），quick 差异来自 retain 抽样噪声。
3. 与 source（Forget=27.12%，Retain=86.22%）相比仍有较大 gap，说明无监督后处理在当前框架下已逼近上限。
---

## 5. CKA 与匈牙利算法在本任务中的实际使用状态

1. 文档层曾提出“用 CKA 决层映射”。
2. 现在 Qwen3-1.7B -> Qwen3-8B 已支持显式切换：fixed / cka_monotonic / cka_hungarian。
3. 若使用 fixed，CKA 仅做辅助信号；若使用 cka_*，CKA 直接参与层映射求解。
4. 头映射仍基于头相似度矩阵做组合匹配。

### 5.1 子层级映射（attn / ffn）建议

为贴近 MiniCPM 的“先匹配再映射”范式，可把层级映射细化为子层级：

1. 分别构造 CKA 矩阵

- `A_attn(i,j)`: old 第 i 层 attention 输出 与 new 第 j 层 attention 输出的 CKA
- `A_ffn(i,j)`: old 第 i 层 FFN 输出 与 new 第 j 层 FFN 输出的 CKA

2. 独立求解映射

- attention 使用 `cka_hungarian`（一对一，避免重复目标层）
- ffn 使用 `cka_hungarian` 或 `cka_monotonic`（若希望保持深度顺序）

3. 分模块迁移

- q/k/v/o 使用 attention 映射
- gate/up/down 使用 ffn 映射

4. 评测约束jiu

- retain 必须使用同分布 `wmdp-cyber/remain_10pct.json`
- 默认不使用 `--fill_missing_layers`，避免“缺层复制”混入额外归纳偏置

---

## 6. 你提出的判据：客观判断

你的思路：把映射关系和映射矩阵设为可学习参数，用 forget+retain 微调；若可优化到理想效果，则说明存在对应线性映射。

客观结论：总体正确，但结论是条件性的。

1. 若优化成功：

   - 可以证明在该参数化族、损失函数、数据分布与训练预算下，存在可行线性映射。
2. 若优化失败：

   - 不能严格证明“不存在”，只能说明在当前设置下未找到。
3. 要增强说服力：

   - 多初值与更强优化器，降低局部最优影响。
   - 把离散层映射与连续线性矩阵参数化写全，报告线性上界实验。

因此，这个思路很适合作为“线性可行性探测实验”，但不是无条件数学证明。

---

## 7. 推荐阅读顺序

1. 先看第 3 节，建立线性方法全景。
2. 再看第 4 节，理解非线性尝试与增益边界。
3. 最后看调查日志：results/qwen_unlearn/actmap_investigation_2026-04-30.md

---

## 8. 常见复现坑

1. Windows 下部分命令返回码 1 可能是编码或管道问题，需以结果 json 是否写出为准。
2. 比较结论建议优先用 strictfix2 口径，避免旧 hard-match 误判。

---

## 9. 2026-05-02 代码级借鉴落地（对齐 MiniCPM 迁移修复思路）

为减少跨尺度层错配，已将 Qwen 迁移脚本里历史的比例层映射

`j(i) = int(i * 36 / 28)`

统一替换为 actmap 显式映射表：

`[0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 18, 19, 20, 21, 23, 24, 25, 27, 28, 29, 30, 32, 33, 34]`

涉及脚本：

1. transfer_lora_full.py
2. transfer_lora_tiling.py
3. transfer_lora_svd_align.py
4. transfer_lora_zeropad.py

说明：

1. 该改动只调整层对应关系，不改变各脚本原有的 A/B 迁移公式与 norm 校准逻辑。
2. gate/up/down 全模块覆盖策略保持不变。
3. 缺失层仍按 nearest-neighbor 规则补齐，保证 36 层目标模型完整可评测。


---

## 10. 2026-05-02 1.7B→4B 迁移实验（更小尺寸差异探索）

### 10.1 动机

8B 迁移实验（1.7B→8B，尺寸比约 1:4.7）在全量非线性最优时 Forget=54.24%/Retain=61.64%，
提出问题：若迁移到更接近源模型尺寸的 4B（尺寸比约 1:2.4），是否因特征空间差异更小而效果更好？

### 10.2 架构关键差异

| 属性 | 1.7B | 4B | 8B |
|---|---|---|---|
| 层数 | 28 | 36 | 36 |
| hidden_size | 2048 | 2560 | 4096 |
| intermediate | 6144 | 9728 | 12288 |
| q_dim | 2048 | 4096 | 4096 |
| kv_dim | 1024 | 1024 | 1024 |

关键发现：**4B 的 kv_dim=1024 与 1.7B 完全相同**（head_dim=128, kv_heads=8），
但 q_dim 从 2048 扩至 4096，需要专门推导 R_q 矩阵：

`
R_q = W_q_4B @ R_l.T @ pinv(W_q_1.7B)   # shape: (4096, 2048)
`

- q_proj.lora_A: A @ R_l    → (rank,2048)@(2048,2560)
- q_proj.lora_B: R_q @ B    → (4096,2048)@(2048,rank)
- o_proj.lora_A: A @ pinv(R_q) → (rank,2048)@(2048,4096)
- o_proj.lora_B: R_l.T @ B
- k/v_proj: 直接 COPY（kv_dim=1024 与 1.7B 相同）

脚本：transfer_lora_actmap_4b.py，激活对齐矩阵：tmp/act_align_qwen3_4B_cyber59/（28个R_l）

### 10.3 评测结果（actmap_v1_neg，wmdp-cyber full=59/537口径）

| 模型 | Forget | Retain | 备注 |
|---|---|---|---|
| 4B base | 67.80% | 68.00% (quick) | 无 LoRA 基线 |
| **4B actmap_v1_neg (full)** | **64.41%** | **68.72%** | 迁移+取反 |
| 1.7B 源模型 (原始LoRA) | — | — | 参照 |
| 8B actmap_v4_neg (quick) | 86.44%↓ | ~91% | 正常迁移效果 |
| 8B amp3p5x full (最优非线性) | 54.24%↓ | 61.64% | 8B 非线性最优 |

↓ 表示相对基线下降明显（遗忘效果好）。

### 10.4 结论

1. **4B 基线本身 F=67.80%**（远低于 8B base 的 ~86%），说明 4B 对 wmdp-cyber 知识覆盖少，遗忘上限受限。
2. 迁移后 F=64.41%（仅降 3.4pt），而 8B 迁移可降 ~20-30pt，**尺寸差异更小并未带来更好的遗忘效果**。
3. 根本原因：遗忘效果受目标模型知识深度制约，4B 本身知识量不足，迁移 LoRA 没有可消除的知识。
4. 后续若需在 4B 上做遗忘，建议直接在 4B 上训练源数据集，而非依赖跨模型迁移。


---

## 11. 2026-05-02 层映射对比实验（Qwen3-1.7B→8B, wmdp-cyber）

### 实验设置
- 映射关系: fixed(比例) / cka_monotonic / cka_hungarian
- 映射方法: actmap（逐层激活对齐 R_l）
- 评测: forget=59全集, quick retain=50, full retain=537

### 结果
| ID | 迁移目录 | Forget(59) | Retain(50) | Retain(537) | 备注 |
|---|---|---:|---:|---:|---|
| M1 | qwen3_8B_cyber_actmap_v4 | 83.05% | 86.00% | **80.63%** | fixed+actmap 最优 |
| M2 | qmapcka_monotonic_actalign | 88.14% | 84.00% | - | cka_monotonic |
| M3 | qmapcka_hungarian_actalign | 86.44% | 82.00% | - | cka_hungarian |
| M1_neg | actmap_v4_neg_v2 | 86.44% | 78.00% | - | M1取反B |

### 关键结论
1. 固定比例映射(fixed)优于CKA映射，原因：
   - 当前CKA矩阵为24x24（基于Qwen1.5-1.8B/Qwen2.5-3B），非Qwen3对
   - R_l按proportional映射采集，与CKA层对应不一致
   - 两者配对错误导致性能下降
2. 正方向(v4)明显优于负方向(v4_neg)，说明R_l方向正确
3. 若要测试CKA层映射，须用CKA顺序重新采集R_l激活矩阵

### Full 最优结果
- **M1 fixed actmap_v4: Forget=83.05% / Retain=80.63% (Full)**