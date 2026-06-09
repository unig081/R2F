# LoRA 跨模型迁移方法论：映射关系与映射方法的理论框架

本文档对项目中所有 LoRA 跨尺度迁移脚本进行系统性方法论梳理，厘清两个核心问题：

1. **映射关系**（Mapping Relationship）：源模型的哪一层 / 哪个注意力头对应目标模型的哪一层 / 哪个头？
2. **映射方法**（Mapping Method）：源模型的 LoRA 权重如何在代数上变换到目标模型空间？

每个部分分别给出"简单基线"（无理论依据）和"有理论依据的设计"，并标注本项目的实际实现状态。

---

## 目录

1. [问题形式化](#1-问题形式化)
2. [Part I：映射关系](#2-part-i映射关系)
   - 2.1 层级映射
   - 2.2 注意力头映射
3. [Part II：映射方法](#3-part-ii映射方法)
   - 3.1 隐藏维度对齐（hidden-dim）
   - 3.2 FFN 中间维度对齐（intermediate-dim）
   - 3.3 LoRA 因子的代数变换推导
   - 3.4 特殊结构处理（q_dim ≠ hidden_dim）
4. [现有实现状态](#4-现有实现状态)
5. [差距分析与改进方向](#5-差距分析与改进方向)
6. [实验结果汇总](#6-实验结果汇总)

---

## 1. 问题形式化

给定：

- 源模型 $M_s$（如 Qwen3-1.7B，$L_s$ 层，隐藏维度 $d_s$），已训练好 LoRA $\{A_i^s, B_i^s\}_{i=0}^{L_s-1}$（对应某任务，如 wmdp-cyber 遗忘）
- 目标模型 $M_t$（如 Qwen3-8B，$L_t$ 层，隐藏维度 $d_t$），从未见过该任务

目标：构造 $\{A_j^t, B_j^t\}_{j=0}^{L_t-1}$，使得在 $M_t$ 上应用迁移后的 LoRA 具有与 $M_s$ 应用原始 LoRA 相近的**功能效果**（这里的"功能"是指在遗忘数据集上减少知识、在保留数据集上维持能力）。

$$
\text{Transfer}: \{A_i^s, B_i^s, M_s, M_t\} \longrightarrow \{A_j^t, B_j^t\}
$$

---

## 2. Part I：映射关系

### 2.1 层级映射（Layer Mapping）

**目标**：建立从源层 $i \in \{0, \ldots, L_s-1\}$ 到目标层 $j \in \{0, \ldots, L_t-1\}$ 的映射函数 $\phi: i \mapsto j$。

#### 基线：比例映射（Proportional / Fixed）

$$
j = \left\lfloor i \cdot \frac{L_t}{L_s} \right\rfloor
$$

**1.7B (28层) → 8B (36层) 的映射表：**
```
[0,1,2,3,5,6,7,9,10,11,12,14,15,16,18,19,20,21,23,24,25,27,28,29,30,32,33,34]
```
8 个未被映射的目标层 `{4,8,13,17,22,26,31,35}` 用 nearest-neighbor 补齐。

**逻辑**：假设同系列模型各层的"相对深度位置"决定其语义功能，浅层负责语法/词法，中层负责语义组合，深层负责任务推理。

**缺陷**：
- 仅假设层深度比例一致，忽略同系列模型中层功能不均匀分布的问题（例如残差结构中不同层的有效信息传递量不同）。
- 对于层数差异较大的模型（如 28→36 有 8 层被跳过），信息丢失较大。

#### 有理论依据的方案：CKA 相似度驱动的层映射

**中心核对齐（CKA）**定义于两个层的激活矩阵之间：

$$
\text{CKA}(H_i^s, H_j^t) = \frac{\|H_j^{t\top} H_i^s\|_F^2}{\|H_i^{s\top} H_i^s\|_F \cdot \|H_j^{t\top} H_j^t\|_F}
$$

其中 $H_i^s \in \mathbb{R}^{N \times d_s}$ 是源模型第 $i$ 层在 $N$ 个样本上的激活均值池化矩阵。CKA 取值 $[0,1]$，越高表示两层的表示空间越相似（同变换不变性）。

**理论依据**：同系列模型（Qwen3-1.7B 与 Qwen3-8B 均从相同数据预训练）的对应深度层具有高度相似的表示空间——这一点已被多篇 scaling 分析论文实验验证（如 Nguyen et al., 2021 "Do Wide and Deep Networks Learn the Same Things?"）。高 CKA 的层对具有相似的函数语义，因此迁移精度更高。

本项目中预计算的 CKA 矩阵存储在：
```
tmp/Qwen1.5-1.8B_Qwen2.5-3B_CKA.pt
tmp/Llama-2-7b-hf_Meta-Llama-3-8B_CKA.pt
tmp/MiniCPM-S-1B-sft-llama-format_MiniCPM-2B-sft-fp32-llama-format_CKA.pt
```

基于 CKA 的层映射有两种变体（已在 `lora_adaption.py` 实现）：

**方案 A：DP 单调最大相似度映射（cka_monotonic）**

约束映射必须保持层深度的单调顺序（$\phi(i) < \phi(i+1)$），在此约束下通过动态规划最大化总 CKA：

$$
\max_{\phi: \text{单调}} \sum_{i=0}^{L_s-1} \text{CKA}(H_i^s, H_{\phi(i)}^t)
$$

**理由**：同系列模型中层级功能通常保持单调顺序（底层→高层不逆），单调约束既减少搜索空间，又防止语义功能交叉映射。

**方案 B：匈牙利算法全局最优匹配（cka_hungarian）**

不加单调约束，求一对一全局最优匹配：

$$
\max_{\phi: \text{一对一}} \sum_{i=0}^{L_s-1} \text{CKA}(H_i^s, H_{\phi(i)}^t)
$$

通过匈牙利算法（`scipy.optimize.linear_sum_assignment`）$O(n^3)$ 精确求解。

**理由**：允许非单调匹配，能发现深层浅层之间的功能对应（如 early exit 现象下的层角色交换）。适合两模型预训练配置差异较大时使用。

**当前实验状态**（`collect_activations.py` 的层映射）：
> ⚠️ `collect_activations.py` 计算 R_l 时使用的层配对仍是比例映射，与 CKA 无关。CKA 方法仅在 `lora_adaption.py` 中被实现并作为可选项，**actmap 版迁移脚本中全程使用的是比例映射的固定列表**。

---

### 2.2 注意力头映射（Head Mapping）

**目标**：在确定层映射后，建立从源层 $i$ 的第 $h$ 个头到目标层 $j$ 的第 $h'$ 个头的映射 $\psi_{ij}: h \mapsto h'$。

这在 $n_{\text{heads}}^s \neq n_{\text{heads}}^t$ 时（如 1B→2B 的 24 heads→36 heads）不可回避；即使头数相同，各头的功能专门化（induction head、retrieval head 等）在不同模型中也可能位置不同。

#### 基线：隐式映射（Implicit / Monolithic）

将整个注意力输出空间视为整体，通过隐藏维度对齐矩阵 $R_l$ 进行全局变换，**不区分各个头的对应关系**。

`transfer_lora_actmap.py` 即采用此方式：

```python
# q_proj.lora_A: (rank, d_s) → (rank, d_t) via A @ R_l
# q_proj.lora_B: (d_s, rank) → (d_t, rank) via R_l.T @ B
```

**缺陷**：忽略了 head 粒度的功能专门化。若两模型的第 $k$ 号头功能分别对应对方第 $k'$ 号头（$k' \neq k$），整体映射会混合不同功能的 head 信息，造成干扰。

#### 有理论依据的方案：CKA 相似度驱动的 Head 匹配

计算每个头的权重 CKA：

$$
\text{CKA}(W_{h}^{q,s}, W_{h'}^{q,t}) = \frac{\|W_{h'}^{q,t\top} W_h^{q,s}\|_F^2}{\|W_h^{q,s\top} W_h^{q,s}\|_F \cdot \|W_{h'}^{q,t\top} W_{h'}^{q,t}\|_F}
$$

再通过**匈牙利算法**对每层求最优头-头一对一匹配（已在 `lora_adaption.py` 中实现为 `max_head_similarity_mapping`）。

**理由**：注意力头的功能专门化（pattern specialization）是跨模型可以被权重相似度捕捉的。高 CKA 的两个头执行相同类型的注意力操作（如句法依存、共指消解），迁移该对头的 LoRA 更有意义。

在找到头匹配 $\psi$ 后，对每个头分别计算 per-head Procrustes 变换矩阵 $L_h$（`lora_adaption.py` 的 `apply_xform_with_prolora`）。

**当前状态**：
> ⚠️ actmap 迁移脚本（`transfer_lora_actmap.py`, `transfer_lora_actmap_4b.py`）均**不做显式 head 映射**，依赖整体 R_l 隐式覆盖。

---

## 3. Part II：映射方法

### 3.1 隐藏维度对齐（Hidden-Dim Alignment）

**目标**：找到线性映射 $R_l: \mathbb{R}^{d_s} \to \mathbb{R}^{d_t}$，使得：

$$
\mathbf{h}_l^t \approx R_l^\top \mathbf{h}_l^s \quad \text{（列向量约定）}
$$

或等价地（行向量）：$\mathbf{h}_{l,\text{row}}^s \cdot R_l \approx \mathbf{h}_{l,\text{row}}^t$，其中 $R_l \in \mathbb{R}^{d_s \times d_t}$。

#### 基线 A：嵌入导出的全局变换 $W_x$

从共享词表的嵌入矩阵推导全局（层无关）的隐藏空间变换：

$$
W_x = E_s^+ \cdot E_t \in \mathbb{R}^{d_s \times d_t}
$$

其中 $E_s, E_t$ 是共同词汇的嵌入行子矩阵（已实现于 `lora_adaption.py` 默认路径）。

**缺陷**：同一矩阵应用于所有层，忽略了不同层表示空间几何结构的差异。在 Transformer 中，浅层表示词法特征，深层表示语义抽象，用统一的嵌入对齐矩阵覆盖所有层是过强假设。

#### 有理论依据的方案：逐层激活对齐（Per-Layer Activation Ridge Regression）

对每层分别求解：

$$
R_l = \arg\min_R \|H_l^s R - H_l^t\|_F^2 + \lambda \|R\|_F^2
$$

其中 $H_l^s \in \mathbb{R}^{N \times d_s}$, $H_l^t \in \mathbb{R}^{N \times d_t}$ 是在 $N$ 个通用文本样本上收集的第 $l$ 层激活均值池化矩阵。

用最小二乘（lstsq, `gelsd` driver）数值求解（已在 `collect_activations.py` 实现）：

```python
result = torch.linalg.lstsq(Ho, Hn, rcond=reg, driver='gelsd')
R_l = result.solution  # (d_s, d_t)
```

**理由**：
- 捕捉每层的**运行时**表示几何（而非仅嵌入空间的初始化几何）。
- ridge 正则化防止样本数 $N \ll d_s$ 时出现过拟合（本项目 $N=64$, $d_s=2048$，严重欠定）。
- 使用**通用**文本（数学题目）而非任务文本，确保 $R_l$ 反映模型通用的隐藏空间结构，而非任务特化的方向。

**注意**：$N \ll d_s$（64 vs 2048）意味着 $R_l$ 只在 $N$ 维激活流形上有意义；在该流形的正交补方向上，$R_l$ 是 arbitrary（零初始化）。这是 actmap 方法的根本限制，也是为什么 `act_align_blend < 1.0` 可以通过插值 $W_x$ 进行正则化。

---

### 3.2 FFN 中间维度对齐（Intermediate-Dim Alignment）

当源模型 FFN intermediate 维度为 $d_{\text{ffn}}^s$，目标模型为 $d_{\text{ffn}}^t$（通常 $d_{\text{ffn}}^t > d_{\text{ffn}}^s$）时，需要一个中间维度对齐矩阵 $P_l \in \mathbb{R}^{d_{\text{ffn}}^t \times d_{\text{ffn}}^s}$。

#### 基线：零填充或均匀平铺

- **零填充**：在多余的 $d_{\text{ffn}}^t - d_{\text{ffn}}^s$ 个神经元上填零，即"这些神经元未被激活"。
- **均匀平铺**（tiling）：将源的 intermediate 向量重复平铺直到目标维度。

**缺陷**：零填充假设目标模型的多余神经元与源知识无关，平铺则假设目标神经元排列与源完全相同——两者均无理论依据。

#### 有理论依据的方案：基于权重的解析推导

**核心思路**：若我们已知 hidden 维对齐 $R_l$，则可以从门控投影权重**解析地**导出 intermediate 对齐矩阵，无需额外数据。

以 gate_proj 为例：源模型 $W_g^s \in \mathbb{R}^{d_{\text{ffn}}^s \times d_s}$，目标模型 $W_g^t \in \mathbb{R}^{d_{\text{ffn}}^t \times d_t}$。

给定 $R_l$，若输入 $x^t \approx R_l^\top x^s$，则期望中间激活的映射关系为：

$$
v^t = W_g^t x^t \approx W_g^t R_l^\top x^s = \underbrace{(W_g^t R_l^\top (W_g^s)^+)}_{\displaystyle P_l} \cdot v^s
$$

即：

$$
\boxed{P_l = W_g^t \cdot R_l^\top \cdot (W_g^s)^+}
$$

其中 $(W_g^s)^+ = \text{pinv}(W_g^s)$ 是 Moore-Penrose 伪逆（已在 `transfer_lora_actmap.py` 的 `compute_P` 函数实现）。

**理由**：此推导直接从"希望两模型的中间激活功能等价"出发，无需额外假设，利用了模型本身的权重信息。当 $W_g^s$ 行满秩时 $(W_g^s)^+$ 精确，行亏秩时退化为最小范数解（ridge 正则化保证数值稳定性）。

**柱范数裁剪（Column Norm Clipping）**：$P_l$ 的每列对应源模型一个 intermediate 神经元映射到目标空间，条件数过大时会放大 LoRA 权重。实现中裁剪列范数超过中位数 3 倍的列：

```python
col_norms = P_l.norm(dim=0, keepdim=True).clamp(min=1e-8)
max_norm = (3.0 * col_norms.median()).clamp(min=1.0)
scale = (max_norm / col_norms).clamp(max=1.0)  # 只缩小，不放大
P_l = P_l * scale
```

---

### 3.3 LoRA 因子的代数变换推导

LoRA 将权重更新分解为低秩矩阵乘积：$\Delta W = B \cdot A$（或 $\Delta W = A^\top B$ 取决于实现），其中 $A \in \mathbb{R}^{r \times d_{\text{in}}}$, $B \in \mathbb{R}^{d_{\text{out}} \times r}$（PEFT 约定）。

设线性变换 $R: \mathbb{R}^{d_{\text{in}}^s} \to \mathbb{R}^{d_{\text{in}}^t}$ 对输入空间对齐，$Q: \mathbb{R}^{d_{\text{out}}^s} \to \mathbb{R}^{d_{\text{out}}^t}$ 对输出空间对齐，则：

$$
\Delta W^t = Q \cdot \Delta W^s \cdot R = Q \cdot B^s \cdot A^s \cdot R = \underbrace{(Q B^s)}_{B^t} \cdot \underbrace{(A^s R)^\top \cdot R^{-\top}}_{A^t}
$$

更自然地：

$$
\boxed{A^t = A^s \cdot R, \quad B^t = Q \cdot B^s}
$$

其中：
- $A^s \in \mathbb{R}^{r \times d_{\text{in}}^s}$，$A^t = A^s R \in \mathbb{R}^{r \times d_{\text{in}}^t}$
- $B^s \in \mathbb{R}^{d_{\text{out}}^s \times r}$，$B^t = Q B^s \in \mathbb{R}^{d_{\text{out}}^t \times r}$

验证：$B^t A^t = Q B^s A^s R = Q \Delta W^s R = \Delta W^t$ ✓

**各模块的 $R$ 与 $Q$ 对应关系（1.7B→8B，$R_l \in \mathbb{R}^{2048 \times 4096}$, $P_l \in \mathbb{R}^{12288 \times 6144}$）：**

| 模块 | $d_{\text{in}}^s \to d_{\text{in}}^t$ | $d_{\text{out}}^s \to d_{\text{out}}^t$ | $R$ (in-空间) | $Q$ (out-空间) |
|------|------|------|------|------|
| q_proj | $2048 \to 4096$ | $2048 \to 4096$ | $R_l$ | $R_l^\top$ |
| k_proj | $2048 \to 4096$ | $1024 \to 1024$ | $R_l$ | $I$ (COPY) |
| v_proj | $2048 \to 4096$ | $1024 \to 1024$ | $R_l$ | $I$ (COPY) |
| o_proj | $2048 \to 4096$ | $2048 \to 4096$ | $R_l$ | $R_l^\top$ |
| gate_proj | $2048 \to 4096$ | $6144 \to 12288$ | $R_l$ | $P_l$ |
| up_proj | $2048 \to 4096$ | $6144 \to 12288$ | $R_l$ | $P_l$ |
| down_proj | $6144 \to 12288$ | $2048 \to 4096$ | $P_l$ | $R_l^\top$ |

**KV proj 直接 COPY 的理由**：Qwen3 系列中 1.7B / 4B / 8B 的 KV head 维度均为 $d_{\text{kv}} = 8 \times 128 = 1024$，输出空间相同，因此 $Q = I$（恒等变换），$B^t = B^s$（直接复制）。

---

### 3.4 特殊结构：q_dim ≠ hidden_dim（4B 模型）

Qwen3-4B 的特殊性（$d_{\text{hidden}}=2560$, $q_{\text{dim}} = n_{\text{heads}} \times d_{\text{head}} = 32 \times 128 = 4096 \neq 2560$）：

q_proj 的输入是 hidden（$2560$），输出是 q_dim（$4096$）。用 $R_l \in \mathbb{R}^{2048 \times 2560}$ 处理输入没问题，但输出空间对齐不能再用 $R_l^\top$（维度不匹配）。

**解决方案**：类比 $P_l$ 的导出方式，从 q_proj 权重解析推导 $R_q$：

$$
R_q = W_q^t \cdot R_l^\top \cdot (W_q^s)^+ \in \mathbb{R}^{4096 \times 2048}
$$

则：
- q_proj.lora_B: $B^t = R_q \cdot B^s \in \mathbb{R}^{4096 \times r}$
- o_proj.lora_A: o_proj 的输入 = q_dim，$A^t = A^s \cdot (R_q)^+ \in \mathbb{R}^{r \times 4096}$

已在 `transfer_lora_actmap_4b.py` 实现。

---

## 4. 现有实现状态

| 组件 | 描述 | 实现状态 | 是否有理论依据 |
|------|------|------|------|
| **层映射** | | | |
| 比例映射 | `j = int(i * L_t / L_s)` | ✅ 所有 actmap 脚本默认 | ❌ 简单基线 |
| CKA 单调映射 | DP 最优单调匹配 | ✅ `lora_adaption.py --cka_monotonic` | ✅ 有理论依据 |
| CKA 匈牙利映射 | 全局最优一对一匹配 | ✅ `lora_adaption.py --cka_hungarian` | ✅ 有理论依据 |
| **头映射** | | | |
| 隐式映射（无头级区分） | R_l 整体变换 | ✅ actmap 脚本 | ❌ 简单基线 |
| CKA 匈牙利头匹配 | 权重 CKA + 匈牙利 | ✅ `lora_adaption.py` | ✅ 有理论依据 |
| **隐藏维度对齐** | | | |
| 嵌入导出 $W_x$ | 共享词汇嵌入 lstsq | ✅ `lora_adaption.py` 默认 | ⚠️ 仅限嵌入空间 |
| 激活对齐 $R_l$ | per-layer ridge regression | ✅ `collect_activations.py` + actmap 脚本 | ✅ 有理论依据 |
| **FFN 维度对齐** | | | |
| 零填充 / 平铺 | transfer_lora_zeropad/tiling | ✅ | ❌ 简单基线 |
| 解析 $P_l$ | $W_g^t R_l^\top (W_g^s)^+$ | ✅ actmap 脚本 | ✅ 有理论依据 |
| **LoRA 因子变换** | | | |
| 简单右乘 $W_x$ | 仅变换 A 矩阵 | ✅ 早期版本 | ❌ 简单基线 |
| 双侧变换 $A R$, $Q B$ | 同时变换 A 和 B | ✅ actmap 脚本 | ✅ 数学正确 |
| 谱校准 | 匹配源/目标 singular value 分布 | ✅ `--spectral_calibrate` | ✅ 有理论依据 |
| ProLoRA 子空间滤波 | 保留与基础权重平行的分量 | ✅ `--use_prolora` | ✅ 有理论依据 |
| per-head Procrustes $L$ | 头空间正交变换 | ✅ `--procrustes_L` | ✅ 有理论依据 |

---

## 5. 差距分析与改进方向

### Gap 1：actmap 版层映射仍为比例映射

**问题**：`collect_activations.py` 内部使用 `int(i * L_t / L_s)` 决定层配对关系来计算 $R_l$，这意味着 $R_l$ 本身是基于比例映射对齐的激活，而非 CKA 最优层配对的激活。

**改进方向**：
1. 先用 `lora_adaption.py` 的 CKA 单调/匈牙利方法确定最优层映射
2. 将该映射传入 `collect_activations.py` 替换内部比例映射
3. 用 CKA 配对的激活重新计算 $R_l$

**预期效果**：$R_l$ 的重建误差（reconstruction error）会降低，迁移精度提升。

### Gap 2：actmap 版无显式头级映射

**问题**：当源目标模型头数不同或头功能排列不同时，整体 $R_l$ 将不同功能的 head 混合，相当于对 Q/O 空间做了非结构化的线性变换。

**改进方向**：
1. 在 actmap 路径中引入 head-level CKA 计算（基于 $W_q$ 权重矩阵）
2. 对每层先做头匹配，再对匹配后的 per-head delta 应用变换
3. 对于头数增多的情况（如 8→32 heads），未匹配的目标头用 nearest-neighbor 头的变换填充

### Gap 3：$N \ll d_s$ 下 $R_l$ 欠定问题

**问题**：$N=64$ 个样本，$d_s=2048$，$R_l$ 只在 64 维激活流形上有约束，其余 $2048-64$ 维方向是 arbitrary。

**改进方向**：
1. 增加 `--n_samples` 到 256～512（覆盖更多流形方向）
2. 用 domain-specific 数据（wmdp-cyber 文本）替代通用数据，使 $R_l$ 更准确地刻画遗忘相关子空间
3. `act_align_blend` 参数可插值 $W_x$ 正则化欠定方向

### Gap 4：$P_l$ 推导基于通用 gate_proj，未考虑遗忘目标

**问题**：$P_l$ 用通用权重推导，对于"遗忘"任务，FFN 中哪些 intermediate 神经元对 cyber 知识贡献最大未被纳入考量。

**改进方向**：
- 对 cyber 相关 prompt 提取 intermediate 激活，构建专用的 intermediate 对齐矩阵（类似 `collect_activations.py` 但针对 intermediate 层）

---

## 6. 实验结果汇总

> 评测口径：wmdp-cyber forget=59全集 / retain=537全量，hard match，judge=none

### Qwen3 1.7B → 8B（主线）

| 方法 | Forget↓ | Retain↑ | 备注 |
|------|------|------|------|
| 8B base（无 LoRA） | ~86% | ~91% | 基线 |
| actmap_v4（比例层映射 + $R_l$ + $P_l$） | — | — | v4 quick: F=83.05% |
| actmap_v4_neg | — | — | quick: F=86.44% |
| **amp3p5x（非线性扩展，全模块）** | **54.24%** | **61.64%** | 当前最优 |

### Qwen3 1.7B → 4B（对照）

| 方法 | Forget↓ | Retain↑ | 备注 |
|------|------|------|------|
| 4B base（无 LoRA） | 67.80% | 68.00% (quick) | 4B 基线 |
| **4B actmap_v1_neg** | **64.41%** | **68.72%** | 仅降 3.4pt |
| 结论 | 4B 对 wmdp-cyber 知识覆盖少，遗忘空间受限 | | |

### MiniCPM 1B → 2B（参照）

| 方法 | Forget↓ | Retain↑ | 备注 |
|------|------|------|------|
| 2B actmap all-modules neg1.25 | 52.54% | 61.45% | 当前最优 |

---

## 7. 总结

本项目的 LoRA 迁移方法已经在以下方面做了有理论依据的设计：

- **映射方法**中的 $R_l$（激活对齐）和 $P_l$（权重解析推导）是**有理论依据**的，优于基线。
- **映射关系**的 CKA 层映射和 head-level 匈牙利匹配在 `lora_adaption.py` 中实现，但**actmap 版本仍使用比例映射**（简单基线）。

主要待补的理论缺口：
1. actmap 路径中引入 CKA 层配对（Gap 1）
2. actmap 路径中引入 head-level 匹配（Gap 2）
3. 用更多样本或 domain-specific 数据改善 $R_l$ 覆盖（Gap 3）
