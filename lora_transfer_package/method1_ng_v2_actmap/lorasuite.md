# LoRASuite: 高效的大语言模型升级 LoRA 适配方案总结

## 1. 核心洞察 (Insight)

随着大语言模型（LLM）版本的频繁迭代（如 Llama-2 $\to$ Llama-3, Qwen-1.5 $\to$ Qwen-2.5），在旧版本模型上训练的 LoRA 权重无法直接用于新版本。传统的做法是**从头重新训练（Retrain from Scratch）**，这不仅耗时耗力，而且计算成本高昂。

**LoRASuite 的核心洞察在于：**

虽然模型架构参数（如隐藏层维度、层数、注意力头数等）发生了变化，但新旧模型之间存在内在的语义和结构对应关系。通过利用已知的新旧模型参数，可以构建**转移矩阵（Transfer Matrix）**和**映射算法（Mapping Algorithm）**，将旧模型的 LoRA 权重“迁移”到新模型上。这种迁移后的权重虽然不能直接使用（存在数值不稳定），但作为一个极佳的初始化点，只需极少数据的轻量级微调（Lightweight Fine-tuning, LFT）即可达到甚至超越全量数据重新训练的效果。

## 2. 问题定义与挑战

论文将 LLM 升级带来的差异归纳为六个显式限制因素，阻碍了 LoRA 权重的直接复用：

1. **词表大小 (Vocabulary Size)**
2. **隐藏层维度 (Hidden Size)**
3. **中间层维度 (Intermediate Size, FFN up/down proj)**
4. **层深度 (Layer Depth)**
5. **注意力头数量 (Attention Head Count)**
6. **注意力机制类型 (Attention Type, e.g., MHA vs GQA)**

LoRASuite 针对同一架构家族内的模型升级（如 `LlamaForCausalLM` 内部版本迭代）提出了一套模块化解决方案。

## 3. 具体方法 (Methodology)

LoRASuite 的流程分为四个主要步骤：**维度变换**、**层映射**、**头映射**、**轻量级微调**。

### 3.1 维度变换：词表与隐藏层/中间层对齐

当隐藏层维度 $d_{hidden}$ 或词表大小 $V$ 发生变化时，需要计算转移矩阵。

#### A. 隐藏层维度变换 ($W_h$)

假设旧模型嵌入权重为 $E_o \in\mathbb{R}^{V_o \times d_o}$，新模型为 $E_n \in\mathbb{R}^{V_n \times d_n}$。

若仅隐藏层维度变化（词表相同或已处理交集），转移矩阵 $W_h$ 可通过线性回归思想求解：

$$
W_h = E_o^{\dagger} E_n
$$

其中 $E_o^{\dagger}$ 是 $E_o$ 的伪逆。在实际操作中，为了稳定性，通常使用共享 token 的子集来计算。

#### B. 中间层维度变换 ($W_i$)

对于 FFN 中的 Up/Down 投影层，设旧权重为 $W_o$，新权重为 $W_n$。转移矩阵 $W_i$ 计算如下：

$$
W_i = W_o^{-1} W_n
$$

*注意：这里利用的是预训练好的基座模型权重，而非 LoRA 权重。*

### 3.2 层映射：基于 CKA 的动态规划

由于新旧模型层数可能不同（例如 32 层 $\to$ 48 层），需要找到最优的层对应关系。

**步骤 1: 计算相似度矩阵**

使用 **Centered Kernel Alignment (CKA)** 衡量旧模型第 $i$ 层和新模型第 $j$ 层的表示相似性。

给定输入批次 $X$，两层激活输出分别为 $H_o^i$ 和 $H_n^j$。

CKA 基于 HSIC (Hilbert-Schmidt Independence Criterion)：

$$
\text{HSIC}(K, L) = \frac{1}{(m-1)^2} \text{tr}(K H L H)
$$

$$
\text{CKA}(H_o^i, H_n^j) = \frac{\text{HSIC}(H_o^i, H_n^j)}{\sqrt{\text{HSIC}(H_o^i, H_o^i) \text{HSIC}(H_n^j, H_n^j)}}
$$

其中 $K, L$ 是线性核矩阵，$H$ 是中心矩阵。为了节省显存，采用 Minibatch 估计 CKA。

得到相似度矩阵 $S \in\mathbb{R}^{L_o \times L_n}$，其中 $S_{i,j} = \text{CKA}(Layer_o^i, Layer_n^j)$。

**步骤 2: 动态规划寻找最优路径**

目标是最大化总相似度，同时保持层的顺序性（有序映射）。

定义 $dp[i][j]$ 为旧模型前 $i$ 层与新模型前 $j$ 层匹配的最大相似度之和。

约束：允许的最大偏移量为 $\Delta_{layer}$。

```python

# 伪代码：CKA-based Layer Mapping

defcka_layer_mapping(S, L_o, L_n, delta):

    dp = -inf * ones((L_o, L_n))

    path = zeros((L_o, L_n), dtype=int)

  

    # 初始化第一行

    for j inrange(0, min(L_n, delta + 1)):

        dp[0][j] = S[0][j]

      

    for i inrange(1, L_o):

        for j inrange(i, min(L_n, i + delta + 1)):

            max_val = -inf

            max_k = -1

            # 搜索前一层的可能匹配点 k

            for k inrange(max(0, j - delta), j):

                if dp[i-1][k] + S[i][j] > max_val:

                    max_val = dp[i-1][k] + S[i][j]

                    max_k = k

            dp[i][j] = max_val

            path[i][j] = max_k

          

    # 回溯获取映射字典 L_dict: old_layer_idx -> new_layer_idx

    return backtrack(path)

```

### 3.3 注意力头映射：基于匈牙利算法

即使在同一层内，注意力头的数量也可能变化（例如 32 头 $\to$ 36 头，或 MHA $\to$ GQA）。

**步骤 1: 构建头交互矩阵**

对于每个头 $h$，定义两个与输入无关的交互矩阵来表征其功能：

1. $W_{QK}^h = W_Q^h (W_K^h)^T$：捕捉 Token 间的注意力强度。
2. $W_{VO}^h = W_V^h (W_O^h)^T$：捕捉 attending 后对隐藏状态的影响。

**步骤 2: 计算头相似度**

对于旧模型的头 $h_o$ 和新模型的头 $h_n$，计算它们交互矩阵的余弦相似度。如果维度不一致，先使用 $W_h$ 进行投影对齐。

$$
\text{Sim}(h_o, h_n) = \text{CosineSimilarity}(\text{Vec}(W_{QK}^{h_o}), \text{Vec}(W_{QK}^{h_n})) + \text{CosineSimilarity}(\text{Vec}(W_{VO}^{h_o}), \text{Vec}(W_{VO}^{h_n}))
$$

**步骤 3: 匈牙利算法求解最优匹配**

构建代价矩阵（相似度矩阵），使用匈牙利算法（Hungarian Algorithm）找到一对一的最大权重匹配。

* 如果新模型头数多于旧模型：未匹配的旧头可以复制，或者新头随机初始化/从零开始。
* 如果涉及 GQA：先将 K/V 头复制以匹配 Q 头数量，再进行映射。

### 3.4 权重转换与重构

对于映射到的每一对层 $(l_o, l_n)$ 和头 $(h_o, h_n)$：

1. **提取旧 LoRA 更新量**：

   $$
   \Delta W_o = B_o A_o
   $$
2. **分割到头级别**：

   将 $\Delta W_o$ 按照旧模型的头维度分割，得到 $\Delta W_o^{h_o}$。
3. **应用转移矩阵**：

   新模型对应头的权重更新量 $\Delta W_n^{h_n}$ 计算如下：

   $$
   \Delta W_n^{h_n} = W_h^T \cdot\Delta W_o^{h_o} \cdot (W_{proj, o}^{h_o})^T \cdot W_h \cdot W_{proj, n}^{h_n}
   $$

   *注：公式 (3) 在原文中略有简化，核心思想是利用基座权重的几何关系将 $\Delta W$ 从旧空间投影到新空间。更直观的理解是：$\Delta W_n \approx T_{in}^{-1} \Delta W_o T_{out}$，其中 $T$ 是由基座权重导出的变换。*

   原文公式 (3) 具体为：

   $$
   (\Delta W_Q^n)_j = W_h^T \cdot (\Delta W_Q^o)_i \cdot ((W_Q^o)_i)^T \cdot W_h \cdot (W_Q^n)_j
   $$

   这里利用了 $(W_Q^o)_i$ 和 $(W_Q^n)_j$ 作为基底变换的一部分，确保数值稳定性。
4. **SVD 分解恢复 LoRA 格式**：

   对转换后的 $\Delta W_n^{h_n}$ 进行奇异值分解（SVD），保留前 $r$ 个奇异值，重构为新的 $A_n, B_n$：

   $$
   \Delta W_n^{h_n} \approx U \Sigma V^T \implies B_n = U \sqrt{\Sigma}, A_n = \sqrt{\Sigma} V^T
   $$

### 3.5 轻量级微调 (Lightweight Fine-tuning, LFT)

由于上述过程仅涉及矩阵运算，未经过梯度反向传播，可能存在数值误差。因此，最后一步是使用极小规模的数据集（如 100-1000 条样本）对新模型的 LoRA 参数进行微调。

**关键技巧：**

* **无 Warm-up**：因为参数已经接近最优解，不需要线性预热。
* **较高学习率**：相比从头训练，可以使用稍高的学习率以快速收敛。

## 4. 实验设置 (Experimental Setup)

* **基准模型对**：

  * MiniCPM-S-1B $\to$ MiniCPM-2B (涵盖所有6种变化)
  * Yi-6B $\to$ Yi-1.5-9B
  * Pythia-1B $\to$ Pythia-1.4B
  * Bloom-560m $\to$ Bloomz-1B1
  * Llama-2-7B $\to$ Llama-3-8B
  * Qwen-1.5-1.8B $\to$ Qwen-2.5-3B
* **任务**：

  * 数学推理：GSM8K, MAWPS, SVAMP, AddSub, MultiArith, SingleEq, AQuA
  * 常识推理：BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC-c/e, OBQA
* **对比基线**：

  * `Vanilla`: 新模型不加 LoRA。
  * `LoRA (Full)`: 在新模型上使用全量数据（10k samples）从头训练 LoRA。
  * `LoRA (Small)`: 在新模型上使用小量数据（100-1k samples）从头训练 LoRA。
  * `LoRASuite w/o LFT`: 仅迁移，不微调。
  * `LoRASuite w/ LFT`: 迁移后 + 小量数据微调。
* **超参数**：

  * LoRA Rank: 32
  * Alpha: 32
  * Dropout: 0
  * Optimizer: AdamW
  * LFT Learning Rate: $1e-3$ (高于常规的 $3e-4$)
  * LFT Epochs: 3
  * Batch Size: 16

## 5. 实验结论与观察 (Results & Observations)

### 5.1 性能表现

* **超越全量重训**：在 MiniCPM 和 Qwen 升级场景中，LoRASuite (w/ LFT) 在数学任务上的平均得分分别比全量重训高出 **+1.4** 和 **+6.6** 分。
* **显著优于小样本重训**：在所有测试模型对中，LoRASuite 均大幅优于同等数据规模下的从头训练（LoRA Small）。例如在 Qwen 升级中，性能提升近 3 倍。
* **迁移的有效性**：`LoRASuite w/o LFT` 的性能通常略低于或持平于 Vanilla 模型，说明单纯的矩阵变换引入了噪声，验证了 LFT 的必要性。

### 5.2 效率提升

* **时间节省**：相比全量重训，LoRASuite 减少了 **78.23%** 的训练时间。
* **显存节省**：峰值显存占用减少 **5.5 GB**（主要得益于无需加载大量数据和维持完整的优化器状态进行长时间训练）。

### 5.3 消融实验与敏感性分析

* **层映射策略**：基于 CKA 的动态规划映射优于简单的“首尾对应”或“中间扩散”策略。
* **相似度度量**：CKA 优于 CCA、Procrustes 和 PWCCA。
* **头映射策略**：基于匈牙利算法的匹配优于直接按顺序映射或直接代数计算。
* **学习率敏感**：LoRASuite 对 LFT 阶段的学习率非常敏感。最佳学习率 ($9e-4$) 比常规 LoRA 训练更高，且性能提升巨大（相比 $1e-4$ 提升约 21%）。
* **数据规模**：随着 LFT 数据量增加，LoRASuite 的优势逐渐缩小，甚至在大数据量下可能因过拟合旧知识而性能下降。这表明 LoRASuite 最适合**低资源/快速适配**场景。

### 5.4 泛化性

* 该方法不仅适用于标准 LoRA，也适用于 **AdaLoRA** 和 **DoRA** 等变体，能显著提升它们在模型升级后的适配效果。
* 支持合并后的 LoRA 权重迁移（如 TIES-Merging 后的适配器）。

## 6. 复现指南 (For Algorithm Engineers)

若要复现 LoRASuite，请遵循以下流程：

1. **准备环境**：

   * 加载旧模型 $M_o$ 和新模型 $M_n$ 的基座权重（Base Weights）。
   * 加载在 $M_o$ 上训练好的 LoRA 权重 $(A_o, B_o)$。
2. **预计算变换矩阵**：

   * 计算嵌入层转移矩阵 $W_h$。
   * 计算 FFN 层转移矩阵 $W_i$（如果需要适配 up/down proj）。
3. **执行层映射**：

   * 采样一个小批次数据，通过 $M_o$ 和 $M_n$ 获取各层激活。
   * 计算 CKA 相似度矩阵 $S$。
   * 运行 DP 算法获取 `layer_map: {old_idx: new_idx}`。
4. **执行头映射与权重转换**：

   * 遍历 `layer_map` 中的每一对层。
   * 提取该层所有头的 $W_Q, W_K, W_V, W_O$。
   * 计算头交互矩阵并构建相似度矩阵。
   * 运行匈牙利算法获取 `head_map: {old_head_idx: new_head_idx}`。
   * 对于每个匹配的头：

     1. 计算 $\Delta W_o = B_o A_o$。
     2. 切片得到 $\Delta W_o^{head}$。
     3. 应用公式 (3) 计算 $\Delta W_n^{head}$。
     4. 对 $\Delta W_n^{head}$ 做 SVD，分解为新的 $A_n, B_n$。
   * 组装新的 LoRA 权重字典。
5. **轻量级微调**：

   * 将新得到的 $A_n, B_n$ 加载到 $M_n$ 的 LoRA 模块中。
   * 准备一个小规模指令微调数据集（~100-1000 条）。
   * 配置 Trainer：

     * `learning_rate`: $1e-3$ (建议网格搜索 $1e-4\sim1e-3$)
     * `warmup_ratio`: 0
     * `epochs`: 3-5
   * 开始训练。
6. **评估**：

   * 在下游任务上评估微调后的模型。

## 7. 局限性与未来工作

* **依赖基座权重**：需要访问新旧模型的完整参数，对于闭源模型不可用。
* **仍需微调**：目前无法完全摆脱微调步骤，未来研究方向是实现真正的“零样本”迁移。
* **隐式变化未处理**：仅处理了架构参数的显式变化，未考虑预训练数据分布变化或 RLHF 策略改变带来的隐式影响。
* **跨架构不支持**：目前仅支持相同架构家族内的升级（如 Llama 到 Llama），不支持跨架构（如 Llama 到 Qwen）。
