# ActMap v4：基于激活对齐的跨架构 LoRA 遗忘迁移

## 摘要

本文提出 ActMap v4，一种基于激活空间对齐的跨架构 LoRA 遗忘能力迁移方法。给定已在源模型 $\mathcal{M}_S$ 上通过梯度上升-梯度下降（GA+GD）训练得到的遗忘 LoRA 适配器 $\Delta\theta_S$，我们的目标是在无需从头重新训练的前提下，将其迁移至架构同族但规模更大的目标模型 $\mathcal{M}_T$，使目标模型合并迁移后 LoRA 即可获得等效的遗忘能力。ActMap v4 通过逐层最小二乘对齐矩阵 $R_l$ 与解析导出的前馈中间维度对齐矩阵 $P_l$，将源 LoRA 参数线性投影至目标参数空间，再经由逐模块幅度恢复机制补偿因维度变换导致的幅度衰减。该方法仅需少量通用校准文本，无需任务数据或梯度优化，具有高效、轻量、可解释的优点。

---

## 1. 引言与问题定义

### 1.1 研究背景

大语言模型（Large Language Models, LLMs）的参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）技术，尤其是低秩适配（Low-Rank Adaptation, LoRA），已广泛用于模型编辑、知识卸载和遗忘学习等任务。然而，LoRA 适配器与基座模型的架构参数高度耦合——当基座模型升级（如同系列模型从 1.7B 扩展至 8B）时，由于隐藏维度、层数和注意力头数等架构参数的改变，已有 LoRA 适配器无法直接复用。传统的解决方案是"从头重新训练"，这在小规模实验时尚可接受，但在大模型场景下将导致高昂的计算开销和重复的数据标注成本。

### 1.2 问题形式化

令源模型 $\mathcal{M}_S$ 拥有 $L_S$ 层 Transformer，隐藏维度 $d_S$，注意力头数 $n_S$；目标模型 $\mathcal{M}_T$ 拥有 $L_T$ 层，隐藏维度 $d_T$，注意力头数 $n_T$。通常 $L_T > L_S$，$d_T > d_S$，$n_T \geq n_S$。在源模型上，我们已经通过 GA+GD 训练得到了遗忘 LoRA 适配器 $\Delta\theta_S$，其编码了从遗忘集 $\mathcal{D}_{\text{forget}}$ 中移除特定知识的能力。迁移的目标是构造 $\Delta\theta_T$，使得在 $\mathcal{M}_T$ 上应用 $\Delta\theta_T$ 后满足：

$$\text{Forget Accuracy}(\mathcal{M}_T \oplus \Delta\theta_T, \mathcal{D}_{\text{forget}}) \ll \text{Base Accuracy}(\mathcal{M}_T, \mathcal{D}_{\text{forget}})$$
$$\text{Retain Accuracy}(\mathcal{M}_T \oplus \Delta\theta_T, \mathcal{D}_{\text{retain}}) \approx \text{Base Accuracy}(\mathcal{M}_T, \mathcal{D}_{\text{retain}})$$

其中 $\mathcal{D}_{\text{retain}}$ 为保留集，用于评估通用能力的保持程度。此问题可形式化为一个跨架构的参数迁移学习任务：给定源参数空间 $\Theta_S$ 中的点 $\Delta\theta_S$，寻找映射 $\Phi: \Theta_S \to \Theta_T$，使得在目标参数空间 $\Theta_T$ 中的像 $\Phi(\Delta\theta_S)$ 具有与 $\Delta\theta_S$ 相近的功能效应。

### 1.3 核心挑战

跨架构 LoRA 迁移面临三重核心挑战。第一，**维度不匹配**：隐藏维度从 $d_S$ 到 $d_T$ 的扩展（如 2048→4096）导致 LoRA 的 $A$ 矩阵（$\mathbb{R}^{r \times d_{\text{in}}}$）和 $B$ 矩阵（$\mathbb{R}^{d_{\text{out}} \times r}$）无法直接复制。第二，**结构异构**：层数差异（如 28→36）和注意力头数差异（如 16→32）使得简单的比例映射或最近邻填充无法准确对齐语义功能对应的层级和子空间。第三，**功能耦合**：遗忘信号与保留能力的表示方向在隐藏空间中存在非线性耦合，单纯线性映射可能导致遗忘方向的不完全迁移或保留能力的非预期损失。

---

## 2. 源端遗忘：GA+GD LoRA 训练

### 2.1 训练目标

在源模型 $\mathcal{M}_S$ 上，我们采用梯度上升与梯度下降相结合的双目标训练范式来训练遗忘 LoRA。整体损失函数定义为：

$$\mathcal{L} = \mathcal{L}_{\text{forget}} + \alpha \cdot \mathcal{L}_{\text{retain}}$$

其中遗忘损失 $\mathcal{L}_{\text{forget}}$ 旨在最大化模型在遗忘集 $\mathcal{D}_{\text{forget}}$ 上的预测损失，从而"擦除"目标知识；保留损失 $\mathcal{L}_{\text{retain}}$ 则最小化模型在保留集 $\mathcal{D}_{\text{retain}}$ 上的预测损失，以维护模型的通用语言理解和推理能力。超参数 $\alpha$ 平衡两者之间的权衡关系。

具体而言，设 $\mathcal{M}_S(x; \theta_S)$ 为源模型在参数 $\theta_S$ 下对输入 $x$ 的负对数似然，则：

$$\mathcal{L}_{\text{forget}} = -\mathbb{E}_{x \sim \mathcal{D}_{\text{forget}}}[\log P_{\mathcal{M}_S}(x | \theta_S + \Delta\theta)]$$
$$\mathcal{L}_{\text{retain}} = \mathbb{E}_{x \sim \mathcal{D}_{\text{retain}}}[\log P_{\mathcal{M}_S}(x | \theta_S + \Delta\theta)]$$

### 2.2 LoRA 参数化

LoRA 将权重更新量 $\Delta W$ 参数化为两个低秩矩阵的乘积：$\Delta W = B A$，其中 $A \in \mathbb{R}^{r \times d_{\text{in}}}$，$B \in \mathbb{R}^{d_{\text{out}} \times r}$，$r \ll \min(d_{\text{in}}, d_{\text{out}})$。在前向推理时，输出计算为 $h = W_0 x + \frac{\alpha}{r} \cdot B A x$，其中 $\alpha$ 为缩放超参数。训练收敛后，$\Delta\theta_S = \{A_i, B_i\}_{i=1}^{L_S}$ 编码了针对遗忘集知识的"反向"表示信号——该信号在源模型的表示空间中能够有效地抑制遗忘相关知识的生成。

### 2.3 与迁移的关联

GA+GD 训练得到的 LoRA 适配器具有一个关键特性：它编码的是**表示空间中的方向信息**而非绝对值。这意味着 $\Delta\theta_S$ 的遗忘功能与源模型的表示几何密切相关——它告诉源模型在每个隐藏子空间中"往哪个方向推"。因此，迁移的核心挑战在于：如何在目标模型的表示空间中重建等价的"推的方向"，而保持其与目标模型自身表示几何的一致性。

---

## 3. 映射关系：层级与注意力头对齐

### 3.1 层级映射

层映射的目标是建立从源层索引 $i \in \{0, \dots, L_S-1\}$ 到目标层索引 $j \in \{0, \dots, L_T-1\}$ 的映射函数 $\phi: i \mapsto j$。我们考虑了三种映射策略。

**比例映射（Proportional Mapping）**是最简单的基线方案，假设两模型的层功能随深度按比例对齐：$j = \lfloor i \cdot L_T / L_S \rfloor$。对于 1.7B（28 层）→8B（36 层）的场景，该映射产生 28 个配对层，剩余 8 个未命中目标层采用最近邻补齐。该方案的假设前提是同系列模型各层的"相对深度位置"决定了其语义功能——浅层负责语法与词法编码，中层负责语义组合，深层负责高级推理。其缺陷在于忽略了层功能的非均匀分布特性，特别是在层数差异较大时，信息丢失显著。

**CKA 驱动的层映射**则基于更坚实的理论依据。我们计算线性中心核对齐（Linear Centered Kernel Alignment, CKA）来度量两模型任意层对之间的表示相似性：

$$\text{CKA}(H_S^{(i)}, H_T^{(j)}) = \frac{\|\tilde{H}_T^{(j)\top} \tilde{H}_S^{(i)}\|_F^2}{\|\tilde{H}_S^{(i)\top} \tilde{H}_S^{(i)}\|_F \cdot \|\tilde{H}_T^{(j)\top} \tilde{H}_T^{(j)}\|_F}$$

其中 $\tilde{H}$ 为列中心化后的激活矩阵，$H_S^{(i)} \in \mathbb{R}^{N \times d_S}$ 是源模型第 $i$ 层在 $N$ 个校准样本上的激活输出。CKA 具有对正交变换和等距变换的不变性，能够捕获表示空间的本质相似性。在 CKA 相似度矩阵 $S \in \mathbb{R}^{L_S \times L_T}$ 的基础上，我们实现了两种最优匹配方案：**单调 DP 匹配**（$\text{cka\_monotonic}$）在保持层序单调性的约束下通过动态规划最大化总 CKA 和，**匈牙利匹配**（$\text{cka\_hungarian}$）则不加单调约束，通过匈牙利算法求一对一全局最优匹配。前者适用于同系列模型（层功能单调递增），后者则适用于可能发生功能重排的跨架构场景。

需要指出的是，CKA 单调 DP 匹配仅产生 $L_S$ 个锚点映射（每源层一个），剩余的 $L_T - L_S$ 个目标层通过**就近填充**确定归属：未分配的目标层继承最近锚点的源层映射。这意味着一个源层的 LoRA 会被复制到 $1 + k$ 个目标层（一个锚点层 + $k$ 个填充层），产生重复映射。重复映射的幅度恢复行为见第 4.4 节。

### 3.2 注意力头映射

当源模型与目标模型的注意力头数不同时（例如 1.7B 的 16 头 vs 8B 的 32 头），需要建立头级别的对应关系。我们区分了两种方法。

**隐式头映射**是 ActMap v4 采用的策略：不单独处理各注意力头，而是通过隐藏维度对齐矩阵 $R_l$ 在整体 hidden 空间层面完成头数变化的自适应。其依据在于，Q 和 O 投影的输出空间归根到底是隐藏维度的线性组合——新模型的 32 个头在 4096 维空间中自动分配 16 头旧 LoRA 的方向信息。对于 K 和 V 投影，若两模型 KV 头数相同（如均为 8 头）且输出维度一致（1024），则 LoRA 的 B 矩阵可直接复制。

**显式头匹配**则通过计算各头的权重 CKA 相似度，再使用匈牙利算法求解最优一对一匹配。具体地，对每个注意力头 $h$，构建其 QK 交互矩阵 $W_{QK}^h = W_Q^h (W_K^h)^\top$ 和 VO 交互矩阵 $W_{VO}^h = W_V^h (W_O^h)^\top$，通过计算两模型对应交互矩阵间的余弦相似度来度量头功能相似性。该方法的优势在于能够捕捉注意力头的功能专门化（如 induction head、retrieval head 等跨模型的对应关系），不足之处在于计算开销较大且依赖于头权重的语义对齐假设。

---

## 4. ActMap v4：激活对齐线性重映射

### 4.1 隐藏维度对齐矩阵 $R_l$

ActMap 的核心思想是通过激活空间的对齐来桥接两模型的表示鸿沟。对每一对映射的层 $(l, l')$，我们利用校准文本分别采集源模型和目标模型的隐藏状态输出，然后通过岭回归（ridge regression）求解线性对齐矩阵：

$$R_l = \arg\min_R \|H_S^{(l)} R - H_T^{(l')}\|_F^2 + \lambda \|R\|_F^2, \quad R \in \mathbb{R}^{d_S \times d_T}$$

其中 $H_S^{(l)} \in \mathbb{R}^{N \times d_S}$ 和 $H_T^{(l')} \in \mathbb{R}^{N \times d_T}$ 分别为源模型第 $l$ 层和目标模型第 $l'$ 层在 $N$ 条通用文本上的激活矩阵（所有 token 的隐藏状态输出）。岭正则化项 $\lambda$（实验中取 $10^{-3}$）在样本数 $N \ll d_S$（通常 $N=64$, $d_S=2048$）条件下防止过拟合。

$R_l$ 的物理意义可以理解为两个模型在第 $l$ 深度处表示空间之间的"翻译器"：它将源模型的隐藏表示线性映射到目标模型的表示子空间中。需要特别注意，由于 $N \ll d_S$，$R_l$ 仅在激活数据所张成的 $N$ 维流形上有约束，在该流形的正交补方向上是任意的。这是 ActMap 方法的根本性局限，意味着 $R_l$ 在样本外方向上的行为缺乏保证。

### 4.2 前馈中间维度对齐矩阵 $P_l$

对于前馈网络（FFN），中间维度从 $d_{\text{ffn}}^S$ 到 $d_{\text{ffn}}^T$ 的扩展（如 6144→12288）无法仅通过 $R_l$ 完成对齐。我们提出了一种从模型权重解析导出中间对齐矩阵的方法，无需额外采集 FFN 中间层的激活数据。

设源模型 gate 投影权重为 $W_g^S \in \mathbb{R}^{d_{\text{ffn}}^S \times d_S}$，目标模型为 $W_g^T \in \mathbb{R}^{d_{\text{ffn}}^T \times d_T}$。若经过 hidden 对齐后输入 $x^T \approx R_l^\top x^S$，则期望的中间激活映射关系为：

$$v^T = W_g^T x^T \approx W_g^T R_l^\top x^S = \underbrace{(W_g^T R_l^\top (W_g^S)^+)}_{P_l} \cdot v^S$$

即 $P_l = W_g^T \cdot R_l^\top \cdot (W_g^S)^+ \in \mathbb{R}^{d_{\text{ffn}}^T \times d_{\text{ffn}}^S}$，其中 $(\cdot)^+$ 表示 Moore-Penrose 伪逆。该推导直接从"希望两模型的中间激活功能等价"出发，利用了模型本身的权重几何信息，当 $W_g^S$ 行满秩时精确成立，行亏秩时退化为最小范数解。

为防止 $P_l$ 的列条件数过大导致 LoRA 权重放大，我们施加列范数裁剪：计算 $P_l$ 各列的 Frobenius 范数，将超过中位数三倍的列等比例压缩至阈值以内。

### 4.3 模块级 LoRA 因子变换

基于上述对齐矩阵，我们对各模块的 LoRA 因子 $A$ 和 $B$ 分别实施变换。设源模型 LoRA 增量为 $\Delta W^S = B^S A^S$，对齐矩阵为输入空间变换 $R$ 和输出空间变换 $Q$，则目标增量为：

$$\Delta W^T = Q \cdot \Delta W^S \cdot R = \underbrace{(Q B^S)}_{B^T} \cdot \underbrace{(A^S R)}_{A^T}$$

对于每个模块，输入空间变换 $R$ 统一使用隐藏维度对齐矩阵 $R_l$（对 FFN down_proj 使用 $P_l$）。输出空间变换 $Q$ 则采用与 $P_l$ 一致的解析推导方式，从各模块自身的权重矩阵导出：

$$Q_{\text{module}} = W_{\text{module}}^T \cdot R_l^\top \cdot (W_{\text{module}}^S)^+$$

其中 $W_{\text{module}}^S$ 和 $W_{\text{module}}^T$ 分别为该模块在源模型和目标模型中的权重矩阵。这一推导的直觉与 $P_l$ 完全相同：它将源模型的输出表示经隐藏空间桥接后投影到目标模型的输出子空间中，使得迁移后的 LoRA 增量在目标模型的输出空间中保持功能等价。各模块的变换对应关系如下表所示：

| 模块 | 输入 $d_{\text{in}}^S \to d_{\text{in}}^T$ | 输出 $d_{\text{out}}^S \to d_{\text{out}}^T$ | $R$（输入空间） | $Q$（输出空间） |
|------|------|------|------|------|
| q_proj | 2048→4096 | 2048→4096 | $R_l$ | $W_q^T \, R_l^\top \, (W_q^S)^+$ |
| k_proj | 2048→4096 | 1024→1024 | $R_l$ | $W_k^T \, R_l^\top \, (W_k^S)^+$ |
| v_proj | 2048→4096 | 1024→1024 | $R_l$ | $W_v^T \, R_l^\top \, (W_v^S)^+$ |
| o_proj | 2048→4096 | 2048→4096 | $R_l$ | $W_o^T \, R_l^\top \, (W_o^S)^+$ |
| gate_proj | 2048→4096 | 6144→12288 | $R_l$ | $P_l$ |
| up_proj | 2048→4096 | 6144→12288 | $R_l$ | $P_l$ |
| down_proj | 6144→12288 | 2048→4096 | $P_l$ | $R_l^\top$ |

需要说明的是，Qwen3-1.7B 的 q_dim = 16 heads × 128 head_dim = 2048 = hidden_size，8B 的 q_dim = 32 × 128 = 4096 = hidden_size，因此 q_proj 和 o_proj 的输入输出空间均与隐藏空间同维，$R_l$ 可直接处理。K 和 V 的 KV 头数两边均为 8（head_dim=128），输出维度保持 1024 不变，但 $Q$ 并非恒等变换——通过权重导出的 $Q_k$ 和 $Q_v$ 在校正两个模型 KV 输出空间的基底旋转后，仍输出 1024 维，这比简单的直接复制更符合方法论的一致性。

对于 FFN 模块，gate_proj 和 up_proj 的输出空间使用前文推导的 $P_l$，down_proj 的输入空间也使用 $P_l$（维度对应关系与前馈中间维度一致），输出空间则使用 $R_l^\top$。

### 4.4 幅度恢复与稳定化

线性变换 $R_l$ 和 $P_l$ 的复合效应会导致 LoRA 增量范数的严重衰减（典型衰减幅度可达两个数量级）。若不加以补偿，迁移后的 LoRA 对目标模型的影响将微乎其微。为此，我们引入逐模块幅度恢复（per-module magnitude restoration）机制——注意这不是保持几何结构的保范变换，而是一个标量重缩放，仅恢复增量大小，不改变变换后的方向结构：

$$\text{rescale} = \frac{\|\frac{\alpha}{r} B^S A^S\|_F}{\|B^T A^T\|_F + \epsilon}$$

仅对 $B$ 矩阵施加缩放：$B^T \leftarrow B^T \times \text{rescale}$。选择仅缩放 $B$ 而非 $A$ 的原因在于：缩放 $B$ 等价于调整整个 LoRA 更新的幅度而不改变其方向，遗忘方向由 $A$ 的行向量方向决定，保留方向信息的同时调整影响力强弱。

对于层映射产生的重复映射情形——即同一个源层 $i$ 的 LoRA 被复制到多个目标层 $j_1, j_2, \dots, j_k$（第 3.1 节的锚点-填充机制或比例映射中的多对一场景）——rescaling 因子的计算方式有所不同。首个使用该源层映射的目标层（通常为锚点层 $j_1$）采用完整 rescaling 恢复，而后续重复映射的目标层（填充层 $j_2, \dots, j_k$）的 rescaling 因子被超参数 `norm_cap` 截断：

$$\text{rescale}_j = \begin{cases} \text{rescale} & j = j_1 \text{（锚点层，完整恢复）} \\ \min(\text{rescale}, \text{norm\_cap}) & j \in \{j_2, \dots, j_k\} \text{（填充层，截断）} \end{cases}$$

其中 `norm_cap` 的默认值为 20。截断机制的必要性在于：源 LoRA 的增量范数 $\|B^S A^S\|_F$ 是对单个层的遗忘信号强度进行校准的，若不加限制地将其复制到多个目标层，遗忘信号将被放大 $k$ 倍，可能导致模型输出崩溃。`norm_cap` 确保每个重复副本的幅度不超过安全阈值。

---

## 5. 结论

本文提出了 ActMap v4，一种基于激活对齐的跨架构 LoRA 遗忘迁移方法。该方法通过逐层岭回归求解隐藏维度对齐矩阵 $R_l$，并从模型权重解析导出前馈中间维度对齐矩阵 $P_l$，将源模型的遗忘 LoRA 参数线性投影至目标模型的参数空间。针对线性变换导致的幅度衰减，逐模块幅度恢复机制有效恢复了遗忘信号的强度。ActMap v4 仅需少量通用校准文本，无需任务数据或梯度优化，具有高效、轻量、可解释的优点。

**未来工作**包括：（1）将 CKA 层映射引入 ActMap 主路径以替代当前的比例映射；（2）在 $R_l$ 计算中使用更多样本和领域特异性数据以改善欠定条件下的对齐质量；（3）探索与行为迁移方法的混合方案，以进一步提升遗忘效果。
