## qwen3_8B_cyber_actmap_v4_neg

**问题定义与起点**

我们从一个已经在 Qwen3-1.7B 上训练好的遗忘 LoRA 出发，目标是把这组 LoRA 参数迁移到 Qwen3-8B，并尽可能保持原始“功能效果”：

1. 在遗忘集上，准确率显著下降（表示遗忘生效）。
2. 在保留集上，准确率尽量稳定（表示通用能力不被破坏）。

对应模型规模差异为：

1. 层数：28 → 36。
2. hidden size：2048 → 4096。
3. FFN intermediate：6144 → 12288。
4. 注意力头数：16 → 32。
5. KV 头数：8 → 8（保持不变）。
   配置可见 config.json 与 config.json。

---

**迁移总流程（actmap v4）**

我们采用的是“无梯度直接映射”路线，核心实现在 transfer_lora_actmap.py。
流程可以分为五步。

### 1. 层映射：先解决 28 层到 36 层

我们先建立源层到目标层的固定映射表：

将目标模型的每一层，按照层号在总层数中的比例，映射到源模型中最接近的层。例如，源模型有28层，目标模型有36层，则目标第i层对应的源层为 **$i \times \frac{28}{36}$**，取最近的整数层号。这样能保证迁移时，目标模型的低层、中层、高层分别对应源模型的低层、中层、高层，结构分布尽量一致。未被直接命中的目标层（4, 8, 13, 17, 22, 26, 31, 35）使用“最近源层”补齐，见 transfer_lora_actmap.py 与 transfer_lora_actmap.py。

> 之前确实做过基于CKA（Centered Kernel Alignment）的层对齐实验。实验结果表明：
>
> * CKA映射在小模型或层间语义差异较大时，偶尔能带来一定提升，但在大模型、层数差异较大或跨架构迁移时，效果并不稳定，且有时还不如简单的比例映射。
> * 主要原因在于CKA需要依赖大量数据计算激活分布，且激活分布受预训练语料、模型结构等多因素影响，导致最优层对齐不易泛化。
> * 实验中，比例映射在大多数情况下迁移效果更稳定，泛化性更好，且实现和部署成本低。
>
> 因此，最终方案优先采用比例映射，CKA作为对比实验存在，未作为主力方案。

直觉上，这一步是在保证“深度语义相对位置”不乱序的前提下，先把层级语义对齐。

### 2. hidden 空间对齐：构造每层线性桥 R_l

对每个源层 i，我们预先计算一个激活对齐矩阵 R_l，满足：

```
$$h_{1.7B} R_l \approx h_{8B}$$
```

对应说明在 transfer_lora_actmap.py。这一步解决的是 2048 维 hidden 到 4096 维 hidden 的跨空间映射问题。

具体来说，hidden空间对齐（即R_l的计算）用的是“激活对齐法”，即直接采集源模型和目标模型在同一批通用文本（如数学题目）上的每一层hidden state（隐藏状态）输出，然后用最小二乘法（ridge regression）拟合出一个线性变换矩阵R_l，使得源模型该层的hidden经过R_l变换后，尽量接近目标模型同层的hidden。

具体流程如下：

1. 采集数据：用一批通用文本（非任务特定文本，在实验中我们采用了“数学题目”数据集（如ft-training_set/math_10k.json），以及部分通用常识文本（如sampled_100_commonsense_170k.json），早期实验曾尝试用任务相关数据（如cyber安全forget集），但发现这样得到的R_l会过拟合于特定任务方向，泛化性差。后续采用“通用文本+数学题目”作为采集数据，得到的R_l能更好地对齐模型的主流表示空间，迁移效果显著提升），分别跑源模型和目标模型，记录每一层的hidden state输出（通常是所有token的输出拼成的矩阵）。
2. 设 **$H_l^s$** 为源模型第l层hidden，**$H_l^t$**为目标模型同层hidden，形状分别为 **$N \times d_s$** 和 **$N \times d_t$**（N为样本数，**$d_s$**/**$d_t$**为hidden维度）。
3. 求解R_l：用最小二乘法（ridge regression）解 **$H_l^s R_l \approx H_l^t$**，即

   ```
   $$R_l = \arg\min_R \|H_l^s R - H_l^t\|_F^2 + \lambda \|R\|_F^2$$
   ```

   得到 **$R_l \in \mathbb{R}^{d_s \times d_t}$**。求解时走的是 lstsq 的 rcond=reg，reg 取 1e-3。
4. 该 **$R_l$** 即为hidden空间对齐用的线性变换，后续所有LoRA参数的hidden侧都用它做维度和分布对齐。

注意：

* 采集的hidden是“全体token在通用文本上的输出”，不是某个特定token或任务。
* 这种方法不区分head，也不区分不同模块，直接对整层hidden空间做全局线性对齐。
* 这样得到的R_l能最大程度捕捉到两个模型在实际推理时hidden空间的几何对应关系。
  > 我们确实做过“只用嵌入层”来算对齐矩阵的实验，且是早期主线之一：全局 W_x 直接由共享词表 embedding 最小二乘得到。结果是基本无效（forget 几乎等于 base，放大后还可能更差）。后面为什么没采用 embedding-only 方法的原因是：
  >
  > * 实验证据上无收益：embedding W_x 与 per-layer Q 权重法都接近“零遗忘迁移效果”。
  > * 方法假设过强：同一个全局线性变换用于所有层，但深层表示经过多次非线性后会与 embedding 空间脱耦。
  > * 尺度差太大：1.7B 到 8B 跨度大，embedding 层桥接对深层语义子空间不够准确。
  > * 所以后面改成逐层激活对齐 R_l（再配 P_l），即用运行时 hidden 几何来拟合，而不是只靠 embedding。
  >

### 3. FFN 中间维对齐：构造 P_l

FFN 维度从 6144 扩到 12288，单靠 **$R_l$** 不够，因此我们额外构造：

```
$$P_l = W_{g,8B} R_l^\top \operatorname{pinv}(W_{g,1.7B})$$
```

实现见 transfer_lora_actmap.py。这一步把“1.7B 的 FFN 中间神经元语义”投影到“8B 的更宽 FFN 空间”。

> 把旧模型的 FFN 中间激活，变成新模型 FFN 中间激活的“桥接矩阵”。公式是
>
> ```
> $$P_l = W_{g}^{t}\, R_l^\top \,(W_{g}^{s})^+$$
> ```
>
> 其中：
>
> * **$W_g^s$**：旧模型该层的 gate 投影（hidden **$\to$** intermediate）
> * **$W_g^t$**：新模型该层的 gate 投影
> * **$R_l$**：hidden 空间对齐矩阵（旧 hidden **$\to$** 新 hidden）
> * **$(W_g^s)^+$**：旧 gate 矩阵的伪逆（把 intermediate 近似拉回 hidden）
>
> 它到底在做什么（按步骤）：
>
> 1. 先“反解”旧 hidden
>    已知旧中间激活 **$v^s = W_g^s h^s$**，想拿到 **$h^s$**，只能近似反推：
>
> ```
> $$h^s \approx (W_g^s)^+ v^s$$
> ```
>
> 2. 再把旧 hidden 映射到新 hidden
>
> ```
> $$h^t \approx R_l^\top h^s$$
> ```
>
> 3. 再用新 gate 打到新 intermediate
>
> ```
> $$v^t \approx W_g^t h^t$$
> ```
>
> 把三步连起来：
>
> ```
> $$v^t \approx W_g^t R_l^\top (W_g^s)^+ v^s
> = P_l v^s$$
> ```
>
> 所以 **$P_l$** 的意义就是： 旧 intermediate 向量 **$v^s$** 经过 **$P_l$**，就近似得到新 intermediate 向量 **$v^t$**。
>
> 为什么要这样构造，而不是直接学一个矩阵？
>
> * 你已经有了 hidden 对齐的可靠信息 **$R_l$**
> * gate 权重 **$W_g^s, W_g^t$** 是模型本身结构
> * 这样推出来的 **$P_l$** 是“功能一致性”导出的解析解，不需要再额外采 intermediate 数据拟合
>
> 维度也对得上（1.7B **$\to$** 8B 例子）：
>
> * **$W_g^s: 6144\times2048$**
> * **$(W_g^s)^+: 2048\times6144$**
> * **$R_l^\top: 4096\times2048$**
> * **$W_g^t: 12288\times4096$**最终
>
> ```
> $$P_l: 12288\times6144$$
> ```
>
> 正好是 旧 intermediate **$\to$** 新 intermediate 的映射。

### 4. 模块级 LoRA 变换：A/B 如何变

设 LoRA 增量为 **$\Delta W = BA$**，则不同模块按维度类型分开变换：

1. hidden 维相关（如 q/o 的 A，q/o 的 B，以及 FFN 某些 A/B）
   * A 侧：**$A_{8B}=A_{1.7B}R_l$**
   * B 侧：**$B_{8B}=R_l^\top B_{1.7B}$**
2. FFN intermediate 维相关
   * **$B_{8B}=P_l B_{1.7B}$**
   * **$A_{8B}=A_{1.7B}P_l^\top$**
3. KV 分支特殊处理
   因为两侧 KV 头数都为 8，KV 输出维一致（1024），所以 k/v 的 lora_B 直接 copy。

> 1. 先变 DeltaW 再分解这条我们以前做过（Legacy-FFN 等），并且在这条 1.7B->8B unlearning 线上效果不如 ActMap。实验结论是现在也看到的：ActMap 这条更稳，Legacy 这类全局 W_x + DeltaW 逆链路线不是当前最优。
> 2. KV 特殊处理只特殊在 k/v 的 lora_B 直接拷贝；k/v 的 lora_A 仍然做 hidden 对齐变换。q/o 没有这个特殊分支，按常规 hidden 变换。即：
>    **k_proj**
>
> * lora_A：transform_A_hidden，也就是 A @ R_l
> * lora_B：copy_tensor，直接拷贝（因为 KV 输出维保持 1024 不变）
>   **v_proj**
> * lora_A：transform_A_hidden，也就是 A @ R_l
> * lora_B：copy_tensor，直接拷贝（同上）
>   **q_proj**
> * lora_A：A @ R_l
> * lora_B：R_l^T @ B
>   **o_proj**
> * lora_A：A @ R_l
> * lora_B：R_l^T @ B
>
> 这样设计的原因（结合结构）
>
> 1. A 作用在输入侧（rank x hidden_in），k/v 的输入 hidden 维是从 2048 到 4096，必须用 R_l 映射。
> 2. k/v 的 B 作用在输出侧（kv_out x rank），而两边 kv_out 都是 1024，所以可以直接 copy。
> 3. q/o 输出侧是 hidden 相关维度变化，B 需要 R_l^T 对齐，不能 copy。

### 5. per-module norm matching（v4 的关键）

只做线性映射会导致某些模块增量范数显著衰减，所以 v4 加了“逐模块范数回标”： 目标是让

```
$$\|B_{8B}A_{8B}\|_F \approx \frac{\alpha}{r}\|B_{1.7B}A_{1.7B}\|_F$$
```

实现位置在 transfer_lora_actmap.py。这一步是 v4 相比早期版本最关键的稳定化修正。

> Per-module norm matching 是一个**幅度补偿**步骤。简单来说，就是要修复一个严重的问题：经过 R_l 和 P_l 线性变换后，LoRA 的权重范数缩小了约 100 倍，如果不补偿，迁移后的 LoRA 对目标模型的影响会弱到几乎没有。
>
> **核心问题：为什么会缩小 100 倍？**
>
> 当你对 LoRA 的 A、B 分别乘以 R_l（维度大约从 2048→4096）时，组合起来 **$B \cdot A$** 的 norm 就严重下降了。具体见：
>
> ```
> $$\text{old: } \left\| \frac{\alpha}{r} B_{\text{1.7B}} A_{\text{1.7B}} \right\|_F \\
> \text{new: } \left\| B_{\text{8B}} A_{\text{8B}} \right\|_F \approx \text{old} / 100$$
> ```
>
> **补偿方法（代码 transfer_lora_actmap.py）：**
>
> 1. 处理完每个模块的 A 和 B 后，当你要写入 B 矩阵时，计算一个标量缩放因子：
>    ```
>    $$\text{rescale} = \frac{\left\| \frac{\alpha}{r} B_{\text{src}} A_{\text{src}} \right\|_F}{\left\| B_{\text{tgt}} A_{\text{tgt}} \right\|_F}$$
>    ```
> 2. 只对 B 进行缩放（不缩放 A）：
>    ```
>    $$B_{\text{tgt}} \leftarrow B_{\text{tgt}} \times \text{rescale}$$
>    ```
> 3. 对于未映射层（8 个没有对应源层的 8B 层），为了防止instability，rescale 上限是 20。
>
> 逻辑链：
>
> ```
> 未映射层 (4,8,13,17,22,26,31,35)
>   ↓
> 用 tgt_to_src(j) 找最近源层 (例如层4找到层3)
>   ↓
> 用源层3的R_l、P_l 对层4做变换
>   ↓
> 计算 rescale = ||B_src @ A_src|| / ||B_tgt @ A_tgt||
>   ↓
> rescale 可能很大（两层差异大）
>   ↓
> 【安全阈值】rescale > 20 就上限到 20，不再放大
> ```
>
> **为什么要限制？**
>
> 当 rescale 特别大时，说明：
>
> * 目标层和源层"差异太大"
> * 用源层的参数来近似目标层，本身就是粗糙的
> * 如果还大幅度放大这个粗糙近似，容易引入噪声或不稳定
>
> 例子：
>
> * 源层 norm = 10，目标层 norm = 0.1，rescale = 100
> * 但这 100 倍放大是建立在"近似"基础上的，可能把噪声放大得更离谱
> * 所以限制到 20 是个折中：有一定补偿，但不过度相信这个近似
>
> **为什么只缩放 B 而不缩放 A？**
>
> 缩放 B 等价于缩放整个 LoRA 更新的幅度，但不改变更新的方向。LoRA 的本质是 **$\Delta W = B A$**，范数只决定了施加多大的"力"，不决定方向是否对。只要遗忘方向被保留了（这由 A 的行向量方向决定），缩放 B 就只是调整影响力强弱，不伤害效果。如果同时缩放 A 和 B，反而可能因数值精度问题破坏方向信息。
>
> **实验效果：**
>
> README_Qwen_Unlearn_Transfer.md 里对比了有无 norm matching：
>
> * v4 without norm matching：norm 缩小 100 倍，Forget ≈ base 性能，相当于没起作用
> * v4 with norm matching：Forget 从 86.44% → 83.05%（降低 3.4%），代表真正的遗忘效果

---

**注意力头数变化是如何适配的**

我们的 actmap v4 里，头适配不是“显式头对头匹配”，而是“隐式适配 + 结构特判”：

1. 对 q/o 等 hidden 相关路径，头数变化（16→32）被吸收到 R_l 的整体 hidden 空间变换中。
2. 对 k/v 分支，由于 KV 头数 8→8 且维度一致，直接 copy 对应张量。
3. 因此 actmap v4 本质是单体矩阵映射（monolithic mapping），不是匈牙利匹配那类显式 head mapping。

这一点在方法学说明里有明确提示：actmap 脚本不做显式 head 映射，见 README_Transfer_Methodology.md。

> ## 头适配：隐式 vs 显式
>
> **关键问题：1.7B 有 16 个注意力头，8B 有 32 个注意力头，怎么对应？**
>
> ### 显式头对头匹配（我们 **没做** ）
>
> 如果显式匹配，会这样干：
>
> 1. 计算 1.7B 的每个头和 8B 的每个头的相似性（CKA、余弦相似度等）
> 2. 解匈牙利算法找最优配对（比如 1.7B 头0 → 8B 头5、1.7B 头1 → 8B 头12 等）
> 3. 对每对头分别做变换
>
> 这种做法的假设： **头是独立的功能单元，可以一一对应** 。
>
> ### 隐式适配（我们 **做的** ）
>
> actmap v4 的思路完全不同： **不单独处理头，而是在 hidden 层面做整体映射** 。
>
> 关键是理解 attention 的结构：
>
> ```
> 输入: h (batch, seq, hidden=2048)  [1.7B]
>   ↓
> Q, K, V = h @ W_q, h @ W_k, h @ W_v
>   (每个权重是 hidden × q_dim/k_dim/v_dim)
>   
> 1.7B: W_q: (2048, 1024), 分成 16 个头，每头 64 维
>       (1024 / 16 = 64)
>
> 8B:   W_q: (4096, 2048), 分成 32 个头，每头 64 维
>       (2048 / 32 = 64)
> ```
>
> **关键发现：**
>
> * Q、O 的输出空间（query embedding）是 hidden 相关维度（1.7B: 1024, 8B: 2048）
> * K、V 的输出空间是头空间（KV output，两边都是 1024）
>
> **我们的做法：**
>
> 1. **对 Q/O（hidden 相关）：**
>
>    * 不管 head 数怎么变（16→32），只关心 hidden 空间怎么变（2048→4096）
>    * 用 R_l 做整体 hidden 映射
>    * **头数变化被吸收到 hidden 维变化中**
>
>    ```
>    A_new = A_old @ R_l  # (rank × 2048) @ (2048 × 4096) = (rank × 4096)
>    B_new = R_l^T @ B_old
>    ```
>
>    新的 A 在 4096 维空间工作，自动适应了 32 个头的需求
> 2. **对 K/V（头空间）：**
>
>    * K/V 头数两边一样（8→8），KV 输出维也一样（1024）
>    * 直接 copy
>
>    ```
>    k_lora_B_new = k_lora_B_old.copy()  # 1024 维，不变
>    ```
>
> **为什么叫"隐式"？**
>
> 因为头匹配不是显式写出来的 mapping table，而是：
>
> * 通过 R_l（hidden 维映射）隐含地处理
> * 新模型的 32 个头在 4096 维空间中自动得到 LoRA 更新
> * 旧模型的 16 个头的方向信息，通过 R_l 被投射到新的 4096 维空间
>
> **为什么不用匈牙利算法？**
>
> 1. **太复杂** ：需要额外的相似度计算、配对算法
> 2. **假设不成立** ：attention 的头不是完全独立的功能单元，A/B 矩阵是跨头共享的（不是每个头单独一套 LoRA）
> 3. **高维映射已经够了** ：R_l 在 2048→4096 维空间做了充分的自由度，足以处理头数变化
>
> （有待进一步实验）

---

**qwen3_8B_cyber_actmap_v4_neg 是怎么来的**

先得到 qwen3_8B_cyber_actmap_v4（上述完整流程产物），再做方向敏感性试验：
对 v4 全部 lora_B 取负，得到 v4_neg 分支。

这一步的动机是检验“方向符号”对遗忘行为的影响（LoRA 低秩分解存在符号不唯一性，方向翻转可能改变功能）。
注意：v4_neg 更像“方向诊断分支起点”，不是最终迁移主结论本身。

---

## qwen3_8B_cyber_actmap_v4_neg_nlpow_AB_g1p5_amp3p0x_attnonly

### A.1 背景与目标

我们从适配器 qwen3_8B_cyber_actmap_v4_neg 出发，构造非梯度变体 qwen3_8B_cyber_actmap_v4_neg_nlpow_AB_g1p5_amp3p0x_attnonly。目标不是直接追求最终最优，而是先把“遗忘相关方向”的能量集中到更少、更强的通道上，为后续子空间重加权与因果规则重加权提供可控输入。

### A.2 对哪些参数做改动，哪些不改

该方案只改 LoRA 权重，不做反向传播训练。
对每个 LoRA 参数张量按模块类型分三类处理：

1. 注意力模块 q_proj, k_proj, v_proj, o_proj 的 lora_A：做 signed-power 变换并保范数。
2. 注意力模块 q_proj, k_proj, v_proj, o_proj 的 lora_B：先做同样的 signed-power 并保范数，再额外乘以 3.0 的幅度因子。
3. FFN 模块 up_proj, gate_proj, down_proj：完全不改，原样拷贝。

也就是说，attn 会被“重塑+放大”，FFN 保持冻结。

### A.3 signed-power 到底是什么

对于任意权重元素 **$x$**，signed-power 定义为：

```
$$\phi_{\gamma}(x)=\operatorname{sign}(x)\cdot |x|^{\gamma},\quad \gamma>1$$
```

本方案中 **$\gamma=1.5$**。
这一步会让大幅值元素变得更大，小幅值元素变得更小，从而提高分布峰度，达到“稀疏化有效方向”的效果。

但直接做幂变换会改变整体能量，因此我们加入范数回缩（norm restore）：

```
$$\tilde{W}=\phi_{\gamma}(W),\quad
W'=\tilde{W}\cdot \frac{\|W\|_F}{\|\tilde{W}\|_F+\epsilon}$$
```

这样可以把总量级拉回到原尺度，避免单纯数值放大带来的不稳定。

### A.4 A/B 两个矩阵为何区别处理

LoRA 增量写作：

```
$$\Delta W = BA$$
```

在本方案中：

1. 对 **$A$**：只做 signed-power+保范数，主要作用是重塑输入子空间的方向分布。
2. 对 **$B$**：做 signed-power+保范数后再乘 3.0，即

```
$$B' = 3.0\cdot \text{NormRestore}(\phi_{1.5}(B))$$
```

目的在于显式增大输出侧扰动幅度，使遗忘信号在前向中更“可见”。

直观上，**$A$** 控制“看什么方向”，**$B$** 控制“往外打多大”，所以对 **$B$** 再乘 3.0 是更直接的遗忘强化手段。

### A.5 为什么是 attn-only，不动 FFN

我们在快筛中观察到：FFN 同步放大通常更容易伤害 retain。因此此处采用“先控风险”的策略：优先在 attention 通道上增强遗忘信号，把 FFN 当作稳定锚点保留。这也是名字里 attnonly 的含义。

### A.6 参数选择理由（审稿人关心）

1. **$\gamma=1.5$**：比 2.0/3.0 更稳，不会过早进入数值崩坏区。
2. amp=3.0：在“有效强化”与“不过早塌缩”之间的折中点；更高幅度在快筛中出现明显退化。
3. attn-only：基于经验风险控制，优先减小对 retain 的副作用。

### A.7 方案A在整条方法链中的定位

方案A不是终点，而是“可操控初始化器”。后续两步（主子空间垂直分量放大 + 显式因果规则重加权）都依赖它提供更高信噪比的初始增量结构。

---

## ng_v2_perp20x_all（主子空间垂直分量放大）

### B.1 核心思想

把 LoRA 增量 **$\Delta W$** 分解为“与基座主子空间对齐部分”和“与基座主子空间正交部分”，压前者、放大后者。

### B.2 具体步骤

1. 恢复增量：**$\Delta W=BA$**。
2. 对基座权重 **$W_{\text{base}}$** 做截断 SVD，取前 **$k=1024$** 奇异向量，得到 **$U,V$**。
3. 投影分解：

```
$$\Delta W_{\parallel}=(UU^\top)\Delta W(VV^\top),\quad
\Delta W_{\perp}=\Delta W-\Delta W_{\parallel}$$
```

4. 重加权：

```
$$\Delta W'=\alpha\Delta W_{\parallel}+\beta\Delta W_{\perp}$$
```

其中 parallel_scale=0.5，perp_scale 对应 perp20x 设定。
5. 再把 **$\Delta W'$** SVD 截断回 LoRA 形式 **$(B',A')$**。

### B.3 为什么这样做

1. **$\Delta W_{\parallel}$** 与基座主方向重合，往往承载更多“通用能力”；过强会拖累 retain，因此乘 0.5。
2. **$\Delta W_{\perp}$** 更可能包含任务专属扰动（包括遗忘信号），因此放大。
3. 只在 q/o 执行且层范围为 all，是在“效果覆盖”与“副作用可控”间平衡后的工程选择。

---

## ng_v2_causal_L0off_L51016_x1p6 / x2p0（显式因果规则层模块重加权）

输入适配器：`ng_v2_perp20x_all`（方案一输出）

操作步骤：对指定层的指定模块的 `lora_B` 矩阵乘以标量 **$s$**（lora_A 不变）：

| 层  | 模块   | 标量**$s$** | 含义                                         |
| --- | ------ | ------------- | -------------------------------------------- |
| L0  | q_proj | 0             | 完全清零（禁用该层遗忘信号，解除防遗忘抑制） |
| L0  | o_proj | 0             | 同上                                         |
| L5  | q_proj | 1.6 或 2.0    | 放大遗忘主效应层                             |
| L5  | o_proj | 1.6 或 2.0    | 同上                                         |
| L10 | q_proj | 1.6 或 2.0    | 放大遗忘主效应层                             |
| L10 | o_proj | 1.6 或 2.0    | 同上                                         |
| L16 | q_proj | 1.6 或 2.0    | 放大遗忘主效应层                             |
| L16 | o_proj | 1.6 或 2.0    | 同上                                         |

因果假设来源（由梯度分析线和第 8.2 节早期实验确认）：

* **L0 q/o 是防遗忘层** ：L0 的 q/o LoRA 信号倾向于抑制遗忘效果；清零后遗忘信号可以更顺畅传播。
* **L5 / L10 / L16 q/o 是遗忘主效应层** ：梯度分析显示这几层的 q/o 权重更新方向与 cyber 知识遗忘最相关；放大其 lora_B 可加强遗忘。
* **不扩展到 k/v 和邻层** ：实验发现扩展到 k/v、增加 L1、放大邻近层均无额外收益，且 retain 会下降。

结果：Forget 64.41%，Retain 68.00%（remain_10pct 全量口径，x1.6 与 x2.0 完全并列）。

### C.1 核心假设

通过前序分析，我们把层-模块角色划分为：

1. L0 的 q/o：更像遗忘抑制位点。
2. L5/L10/L16 的 q/o：更像遗忘主效应位点。

### C.2 规则操作

在方案B输出的适配器上，对指定 lora_B 直接乘标量：

1. L0:q=0, o=0（关闭）。
2. L5/L10/L16:q,o 乘 **$s$**，其中 **$s\in\{1.6,2.0\}$**。
3. 其余层模块保持不变。

### C.3 为什么只改 lora_B

改 lora_B 等价于直接调输出侧增量强度，作用更“干净”、更可解释；同时避免同时改 A/B 导致耦合增多、可解释性下降。

### C.4 参数动机

1. 1.6 与 2.0 来自快筛得到的数值。
2. 继续增大到 2.2 无稳定收益。
3. 扩展到 k/v 或邻层会额外伤 retain，因此不采用。

---

结果为：

| 条目                                                        | 遗忘准确率 Forget       | 保留集准确率 Retain     |
| ----------------------------------------------------------- | ----------------------- | ----------------------- |
| qwen3_8B_cyber_actmap_v4_neg                                | 83.05% (49/59)          | 80.00% (80/100)         |
| qwen3_8B_cyber_actmap_v4_neg_nlpow_AB_g1p5_amp3p0x_attnonly | 67.80% (40/59)          | 70.00% (35/50)          |
| ng_v2_perp20x_all                                           | 66.10% (39/59)          | 66.00% (33/50)          |
| ng_v2_causal_L0off_L51016_x1p6                              | 64.41% (38/59)          | 68.00% (34/50)          |
| ng_v2_causal_L0off_L51016_x2p0                              | 64.41% (38/59)          | 68.00% (34/50)          |
| 微调后（ng_v2_x2p0_stage2mcq_try）                          | 72.88% (43/59)          | 74.00% (37/50)          |
| **1.7B base**                                         | **98.31%**        | **98.51%**        |
| **1.7B + LoRA**                                       | **27.12%**        | **86.22%**        |
| **8B base**                                           | **86.44%**(51/59) | **82.00%**(41/50) |

## **附：DRT方法线**

### **1. 研究动机**

#### **问题设定**

给定：

* 在小模型（1.7B）上预训练的遗忘 LoRA 适配器，已实现特定知识的有效移除
* 需要将该适配器迁移到大模型（8B），保持遗忘能力

 **核心挑战** ：

1. **维度不匹配** ：隐藏层维度 **$d_{1.7B}=2048 \to d_{8B}=4096$**（2倍升级）
2. **特征空间转换** ：旧模型的特征分布与新模型的表示空间不对齐
3. **遗忘知识转移** ：需要保留遗忘知识方向的语义，同时保护保留知识

 **常规方法的局限** ：

* **方法1（权重直接映射）** ：**$\Delta W_{new} = L\Delta W_{old}R$** 基于固定的权重矩阵 **$L, R$**，无法捕捉动态激活统计
* **方法2（从零训练）** ：计算昂贵，且初始化随机，容易陷入局部最优

---

### **2. 方法论**

#### **2.1 核心思想：从"增量"而非"权重"出发**

与传统方法直接迁移 LoRA 权重矩阵 **$\Delta W = BA$** 不同， **DRT 的关键洞察是** ：

**$\boxed{\text{迁移LoRA产生的"遗忘效应增量"}（\Delta h），而非权重本身}$**

 **直观理解** ：

* LoRA 在 1.7B 上的作用：**$h_{1.7B}^{new} = h_{1.7B}^{base} + \Delta h_{1.7B} = h_{1.7B}^{base} + \Delta W_{1.7B} x_{1.7B}$**
* 我们要在 8B 上**复现相同的遗忘方向** **$\Delta h$**（映射到新维度后）
* 同时 **保证在保留样本上无副作用** （不改变保留知识）

---

#### **2.2 数学形式化**

**第一步：提取遗忘方向**

在 1.7B 上，对遗忘样本 **$\mathcal{D}_f$** 的每个样本，提取遗忘 LoRA 的输出增量：

**$\Delta h_{old}^{(i)} = \Delta W_{1.7B} x_{1.7B}^{(i)} \quad \forall i \in \mathcal{D}_f$**

其中 **$x_{1.7B}^{(i)} \in \mathbb{R}^{d_{old}}$** 是 1.7B 在第 **$i$** 个遗忘样本的激活，**$\Delta W_{1.7B} \in \mathbb{R}^{d_{old} \times d_{old}}$** 是源 LoRA 的低秩增量。

**第二步：维度映射**

使用线性映射 **$M: \mathbb{R}^{d_{old}} \to \mathbb{R}^{d_{new}}$** 将遗忘方向映射到新维度：

**$\Delta h_{target}^{(i)} = M \Delta h_{old}^{(i)} \quad \forall i \in \mathcal{D}_f$**

映射 **$M$** 通过在保留样本上进行最小二乘拟合得到： **$M^* = \arg\min_M \sum_{i \in \mathcal{D}_r} \|h_{8B}^{base(i)} M - \Delta h_{target}^{(i)}\|_2^2 + \lambda \|M\|_F^2$**

**第三步：闭式求解（核心贡献）**

构造联合约束优化问题：

**$\min_{\Delta W_n} \left\{ \sum_{i \in \mathcal{D}_f} \|X_f^{(i)} \Delta W_n^T - \Delta h_{target}^{(i)}\|_2^2 + \lambda_r \sum_{j \in \mathcal{D}_r} \|X_r^{(j)} \Delta W_n^T\|_2^2 + \rho \|\Delta W_n\|_F^2 \right\}$**

其中：

* 第一项：在遗忘样本上**匹配目标遗忘方向**
* 第二项：在保留样本上 **强制近似零增量** （null-space 约束）
* 第三项：ridge 正则化，防止过拟合

 **改写成矩阵形式** ： 定义增广输入矩阵和目标：

**$X = \begin{bmatrix} X_f \\ \sqrt{\lambda_r} X_r \end{bmatrix} \in \mathbb{R}^{(N_f + N_r) \times d_{in}}, \quad Y = \begin{bmatrix} \Delta H_{target} \\ 0 \end{bmatrix} \in \mathbb{R}^{(N_f + N_r) \times d_{out}}$**

其中 **$N_f, N_r$** 分别为遗忘、保留样本数，**$\lambda_r$** 是保留约束权重。

最优解（via Tikhonov 正则化）：

**$\boxed{\Delta W_n^* = Y^T (\rho I + XX^T)^{-1} X}$**

 **计算技巧** （kernel trick）：

* 若 **$N_f + N_r \ll d_{in}$**，直接计算 **$K = XX^T$** （维度 **$(N_f+N_r) \times (N_f+N_r)$**）
* 然后求逆：**$(\rho I + K)^{-1}$**（耗时 **$O(N^3)$** 而非 **$O(d^3)$**）
* 最后投影：**$\Delta W_n = Y^T (\rho I + K)^{-1} X$**

 **优势** ：一次性解出，无需迭代，保证全局最优。

---

#### **2.3 低秩分解与稳定化**

对解 **$\Delta W_n$** 进行 SVD 分解：

**$\Delta W_n = U \Sigma V^T$**

 **截断到秩 **$r$**** （通常 **$r=32$**）：

**$\Delta W_n^{(r)} = U_{:,1:r} \Sigma_{1:r,1:r} V_{:,1:r}^T$**

然后分解为 LoRA 形式： **$\boxed{\Delta W_n^{(r)} = B_n A_n, \quad \text{其中} \quad B_n = U_{:,1:r}\sqrt{\Sigma_{1:r,1:r}}, \, A_n = \sqrt{\Sigma_{1:r,1:r}} V_{:,1:r}^T}$**

 **范数校准** ： 为避免迁移过程中的能量丧失（常见现象），对 **$B_n$** 进行缩放：

**$\text{scale} = \frac{\|\Delta W_{old}\|_F}{\|\Delta W_n^{(r)}\|_F}$**

**$B_n^{calibrated} = \text{scale} \cdot B_n$**

这保证了新 LoRA 在模型中的相对扰动强度与源 LoRA 一致。

---

### `drt_lr2_as16.0`

`drt_lr2_as16.0` 是怎么做的

这个实验是  **早期的 DRT 全层 attention 迁移基线** ，对应适配器： adapter_config.json

* 目标模型：Qwen3-8B
* LoRA rank：`r=32`
* LoRA alpha：`512`
* 目标模块：`q_proj / k_proj / v_proj / o_proj`
* 覆盖层：`0-35`，也就是 **全 36 层 attention**
* 不是 top8，而是 **全层 attention DRT**

这个名字通常可读成：

* `drt`：Delta-Retain Transfer，闭式解迁移
* `lr2`：这里是早期命名，指 **retain 约束强度较低的那档（lambda_retain=2 这组扫参）**
* `as16.0`：`alpha_scale=16.0`，最后得到 `lora_alpha=512`

它的做法可以概括成：

1. 在 1.7B 上提取旧 LoRA 对 forget 样本产生的增量方向 **$\Delta h_{old}$**
2. 在 8B 上收集 forget / retain 输入
3. 用 DRT 的闭式解直接求新模型上的 **$\Delta W$**
4. 做 SVD 截断回 LoRA
5. 应用到 **全层 q/k/v/o**

结果如下：

| 模型                        | WMDP forget              | WMDP-retain              | HSW-retain               |
| --------------------------- | ------------------------ | ------------------------ | ------------------------ |
| `qwen3_8B_drt_lr2_as16.0` | **22/59 = 37.29%** | **22/50 = 44.00%** | **42/50 = 84.00%** |

证据：

* WMDP forget + WMDP-retain 在eval_qwen3_8B_drt_lr2_as16.0_wmdpretain.json
* HSW-retain 在eval_drt_lr2_as16.0_hsw.json

这条结果的含义很明确：

* 它在  **forget 上很强** ，已经到了你记得的 “30%多”
* 但  **WMDP-retain 很差** ，只有 44%
* 对 HSW-retain 却不差，有 84%

所以它的问题不是“不会遗忘”，而是  **对 WMDP retain 集的分离性不好** 。这也是后来没有把它直接当最终主线的原因。

---

**3. 我们后面做微调用的 base adapter 是哪个**

后续 Stage2 微调**不是基于 **`drt_lr2_as16.0`。
你们真正拿来做 Stage2 的 base adapter 是：

adapter_config.json

报告里写得很明确，见：

* STAGE2_V12_DETAILED_REPORT.md
* STAGE2_V12_DETAILED_REPORT.md

这条 adapter 的含义是：

* `drt`：还是 DRT
* `as16`：alpha_scale=16
* `r5950`：forget=59, retain=50 这组探针规模
* `l02`：`lambda_retain=0.2`
* `lmapRw02`：用了  **retain-aware linear_map** ，retain weight = 0.2
* `ns0`：没启用显式 nullspace
* `top8`：只迁移 **最后 8 层 attention**

也就是说，**真正拿来做微调的不是早期全层基线 **`drt_lr2_as16.0`，而是后来的 DRT + linear_map + top8 迁移产物。

---

**4. 这个 base adapter 本身的起始结果**

这个 base adapter 在微调前的起始表现，报告给的是：

| base adapter                                      | WMDP forget              | WMDP-retain              |
| ------------------------------------------------- | ------------------------ | ------------------------ |
| `qwen3_8B_drt_as16_r5950_l02_lmapRw02_ns0_top8` | **43/59 = 72.88%** | **32/50 = 64.00%** |

证据在：

* STAGE2_V12_DETAILED_REPORT.md
* eval_top8_lora_only_wmdp_hardmatch.json

所以这条线的逻辑是：

* 它的 forget 没有 `drt_lr2_as16.0` 那么低
* 但它更适合继续做稳定化微调
* 于是它被选成了 Stage2 的初始化器

---

**5. 我们用了什么微调方法**

你们后续用的是  **Stage2 MCQ-targeted finetune** ，脚本是：

stage2_mcq_finetune.py

训练命令在：

STAGE2_V12_DETAILED_REPORT.md

核心方法不是普通 full-seq finetune，而是：

1. **只在 **`Answer:` 位置训练
   * 只优化 A/B/C/D 这个答案 token
   * 避免整段生成损失导致语言能力整体塌缩
2. **双目标损失**
   * Forget loss：降低 forget 样本正确答案 token 的概率
   * Retain loss：提高 retain 样本正确答案 token 的交叉熵正确性
3. **probe 选模**
   * 周期性在小验证集上看 forget/retain
   * 不是保存最后一步，而是保存 probe 最优 checkpoint
4. **retain floor early stop**
   * 如果 retain probe 连续低于阈值，就提前停
5. **forget_interval 控制**
   * 最优版本 v12 用的是 `forget_interval=2`
   * 即不是每步都施加 forget loss，而是隔步施加

从脚本实现看，这个机制对应：

* 只在最后答案位取 logits：stage2_mcq_finetune.py
* forget loss：压低 gold answer 概率
* retain loss：标准 CE
* `forget_interval`：stage2_mcq_finetune.py

一句话概括就是：

**用 DRT-top8 迁移产物做 warm start，再用 MCQ 定点损失做轻量稳定化微调。**

---

**6. 最终微调结果：三个数据集准确率**

你们最终主结果是 v12，对应适配器：

eval_stage2_mcq_full_v12_probe_lf900_lr160_fi2_t32_fixjudge.json 和 eval_stage2_mcq_full_v12_probe_lf900_lr160_fi2_hswretain_t32_fixjudge.json

最终三项结果如下：

| 最终微调模型 v12                                       | WMDP forget              | WMDP-retain              | HSW-retain               |
| ------------------------------------------------------ | ------------------------ | ------------------------ | ------------------------ |
| `qwen3_8B_stage2_mcq_full_v12_probe_lf900_lr160_fi2` | **33/59 = 55.93%** | **41/50 = 82.00%** | **43/50 = 86.00%** |

证据：

* WMDP forget + WMDP-retain：eval_stage2_mcq_full_v12_probe_lf900_lr160_fi2_t32_fixjudge.json
* HSW-retain：eval_stage2_mcq_full_v12_probe_lf900_lr160_fi2_hswretain_t32_fixjudge.json

---

| 路线                                                   | 作用                                  | 三项结果              |
| ------------------------------------------------------ | ------------------------------------- | --------------------- |
| `drt_lr2_as16.0`                                     | 早期 DRT 全层强遗忘基线               | 37.29 / 44.00 / 84.00 |
| `drt_as16_r5950_l02_lmapRw02_ns0_top8 -> Stage2 v12` | 最终主线：更稳的迁移初始化 + MCQ 微调 | 55.93 / 82.00 / 86.00 |

所以：

* 如果你只看 forget，`drt_lr2_as16.0` 更猛
* 如果你看最终可用折中，**v12 才是最终方案**

因为它把：

* WMDP retain 从 44% 或 64% 这种不稳定状态
* 拉到了 **82%**
* 同时 HSW retain 到 **86%**

代价是 forget 从 37.29% 回升到 55.93%。

**一、线性映射相关的方法到底有哪些**

严格按 delta_retain_transfer.py 的实现看，目标映射只有这两类：

| 类别           | 代码参数                      | 本质                                                                                                     |
| -------------- | ----------------------------- | -------------------------------------------------------------------------------------------------------- |
| `resize`     | `target_mapping=resize`     | 直接对**$\Delta h_{old}$**做输出维插值                                                                 |
| `linear_map` | `target_mapping=linear_map` | 先拟合 old output 到 new output 的线性算子**$M$**，再令**$\Delta h_{target}=M\Delta h_{old}$** |

其中：

1. **resize**

* 实现在 delta_retain_transfer.py
* 就是一维线性插值，把旧的输出维度拉伸到新的输出维度
* 优点是简单、稳、不需要再拟合映射矩阵
* 缺点是它只管“维度对齐”，不管“语义对齐”

2. **plain linear_map**

* 实现在 delta_retain_transfer.py
* 只用 forget 样本上的 old/new 模块输出拟合 Ridge 线性映射：

```
$$M = O_{new} O_{old}^{\top}(O_{old} O_{old}^{\top} + \lambda I)^{-1}$$
```

* 然后：

```
$$\Delta h_{target} = M \Delta h_{old}$$
```

* 代表实验就是：
  * eval_drt_as16_r4802_l02_lmap_top8_wmdpretain.json
  * eval_drt_as16_r4802_l02_lmap_top8_hswretain.json

3. **retain-aware linear_map**

* 还是同一个 `linear_map`，但打开了 `target_map_use_retain`
* 在拟合 **$M$** 时把 retain 输出也拼进去，并乘 **$\sqrt{w_r}$** 权重，见 delta_retain_transfer.py
* 也就是：
  * forget 输出参与拟合
  * retain 输出也参与拟合
  * 通过 `target_map_retain_weight` 控制 retain 对映射矩阵的约束强度
* 这就是命名里的 `lmapRw02 / lmapRw04 / lmapRw1` 这些

所以如果你问“线性映射分别有哪几种方法”，准确说法是：

1. **非学习式映射** ：`resize`
2. **学习式线性映射** ：`linear_map`
3. **学习式 retain-aware 线性映射** ：`linear_map + target_map_use_retain`

从结果命名上你们又做了这些实验分支：

* `lmap_top8`：plain linear_map + top8
* `lmap_ns8_top8` / `lmap_ns16_top8`：plain linear_map + nullspace
* `lmapRw01/02/03/04/05/06/1`：retain-aware linear_map，不同 retain weight
* `..._top8 / ..._L20_35 / ..._L24_35 / ..._all`：不同目标层范围
* `..._wmdpRetOnly`：映射/约束只偏向 WMDP retain 的特化版本

---

**二、为什么后来会演变成 **`drt_as16_r5950_l02_lmapRw02_ns0_top8`

这条名字不是随便长出来的，它是你们前面几轮失败和折中之后的“工程化稳定版”。

先说逻辑主线：

1. **最早的强遗忘基线太猛，但 retain 不行**

* 典型例子就是 eval_qwen3_8B_drt_lr2_as16.0_wmdpretain.json
* 结果：
  * Forget = 37.29%
  * WMDP-retain = 44.00%
  * HSW-retain = 84.00%
* 说明：它能忘，但**对 WMDP retain 伤得太重**

这逼着你们从“只追求 forget 更低”转向“要有可用的 retain 平衡”。

2. **单纯调 DRT 的 retain 约束后，出现了更可控的 Pareto 线**

* 例如：
  * eval_drt_as16_r3515_wmdpretain.json: 66.10 / 72.00
  * eval_drt_as16_r4010_l05_wmdpretain.json: 64.41 / 66.00
  * eval_drt_as16_r4505_l05_wmdpretain.json: 59.32 / 62.00
  * eval_drt_as16_r4802_l02_wmdpretain.json: 52.54 / 54.00
* 说明：随着 retain 约束增强，forget 继续下降，但 WMDP retain 还是不理想

于是问题不再只是 DRT 主方程的权重，而是： **旧模型的遗忘方向 **$\Delta h_{old}$** 映到 8B 输出空间的方式可能不对。**

3. **因此引入 linear_map，替代纯 resize**

* 你们意识到仅靠 resize 只是“尺寸对齐”，并不保证 old/new 输出语义对齐
* 所以用 old/new base outputs 拟合 **$M$**，先把旧增量映射到新空间，再做 DRT
* 这一步本质上是在补：**“旧 LoRA 的作用方向在新模型里应该落到哪里”**

4. **plain linear_map 虽然更聪明，但仍然不够 retain-aware**

* 代表点：
  * eval_drt_as16_r4802_l02_lmap_top8_wmdpretain.json: 55.93 / 50.00
  * eval_drt_as16_r4802_l02_lmap_top8_hswretain.json: 55.93 / 88.00
* 这个结果很有代表性：
  * HSW retain 很高
  * 但 WMDP-retain 仍然差
* 说明 plain linear_map 还是更像“forget 对齐器”，不是“forget/retain 兼顾对齐器”

5. **所以再往前一步，变成 retain-aware linear_map**

* 也就是 `lmapRwXX`
* 在拟合 **$M$** 时不只看 forget，还把 retain 输出一并纳入拟合
* `Rw` 就是 retain_weight
* 这一步的含义是：**你不只是想知道“旧的遗忘方向映到新空间哪里”，还想让这个映射不要破坏 retain 行为几何。**

6. **然后再做层范围收缩，最终走向 top8**

* 这是另一个关键变化
* 从全层 attention，收缩到最后 8 层 top8
* 报告中已经明确写了为什么，见 STAGE2_V12_DETAILED_REPORT.md
* 原因不是“top8 直接评测最强”，而是：
  * 参数更少
  * 风险更低
  * 稳定性更高
  * 更适合作为 Stage2 微调初始化器

也就是说，`drt_as16_r5950_l02_lmapRw02_ns0_top8` 不是“单轮直评最优”，而是“ **更适合当下一阶段微调起点** ”的那个点。

---

**三、把这个名字逐段讲清楚**

名字在 STAGE2_V12_DETAILED_REPORT.md 也被拆过，我这里用“为什么这么定”的方式重讲一遍。

### 1. `drt`

表示主框架仍然是 DRT。

也就是：

* 先得到 **$\Delta h_{target}$**
* 再用 forget / retain 输入做闭式解
* 最终求得 8B 上的 **$\Delta W$**
* 再 SVD 回 LoRA

不是梯度训练，不是从零学，而是 **闭式迁移初始化** 。

### 2. `as16`

表示 `alpha_scale = 16`，最后 LoRA alpha 变成 512。
这点在 adapter_config.json 里可验证。

为什么不是 as1 / as4 / as8？ 因为早期扫参里你们已经观察到强一些的 LoRA 幅度更容易把遗忘方向显性化；最终这条线上保留了更大的 alpha 设定。

### 3. `r5950`

表示 DRT 解算时用的探针规模是：

* forget = 59
* retain = 50

这个在 STAGE2_V12_DETAILED_REPORT.md 有说明。

为什么要显式写这个？ 因为 DRT 是闭式解，它对样本矩阵 **$X_f, X_r$** 非常敏感。
`r5950` 其实就是在强调：这不是小样本 smoke，而是完整 forget59 + retain50 的主解。

### 4. `l02`

表示 `lambda_retain = 0.2`。
也就是 retain null-space / 近零约束的主权重是 0.2。
见 STAGE2_V12_DETAILED_REPORT.md

为什么是 0.2？ 因为它是你们这条线上比较稳定的折中点：

* retain 权重太小，retain 保不住
* retain 权重太大，forget 会被压回去

`l02` 对应的是“够用但不过度”的 retain 约束。

### 5. `lmapRw02`

这是这条线最关键的演化结果。

它表示：

* `target_mapping = linear_map`
* `target_map_use_retain = True`
* `target_map_retain_weight = 0.2`

也就是在拟合 old->new 输出映射矩阵 **$M$** 时，retain 输出也参与，但权重只给 0.2。
见 STAGE2_V12_DETAILED_REPORT.md 和 delta_retain_transfer.py

为什么最后不是 `Rw04`、`Rw1`？ 因为更大的 retain-aware 映射权重会把映射过度往 retain 侧拉，容易损伤 forget 迁移信号。
你们后面确实试过：

* `Rw04`
* `Rw1`
* 甚至配不同层范围

但最终 Stage2 体系没有建立在这些点上，而是建立在 `Rw02 + top8` 这条更稳的初始化线上。

### 6. `ns0`

表示没有额外启用显式 retain nullspace 投影。
对应参数 `retain_nullspace_rank = 0`，见 delta_retain_transfer.py

为什么最后不用 `ns8` / `ns16`？ 因为 nullspace 投影虽然有时能抬高 retain，但也会让增量几何被压得过头，导致迁移效果更脆。
你们确实做过：

* `ns8`
* `ns16`

例如：

* eval_drt_as16_r4802_l02_ns8_wmdpretain.json
* eval_drt_as16_r4802_l02_ns16_wmdpretain.json

但在最终作为 Stage2 起点时，你们更偏向保留原始迁移几何，不再加这个额外投影，所以选了 `ns0`。

### 7. `top8`

表示只迁移 8B 的最后 8 层 attention，即 28-35 层。
这点在报告里说得很明确，见 STAGE2_V12_DETAILED_REPORT.md

为什么最后是 top8，而不是：

* `all`
* `L20_35`
* `L24_35`

因为你们后面真正目标不只是“迁移完直接评测”，而是“ **迁移后还能做稳定的小步微调** ”。

top8 的价值在于：

* 参数更少
* 局部修改更集中
* 对 base model 破坏更弱
* 更适合作为 Stage2 MCQ finetune 的 warm start

这也是为什么报告里最后说： 所有 v11/v12/v13/v14 都是从这个初始化器继续微调出来的，见 STAGE2_V12_DETAILED_REPORT.md

---

**四、为什么是这条，而不是其它同代变体**

如果只看迁移后直接评测，其实不止一条能打：

* eval_drt_as16_r5950_l02_lmapRw04_ns0_L24_35_wmdpretain.json: 62.71 / 66.00
* **eval_drt_as16_r5950_l02_lmapRw02_ns0_top8_wmdpretain.json: 77.97 / 72.00**

按直评看，`L24_35 + Rw04` 甚至更像“直接更强”的点。
但你们最后选 `top8 + Rw02 + ns0`，更像是在选一个 **微调初始化器** ，不是选一个 **单步终点** 。

所以它演化成这条主线，核心原因不是“它单次评测最好”，而是：

1. 它已经完成了从 resize 到 linear_map 的升级
2. 它已经把 retain 信息放进映射矩阵拟合里
3. 它把层范围压缩到 top8，降低了迁移噪声
4. 它没有再叠加 nullspace 这类额外刚性约束，保留了可微调性
5. 它最终被证明适合作为 Stage2 v12 的初始化器

最终这条线微调后给出：

* WMDP forget = 55.93%
* WMDP-retain = 82.00%
* HSW-retain = 86.00%

对应：

* eval_stage2_mcq_full_v12_probe_lf900_lr160_fi2_t32_fixjudge.json
* eval_stage2_mcq_full_v12_probe_lf900_lr160_fi2_hswretain_t32_fixjudge.json

---

**五、一句话总结**

你们这条线的演化其实就是：

**全层强遗忘但 retain 崩的 DRT 基线** → **只调 DRT 权重的 Pareto 扫参** → **发现 resize 不够，换成 linear_map** → **发现 plain linear_map 仍不够，改成 retain-aware linear_map** → **发现全层太重，收缩到 top8** → **为了保留可微调性，去掉额外 nullspace 投影** → **最终得到 **`drt_as16_r5950_l02_lmapRw02_ns0_top8` 作为 Stage2 初始化器
