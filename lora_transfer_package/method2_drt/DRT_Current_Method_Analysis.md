# DRT方法现状分析与限制

## 当前方法：Delta-Retain Transfer (DRT)

### 数学表述
```
minimize:   ||ΔW_forget||²_F
subject to: ||ΔW_retain||²_F ≤ λ_retain · threshold
           
where: ΔW = W_new - (W_old + α·AB^T)
```

### 实现细节
1. **数据收集**：从1.7B base和8B base收集激活向量
   - 遗忘集激活：A_forget ∈ ℝ^(59×d)
   - 保留集激活：A_retain ∈ ℝ^(50×d)

2. **目标优化**：逐层求解注意力模块的LoRA权重更新
   - Ridge正则化：ridge=1e-2（数值稳定）
   - Cholesky分解求解约束二次规划
   - SVD rank-32 LoRA输出

3. **参数化**
   - λ_retain ∈ {0.2, 0.5, 2.0} 控制遗忘-保留权衡
   - 固定alpha_scale=16.0，lora_r=32

### 实验结果

#### 消融链（完整对比）
| 配置 | 遗忘准确率 | HSW保留 | WMDP保留 |
|-----|---------|--------|--------|
| 1.7B Base | 100% | 100% | 100% |
| 1.7B+LoRA | **27.12%** | **78%** | **94%** |
| 8B Base | 86.44% | 86% | 82% |
| 8B+DRT (r4505_l05) | 59.32% | 82% | 62% |
| 8B+DRT (r4010_l05) | 64.41% | 82% | 66% |
| 8B+DRT (r4802_l02) | 52.54% | 90% | 54% |
| 8B+DRT (lmap_top8) | 55.93% | 90% | 50% |
| 8B+DRT (lmap+ns8_top8) | 55.93% | 84% | 50% |
| 8B+DRT (lmap+retain-aware+ns8_top8) | 84.75% | 82% | 84% |

#### 关键观察

**分离度对比**：
```
1.7B+LoRA:
  遗忘准确率 27.12% 
  WMDP保留  94.00%
  分离度: 遗忘 ⊕ 保留 (Δ66.88%) ✓ 良好分离

8B+DRT r4802_l02:
  遗忘准确率 52.54%
  WMDP保留  54.00%
  分离度: 遗忘 ≈ 保留 (Δ-1.46%) ✗ 无法分离
```

### 现有方法的局限

#### 问题1：保留约束不足（Retain Constraint Problem）
- 当前约束：`||ΔW_retain||² ≤ λ·threshold`
- **问题**：这只是限制激活变化的L2范数，没有保证遗忘方向的几何结构
- **现象**：遗忘可以通过改变"混合"方向来减少激活差异，而非纯粹的遗忘

#### 问题2：目标构建偏置（Target Construction Bias）
- 在线性映射目标 `linear_map` 中，如果只用 forget 样本拟合 `M`，则 `M` 学到的是“forget 局部对齐”。
- 当该 `M` 应用于 retain 分布时，容易发生外推偏移，表现为 retain 崩塌（WMDP retain 50%）。
- **根本原因**：目标迁移阶段缺少 retain 分布约束，不是简单的求解器问题。

#### 问题3：8B参数空间的信息压缩
- WMDP遗忘集 vs 保留集基线相似度：Δ只有4.44%
- 当DRT强制遗忘时，必然伤害保留（因为两个集合本来就相似）
- **但是**：我们在HSW上达到90%保留，说明遗忘方向*可以*被保护

### 核心洞察

**三个事实的矛盾**：
1. ✓ 8B Base能在遗忘集上达到86% → **找到遗忘方向**
2. ✓ DRT能将其降到52-64% → **成功激活遗忘方向**
3. ✗ 但WMDP保留从82%→54-66% → **遗忘方向与保留方向耦合**

**结论**：问题不是找不到遗忘方向，而是**找到的遗忘方向不是纯遗忘方向**，它在保留方向上有非零投影。

---

## 下一步改进方向

### 方向1：retain-aware 映射 + 零空间正交化
基本思想：在求解遗忘权重时，强制其垂直于保留方向

```
1. 计算保留方向: v_retain = mean(A_retain_8B) - mean(A_retain_1.7B)
2. 构造正交补空间: P_ortho = I - v_retain·v_retain^T / ||v_retain||²
3. 在零空间中优化: ΔW_forget' = P_ortho · ΔW_forget

并且在 `linear_map` 目标构建阶段联合拟合 forget+retain：

```
O_old = [O_old_forget, sqrt(w_r) O_old_retain]
O_new = [O_new_forget, sqrt(w_r) O_new_retain]
M = O_new O_old^T (O_old O_old^T + λI)^(-1)
ΔH_target_new = M ΔH_old
```
```

### 方向2：双目标优化（Bi-objective Optimization）
```
minimize: ||ΔW_forget||² + μ·(ΔW_forget · v_retain)²
subject to: ||ΔW_retain||² ≤ λ·threshold
```

### 方向3：子空间投影（Subspace Projection）
- 在低秩子空间中分别表达遗忘和保留
- 确保两个子空间的独立性
- 可能需要更大的LoRA秩（r>32）

---

## 理论依据（文献）

1. Learning without Forgetting (Li & Hoiem, 2016, arXiv:1606.09282)
- 核心启发：新任务更新时要显式保留旧任务行为约束。
- 对应到本项目：`linear_map` 不能只看 forget，需把 retain 分布纳入映射拟合。

2. Orthogonal Gradient Descent (Farajtabar et al., 2019, arXiv:1910.07104)
- 核心启发：通过正交/投影机制抑制对旧任务子空间的干扰。
- 对应到本项目：retain nullspace 投影是合理方向，但应与 retain-aware target 联动。

3. Similarity of Neural Network Representations Revisited (Kornblith et al., 2019, arXiv:1905.00414)
- 核心启发：表示相似性评估必须考虑样本数与表示维度关系；仅局部样本拟合的“相似映射”可能不稳。
- 对应到本项目：仅 forget 子集拟合 `M` 容易失真，跨分布泛化差。

4. SVCCA (Raghu et al., 2017, arXiv:1706.05806)
- 核心启发：跨网络表示比较本质上是子空间问题，不是逐坐标点对点问题。
- 对应到本项目：应优先做子空间一致性与分布覆盖，而非单一集合拟合。

## 新结论（2026-05-09）

- 先前 `lmap_top8` 与 `lmap+ns8_top8` 的失败，并不意味着 linear_map 思路无效；真正问题是“只用 forget 拟合映射”。
- 增加 retain-aware 拟合后：
   - WMDP retain 从 50% 提升到 84%
   - HSW retain 维持在 82%
   - 但 forget 回升到 84.75%（遗忘不足）
- 因此当前最准确的机制判断是：
   - 旧方案主要错在目标构建偏置（distribution shift）；
   - 新方案修复了 retain 崩塌，但过度保留导致遗忘变弱；
   - 下一步应在 retain-aware 映射上调低 retain 权重或分层分模块加权，寻找可证伪的 Pareto 前沿。

## 持续串行实验进展（2026-05-09，新增）

在 `target_mapping=linear_map + target_map_use_retain` 下，围绕三类控制量做串行扫描：
- retain 映射权重 `target_map_retain_weight`
- 作用层范围（top8 / L24-35 / L20-35 / all）
- retain nullspace 强度（ns0 / ns4 / ns8）

### WMDP retain 口径关键结果

| 配置 | Forget | WMDP Retain |
|---|---:|---:|
| top8, rw=1.0, ns8 | 84.75% | 84.00% |
| top8, rw=0.3, ns8 | 79.66% | 78.00% |
| top8, rw=0.2, ns0 | 77.97% | 72.00% |
| top8, rw=0.1, ns8 | 77.97% | 66.00% |
| L24-35, rw=0.4, ns0 | **62.71%** | 66.00% |
| L24-35, rw=0.4, ns4 | **62.71%** | 66.00% |
| L24-35, rw=0.5, ns0 | 76.27% | 68.00% |
| L24-35, rw=0.6, ns0 | 77.97% | 70.00% |
| L20-35, rw=0.2, ns0 | 66.10% | 54.00% |
| all, rw=0.2, ns0 | 69.49% | 48.00% |
| hybrid(top8 + L24-27注入) | 74.58% | 70.00% |

### HSW retain 口径补充

| 配置 | Forget | HSW Retain |
|---|---:|---:|
| top8, rw=0.2, ns0 | 77.97% | 82.00% |
| L24-35, rw=0.4, ns0 | 62.71% | 80.00% |
| hybrid(top8 + L24-27注入) | 74.58% | 80.00% |

### 新机制结论

1. **层覆盖范围是第一主导因子**：
- 从 top8 扩到中深层（如 L24-35 / L20-35）可显著降低 forget；
- 但 WMDP retain 同步明显下降，呈强 trade-off。

2. **retain 权重存在不稳定区间**：
- 在 L24-35 下，`rw` 从 0.4 到 0.6 出现非平滑跳变（forget 62.71% -> 77.97%）。
- 表明映射与闭式解在该子空间处于高敏区，需更细粒度扫描或分模块权重。

3. **当前最可用三类点**：
- 保留优先：top8, rw=0.2, ns0（77.97 / 72.00）
- 遗忘优先：L24-35, rw=0.4, ns0（62.71 / 66.00）
- 折中优先：hybrid(top8 + L24-27注入)（74.58 / 70.00），且 HSW retain=80.00%

4. **跨保留集诊断**：
- 低 forget 点在 HSW retain 仍较高（80%），但 WMDP retain 低（66%），说明不是全面退化，而是 retain 分布偏置问题仍在。

### 为什么零空间方法可行？

**数学基础**：
- 保留集的所有信息可以编码在保留激活方向上
- 存在垂直于此方向的子空间
- 在此子空间中的修改不会影响保留集

**实证支持**：
- 1.7B+LoRA已经达到了这种几何分离 (27% vs 94%)
- DRT在HSW上的成功(90%保留)证明了分离的可能性
- 问题仅在于WMDP内部分布相似导致的固有难度

### 为什么当前方法失败？

L2范数约束 → 激活空间约束
但激活空间中的"小变化"可能对应权重空间中的"混合方向修改"

需要的是：权重空间中的**方向约束**，而非幅度约束
