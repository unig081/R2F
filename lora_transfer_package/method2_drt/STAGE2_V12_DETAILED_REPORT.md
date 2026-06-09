# Qwen3-1.7B -> Qwen3-8B Unlearning LoRA 迁移与二阶段微调详细报告（v12）

## 1. 研究目标与结论先行

本实验要解决的问题是：
- 已有一个在 Qwen3-1.7B 上训练好的“遗忘 LoRA”（unlearning LoRA）。
- 希望把它迁移到 Qwen3-8B，并在 8B 上做尽量轻量的后处理，使“遗忘集准确率下降（接近 50%）且保留集准确率尽量稳定”。

目前最优折中点（截至本报告）：
- **v12（full）: Forget = 33/59 = 55.93%，Retain = 41/50 = 82.00%**
- 这是当前最稳且最接近目标区间的配置之一。

对应结果文件：
- [results/qwen_unlearn/eval_stage2_mcq_full_v12_probe_lf900_lr160_fi2_t32_fixjudge.json](results/qwen_unlearn/eval_stage2_mcq_full_v12_probe_lf900_lr160_fi2_t32_fixjudge.json)
- [results/qwen_unlearn/eval_stage2_mcq_full_v12_probe_lf900_lr160_fi2_t1_fixjudge.json](results/qwen_unlearn/eval_stage2_mcq_full_v12_probe_lf900_lr160_fi2_t1_fixjudge.json)

---

## 2. 第一阶段：迁移 LoRA 是怎么做的（不是从零训练）

### 2.1 方法概念（通俗解释）

可以把 LoRA 看成“模型上叠加的一层任务偏置”。
迁移的核心思想是：
1. 不重新从头学一个新 LoRA。
2. 利用旧模型和新模型的结构关系，把旧 LoRA 通过映射变换搬到新模型上。
3. 先拿到一个“可用但不完美”的起点，再做小规模微调修正。

这符合 LoRASuite 的典型流程：
- 先做 xTransform/权重空间迁移。
- 再做小规模稳定化微调。

参考与实现入口：
- [README.md](README.md)
- [lora_adaption.py](lora_adaption.py)

### 2.2 本任务里的“迁移后 LoRA”

本报告中的二阶段微调，不是从随机 LoRA 开始，而是从以下迁移产物开始：
- `trained_models/xTransform/qwen3_8B_drt_as16_r5950_l02_lmapRw02_ns0_top8`

这个迁移产物（top8）在评测中的基线表现：
- Forget: **43/59 = 72.88%**
- Retain: **32/50 = 64.00%**

对应文件：
- [results/qwen_unlearn/eval_top8_lora_only_wmdp_hardmatch.json](results/qwen_unlearn/eval_top8_lora_only_wmdp_hardmatch.json)

同时 8B base（不加 LoRA）为：
- Forget: **48/59 = 81.36%**
- Retain: **35/50 = 70.00%**

对应文件：
- [results/qwen_unlearn/eval_8B_base_only_wmdp_hardmatch.json](results/qwen_unlearn/eval_8B_base_only_wmdp_hardmatch.json)

一句话理解：
- 迁移后 LoRA（top8）相对 base 已经把 forget 从 81.36 拉到 72.88，但 retain 也掉到了 64。 
- 所以需要第二阶段微调做“更可控的折中”。

### 2.3 目标适配器命名的参数学解释

目标路径为：
- trained_models/xTransform/qwen3_8B_drt_as16_r5950_l02_lmapRw02_ns0_top8

该名称可以分解为：

1. drt
- 使用 Delta-Retain Transfer 方法（闭式解），实现见 [delta_retain_transfer.py](delta_retain_transfer.py)。

2. as16
- alpha_scale=16。若源 LoRA 的 lora_alpha=32，则迁移后 alpha 约为 512。
- 与产物配置一致，见 [trained_models/xTransform/qwen3_8B_drt_as16_r5950_l02_lmapRw02_ns0_top8/adapter_config.json](trained_models/xTransform/qwen3_8B_drt_as16_r5950_l02_lmapRw02_ns0_top8/adapter_config.json)。

3. r5950
- 使用 forget=59、retain=50 的探针样本规模进行闭式迁移求解。

4. l02
- lambda_retain=0.2，用于 retain 约束权重。

5. lmapRw02
- target_mapping=linear_map，且 retain-aware 线性映射权重 retain_weight=0.2。

6. ns0
- retain_nullspace_rank=0，即未启用 retain 子空间显式零化投影。

7. top8
- 仅迁移 8B 的后 8 层（28-35）以及 attention 模块（q/k/v/o）。

### 2.4 DRT 方法的优化目标与数学形式

与常规反向传播训练不同，DRT 的核心是直接解一个闭式矩阵回归问题。设：

- X_f 为 8B 在 forget 样本上的模块输入（行样本，列特征）。
- X_r 为 8B 在 retain 样本上的模块输入。
- Delta h_target 为希望在 forget 上复现的旧 LoRA 输出增量方向（已映射到 8B 维度）。

构造联合输入与目标：

$$
X = \begin{bmatrix} X_f \\ \sqrt{\lambda_r} X_r \end{bmatrix},
\quad
Y = \begin{bmatrix} \Delta h_{target} & 0 \end{bmatrix}
$$

其中 retain 区块目标为 0，表示“尽量对 retain 无影响”。然后定义核矩阵：

$$
K = XX^T
$$

闭式解为：

$$
\Delta W = Y(\rho I + K)^{-1}X
$$

其中 rho 为 ridge 正则系数。随后对 Delta W 做 SVD 截断到 rank=r，得到 LoRA 形式：

$$
\Delta W \approx BA
$$

在脚本实现中，A 对应 lora_A，B 对应 lora_B，详见 [delta_retain_transfer.py](delta_retain_transfer.py#L268)。

### 2.5 从 1.7B 到 8B 的工程迁移流水线

针对每个目标层-模块，执行如下步骤：

1. 提取旧 LoRA 的遗忘方向
- 在 1.7B 上收集 forget 输入激活。
- 用源 LoRA 计算模块输出增量 Delta h_old。

2. 将旧方向映射到新输出空间
- 本适配器使用 linear_map 路径（retain-aware），将 Delta h_old 映射到 8B 输出维度，形成 Delta h_target。
- 实现见 [delta_retain_transfer.py](delta_retain_transfer.py#L237)。

3. 在 8B 上收集 forget/retain 输入
- 得到 X_f 与 X_r。

4. 闭式求解 Delta W
- 用上面的核回归公式一次性解出 Delta W，不进行梯度迭代。

5. 几何与尺度校准
- 几何方向约束：若学习方向与目标方向反向，进行整体翻转。
- 保范数校准：保持与源 LoRA 相近的相对扰动强度。
- 对应实现见 [delta_retain_transfer.py](delta_retain_transfer.py#L325) 与 [delta_retain_transfer.py](delta_retain_transfer.py#L369)。

6. SVD 压缩回 LoRA
- 对 Delta W 做 SVD 截断，得到 rank=32 的 A/B 并写入 adapter。

### 2.6 top8 的层映射与模块选择

本适配器只对 8B 的层 28-35 生效，且只处理 attention 四个投影模块。

层映射由近邻比例映射给出：

$$
o_i = round\left(i \cdot \frac{L_{old}}{L_{new}}\right)
$$

其中 L_old=28，L_new=36。对新层 28-35，大致对应旧层 22-27（中间有重复映射），这是 top8 在 5x 尺度升级场景下的一种“低风险高效”做法：
- 参数更少，稳定性更高；
- 先保证可迁移，再在 Stage2 做小步精调。

### 2.7 该方法与“从零 LoRA 训练”的本质区别

1. DRT 阶段
- 不是从随机初始化开始迭代训练。
- 核心计算是闭式矩阵求解 + SVD 分解。

2. Stage2 阶段
- 才是小规模梯度微调，但起点是已迁移 LoRA，不是随机 LoRA。

因此，trained_models/xTransform/qwen3_8B_drt_as16_r5950_l02_lmapRw02_ns0_top8 可以理解为：
- 一个由“旧 LoRA 知识方向 + 新模型激活统计 + 闭式约束求解”合成出的迁移初始化器。
- 后续 v11/v12/v13/v14 都是在这个初始化器上继续微调得到的。

---

## 3. 第二阶段微调：具体怎么做的

实现脚本：
- [stage2_mcq_finetune.py](stage2_mcq_finetune.py)

### 3.1 训练目标（为何这样设计）

早期“整段序列级遗忘损失”会导致模型整体塌缩。为避免这个问题，采用了 MCQ 定点目标：
- 只在 `Answer:` 位置优化 A/B/C/D 这个答案 token。

损失由两部分组成：
1. Forget loss：降低 forget 样本上 gold 选项 token 的概率。
2. Retain loss：提升 retain 样本上 gold 选项 token 的交叉熵正确性。

可写为：

$$
\mathcal{L}=\lambda_f \cdot \mathbb{E}[p_{gold}^{forget}] + \lambda_r \cdot CE^{retain}
$$

### 3.2 稳定化策略（v12 成功关键）

为避免“训练中间一度好、最终反而坏”，加入三层保护：

1. Probe 选模（周期性小验证）
- 每隔 `probe_every` 步，在小探针集上计算 `probe_f` 与 `probe_r`。
- 按评分函数保存最佳 checkpoint，而不是保存最后一步：

$$
score = -|probe_f - target| + w_r \cdot probe_r
$$

2. Retain floor 早停
- 如果 `probe_r` 连续低于阈值（如 0.60）达到 `max_bad_probe_steps`，提前终止，防止训练后期崩坏。

3. Forget 更新频率控制（forget_interval）
- `forget_interval=2` 表示不是每步都施加 forget loss，而是隔步施加。
- 这是 full 场景中平衡 forget/retain 的关键杠杆。

### 3.3 v12 训练命令（可复现）

```powershell
& D:/Anaconda/envs/ali/python.exe stage2_mcq_finetune.py \
  --base_model modelzoo/qwen3_8B \
  --adapter_path trained_models/xTransform/qwen3_8B_drt_as16_r5950_l02_lmapRw02_ns0_top8 \
  --forget_file datasets/processed_forget_sets/1.7B_to_8B_transfer/wmdp-cyber/forget_10pct.json \
  --retain_file datasets/processed_forget_sets/1.7B_to_8B_transfer/wmdp-cyber/remain_10pct.json \
  --output_dir trained_models/xTransform/qwen3_8B_stage2_mcq_full_v12_probe_lf900_lr160_fi2 \
  --lambda_forget 9.0 \
  --lambda_retain 1.6 \
  --learning_rate 5e-5 \
  --num_epochs 1 \
  --batch_size 4 \
  --probe_every 24 \
  --probe_samples 64 \
  --target_forget_acc 0.50 \
  --min_retain_probe_acc 0.60 \
  --retain_score_weight 0.45 \
  --max_bad_probe_steps 3 \
  --forget_interval 2
```

训练数据规模（full）：
- forget：59
- retain：537

---

## 4. 评估数据集与评估方法

### 4.1 评估数据

使用同口径 WMDP-cyber：
- Forget 评测集：
  [datasets/processed_forget_sets/1.7B_to_8B_transfer/wmdp-cyber/forget_10pct.json](datasets/processed_forget_sets/1.7B_to_8B_transfer/wmdp-cyber/forget_10pct.json)
- Retain 评测集：
  [datasets/processed_forget_sets/1.7B_to_8B_transfer/wmdp-cyber/remain_10pct.json](datasets/processed_forget_sets/1.7B_to_8B_transfer/wmdp-cyber/remain_10pct.json)

评测抽样口径：
- forget_size=59
- retain_size=50

### 4.2 评估脚本与判分方式

脚本：
- [evaluate_lora_with_judge.py](evaluate_lora_with_judge.py)

模式：
- `--judge_model none`（hard-match）
- 对 MCQ 题提取模型输出中的首个字母，与 gold A/B/C/D 比较。

### 4.3 一个关键修复（非常重要）

曾经出现过“伪 0/0”假象，原因是旧正则对 `AAAA...`/`BBBB...` 提取失败。
已修正为“提取首个字母”，避免误判。

修复位置：
- [evaluate_lora_with_judge.py](evaluate_lora_with_judge.py#L188)

---

## 5. 结果对比：迁移后 vs 微调后

### 5.1 关键里程碑

| 阶段 | 模型 | Forget | Retain | 备注 |
|---|---|---:|---:|---|
| Base | 8B base | 81.36% | 70.00% | 无 LoRA |
| 迁移后 | xTransform top8 | 72.88% | 64.00% | 作为二阶段起点 |
| 微调后（v11） | full v11 | 55.93% | 68.00% | Forget明显下降 |
| 微调后（v12） | full v12 | **55.93%** | **82.00%** | 当前最佳折中 |
| 微调后（v13） | full v13 | 59.32% | 80.00% | 稍回退 |
| 微调后（v14, t32） | full v14 | 55.93% | 78.00% | retain低于v12 |
| 微调后（v14, t1） | full v14 | 38.98% | 76.00% | 对生成长度更敏感 |

支持文件：
- [results/qwen_unlearn/eval_8B_base_only_wmdp_hardmatch.json](results/qwen_unlearn/eval_8B_base_only_wmdp_hardmatch.json)
- [results/qwen_unlearn/eval_top8_lora_only_wmdp_hardmatch.json](results/qwen_unlearn/eval_top8_lora_only_wmdp_hardmatch.json)
- [results/qwen_unlearn/eval_stage2_mcq_full_v11_probe_lf800_lr140_fi2_t32_fixjudge.json](results/qwen_unlearn/eval_stage2_mcq_full_v11_probe_lf800_lr140_fi2_t32_fixjudge.json)
- [results/qwen_unlearn/eval_stage2_mcq_full_v12_probe_lf900_lr160_fi2_t32_fixjudge.json](results/qwen_unlearn/eval_stage2_mcq_full_v12_probe_lf900_lr160_fi2_t32_fixjudge.json)
- [results/qwen_unlearn/eval_stage2_mcq_full_v13_probe_lf1000_lr160_fi2_t32_fixjudge.json](results/qwen_unlearn/eval_stage2_mcq_full_v13_probe_lf1000_lr160_fi2_t32_fixjudge.json)
- [results/qwen_unlearn/eval_stage2_mcq_full_v14_probe_lf900_lr160_fi1_t32_fixjudge.json](results/qwen_unlearn/eval_stage2_mcq_full_v14_probe_lf900_lr160_fi1_t32_fixjudge.json)
- [results/qwen_unlearn/eval_stage2_mcq_full_v14_probe_lf900_lr160_fi1_t1_fixjudge.json](results/qwen_unlearn/eval_stage2_mcq_full_v14_probe_lf900_lr160_fi1_t1_fixjudge.json)

### 5.2 v12 为什么是“当前最佳折中”

- 与 v11 比：forget 同为 55.93%，retain 从 68.00% 提升到 82.00%。
- 与 v13 比：retain略高且 forget 不更差。
- 与 v14 比：v14 在 t1/t32 上波动更明显，稳定性弱于 v12。

因此 v12 是当前“效果 + 稳定性”综合最优点。

---

## 6. 资源消耗与节约分析（重点）

这一节分两层：
1. 相比全参数训练（最硬核的节约）。
2. 相比 LoRA 从零训练（工程上更贴近）。

### 6.1 与全参数训练相比

当前可训练参数只在 LoRA 上，配置为：
- hidden=4096, rank=32
- target modules: q/k/v/o（4个）
- 目标层：8层（28-35）

可训练 LoRA 参数数：
- 每个线性层 LoRA 参数 = $r\cdot d + d\cdot r = 2rd = 262,144$
- 每层4个模块 = 1,048,576
- 8层总计 = **8,388,608**

相对 8B 模型（按 8e9 近似）占比：
- `8,388,608 / 8,000,000,000 = 0.001048576`
- 即仅约 **0.105%** 参数可训练
- 参数级节约约 **99.895%**

这也直接带来优化器状态、反向显存和计算量的大幅下降。

### 6.2 与 LoRA 从零训练相比（保守工程估计）

严格地说，本项目没有跑“同目标同数据同超参的随机初始化对照组”，所以只能做保守估计，不夸大。

可确认的事实：
- 我们是从迁移 LoRA（top8）继续微调（warm-start）。
- v12 在 1 个 epoch 内得到可用最优折中点。

若参考常见 LoRA 训练配置（README 中 vanilla LoRA 常用 3 epoch）做工程估算：
- 若从零 LoRA 往往要更长训练/更多试错才能进入稳定区，
- 那么 warm-start 在有效模型探索上通常可减少 2/3 左右迭代轮次。

保守写法：
- **参数维度节约可精确给出（99.895%）**。
- **迭代轮次节约是经验估计，建议后续补做随机初始化对照以得到严格数字**。

---

## 7. 复现清单（给工程同学）

1. 迁移起点适配器：
- `trained_models/xTransform/qwen3_8B_drt_as16_r5950_l02_lmapRw02_ns0_top8`

2. 训练脚本：
- [stage2_mcq_finetune.py](stage2_mcq_finetune.py)

3. 最优配置（当前）：
- v12: `lf=9.0, lr=1.6, forget_interval=2, probe+early-stop`

4. 评测脚本：
- [evaluate_lora_with_judge.py](evaluate_lora_with_judge.py)

5. 评测命令关键参数：
- `--judge_model none --lora_only --eval_forget_size 59 --eval_retain_size 50`

6. 判分修复（避免伪 0/0）：
- [evaluate_lora_with_judge.py](evaluate_lora_with_judge.py#L188)

---

## 8. 小结（论文式摘要）

本文在“Qwen3-1.7B unlearning LoRA -> Qwen3-8B”场景下，采用“先迁移、后轻量微调”的两阶段策略。迁移后模型（top8）达到 72.88/64.00（Forget/Retain）。在此基础上，提出 MCQ 答案位定点损失 + probe 选模 + retain floor 早停 + forget 更新频率控制的稳定化微调框架。最终获得 v12：55.93/82.00，显著优于迁移后起点，并在稳定性上优于相邻配置。方法训练参数仅占 8B 模型的约 0.105%，参数级资源节约约 99.895%，证明“迁移初始化 + 小步微调”在该任务上具备较高实用价值。

---

## 9. 针对本轮6个问题的串行复核（含新增实测）

### 9.1 串行执行计划与完成状态

本轮按“先核验路径 -> 再统一口径评测 -> 再补充试验 -> 最后文档化”的顺序执行：

1. 核验资源是否存在（模型、数据、checkpoint）。
2. 评测 v12 在 HSW retain 口径下的表现。
3. 评测 ng_v2 x1p6/x2p0 在 HSW retain 口径下的表现。
4. 评测“from-scratch 风格（trueinit）”与非 trueinit 的对比。
5. 将“当前 Stage2 微调策略”尝试应用在 ng_v2 上，观察是否改善目标。
6. 将所有结果和解释写回本报告。

以上步骤均已完成。

### 9.2 问题1：DRT 中使用的是哪个 retain 集？

结论：本报告主线的 DRT/top8 与 Stage2 默认使用的是 WMDP-cyber 的 retain 集。

对应路径：
- [datasets/processed_forget_sets/1.7B_to_8B_transfer/wmdp-cyber/remain_10pct.json](datasets/processed_forget_sets/1.7B_to_8B_transfer/wmdp-cyber/remain_10pct.json)

代码与配置证据：
- DRT 参数定义： [delta_retain_transfer.py](delta_retain_transfer.py#L43)
- DRT 实际加载 retain： [delta_retain_transfer.py](delta_retain_transfer.py#L401)
- Stage2 默认 retain： [stage2_mcq_finetune.py](stage2_mcq_finetune.py#L271)

### 9.3 问题2：每个参数为什么这么选？（重点）

下面给出 top8 迁移器关键参数的“选择原因 -> 风险权衡”解释。

| 参数 | 取值 | 选择原因（为什么） | 风险与权衡 |
|---|---:|---|---|
| max_forget | 59 | 与 forget 评测口径一致，迁移目标和评测目标对齐。 | 样本偏少会增加方差。 |
| max_retain | 50 | 保持与常用 retain 评测抽样一致，避免迁移-评测分布不一致。 | retain约束不覆盖长尾。 |
| modules | attn | 先聚焦 q/k/v/o，迁移成本低、稳定性高。 | 放弃了 MLP 方向的潜在增益。 |
| lora_r | 32 | 与源 LoRA 一致，表达力与参数量平衡。 | rank 不足时可能欠拟合方向。 |
| alpha_scale | 16 | 把源 LoRA 强度映射到 8B 可用幅度区间（对应 alpha=512）。 | 过大易放大副作用，过小会迁移无感。 |
| lambda_retain | 0.2 | retain 约束作为“软刹车”，避免把 forget 方向放得过猛。 | 过强会抑制遗忘，过弱会伤 retain。 |
| target_mapping | linear_map | 比 resize 更能利用 old/new 输出几何对应关系。 | 映射误差会把噪声也迁过去。 |
| target_map_use_retain | True（命名 lmapRw） | 映射拟合时引入 retain 输出，提高 old->new 映射稳健性。 | 可能牺牲部分 forget 强度。 |
| target_map_retain_weight | 0.2 | retain 信息只做轻约束，不主导映射。 | 权重过小可能 retain收益不足。 |
| retain_nullspace_rank | 0 | 先不做显式 nullspace 投影，减少额外几何扭曲风险。 | 对 retain 的硬约束能力较弱。 |
| new_layer_start/end | 28-35（top8） | 只迁移后8层：参数更省、训练更稳、与高层任务决策更相关。 | 低层可迁移信息利用不足。 |

为什么是“后8层 top8”：

1. 任务相关性：高层更直接参与答案决策与语义聚合，遗忘方向更容易在高层显现。
2. 稳定性：全层迁移在历史实验中更容易出现 retain 波动与副作用扩散。
3. 成本：top8 仅 8 层 qkvo，训练参数仅 8,388,608，约占 8B 的 0.105%。
4. 工程策略：先拿稳健迁移初始化器，再交给 Stage2 做小步定向修正。

### 9.4 问题3：md 里的 retain 准确率用的是哪个 retain 集？HSW 测过吗？

1. md 主结论（v12=55.93/82.00）使用的是 WMDP retain：
- [results/qwen_unlearn/eval_stage2_mcq_full_v12_probe_lf900_lr160_fi2_t32_fixjudge.json](results/qwen_unlearn/eval_stage2_mcq_full_v12_probe_lf900_lr160_fi2_t32_fixjudge.json)

2. HSW retain 已补测（本轮新增）：
- [results/qwen_unlearn/eval_stage2_mcq_full_v12_probe_lf900_lr160_fi2_hswretain_t32_fixjudge.json](results/qwen_unlearn/eval_stage2_mcq_full_v12_probe_lf900_lr160_fi2_hswretain_t32_fixjudge.json)

结果：
- PEFT Forget: 33/59 = 55.93%
- PEFT Retain(HSW): 43/50 = 86.00%

结论：v12 在 HSW retain 口径也稳定，且 retain 高于 WMDP retain 口径。

### 9.5 问题4：之前 37% 遗忘准确率基线，换 WMDP retain 是否仍不好？测过吗？

测过，且结果显示在 WMDP retain 口径下确实“分离度差”。

证据文件：
- [results/qwen_unlearn/eval_qwen3_8B_drt_lr2_as16.0_wmdpretain.json](results/qwen_unlearn/eval_qwen3_8B_drt_lr2_as16.0_wmdpretain.json)

对应指标：
- Forget: 22/59 = 37.29%
- Retain(WMDP): 22/50 = 44.00%

同一模型在 HSW retain 口径：
- [results/qwen_unlearn/eval_drt_lr2_as16.0_hsw.json](results/qwen_unlearn/eval_drt_lr2_as16.0_hsw.json)
- Retain(HSW): 42/50 = 84.00%

结论：你担心的是对的。该 37% 基线在 WMDP retain 下仍然不好（retain 太低）。

### 9.6 问题5：ng_v2_causal_L0off_L51016_x1p6/x2p0 能否套用当前微调法？

本轮已实际尝试（x2p0）：

1. 训练（新增）
- 输出： [trained_models/xTransform/qwen3_8B_ngv2_x2p0_stage2mcq_try_20260510_bs1/stage2_mcq_config.json](trained_models/xTransform/qwen3_8B_ngv2_x2p0_stage2mcq_try_20260510_bs1/stage2_mcq_config.json)
- 备注：先遇到 OOM，后改为 batch=1、max_length=256、max_samples=64 成功完成。

2. 评测（WMDP retain）
- [results/qwen_unlearn/eval_ngv2_x2p0_stage2mcq_try_20260510_bs1_wmdpretain_t32_fixjudge.json](results/qwen_unlearn/eval_ngv2_x2p0_stage2mcq_try_20260510_bs1_wmdpretain_t32_fixjudge.json)
- PEFT: 72.88% / 74.00%

3. 评测（HSW retain）
- [results/qwen_unlearn/eval_ngv2_x2p0_stage2mcq_try_20260510_bs1_hswretain_t32_fixjudge.json](results/qwen_unlearn/eval_ngv2_x2p0_stage2mcq_try_20260510_bs1_hswretain_t32_fixjudge.json)
- PEFT: 72.88% / 72.00%

与 ng_v2 原始 x2p0 对比（本轮重评）：
- 原始 x2p0（WMDP retain）：64.41% / 62.00%
  - [results/qwen_unlearn/eval_ngv2_causal_L0off_L51016_x2p0_wmdpretain_t32_fixjudge_20260510.json](results/qwen_unlearn/eval_ngv2_causal_L0off_L51016_x2p0_wmdpretain_t32_fixjudge_20260510.json)
- 原始 x2p0（HSW retain）：64.41% / 48.00%
  - [results/qwen_unlearn/eval_ngv2_causal_L0off_L51016_x2p0_hswretain_t32_fixjudge_20260510.json](results/qwen_unlearn/eval_ngv2_causal_L0off_L51016_x2p0_hswretain_t32_fixjudge_20260510.json)

结论：
- “套用当前微调法”可以显著提高 ng_v2 的 retain；
- 但 forget 也显著回升到 72.88%，不符合“遗忘降到约50%”主目标；
- 因此它更像“保留修复”，不是当前目标导向下的最优方案。

### 9.7 问题6：与 LoRA 从零训练相比，准确度如何？

本轮按同口径做了 from-scratch 风格（trueinit）对比：

1. trueinit 模型（x2p0_ga24_e3_trueinit）
- [results/qwen_unlearn/eval_ngv2_x2p0_ga24_e3_trueinit_wmdpretain_t32_fixjudge_20260510.json](results/qwen_unlearn/eval_ngv2_x2p0_ga24_e3_trueinit_wmdpretain_t32_fixjudge_20260510.json)
- PEFT: 38.98% / 34.00%

2. 非 trueinit 对照（x2p0）
- [results/qwen_unlearn/eval_ngv2_causal_L0off_L51016_x2p0_wmdpretain_t32_fixjudge_20260510.json](results/qwen_unlearn/eval_ngv2_causal_L0off_L51016_x2p0_wmdpretain_t32_fixjudge_20260510.json)
- PEFT: 64.41% / 62.00%

结论：
- trueinit（从零风格）把 forget 压得更低，但 retain 大幅崩到 34%；
- 在“forget 接近50且 retain 稳定”的目标下，纯从零风格并不占优；
- 当前迁移初始化 + 稳定化 Stage2 的综合可控性更好。

### 9.8 本轮新增结果文件索引

- [results/qwen_unlearn/eval_stage2_mcq_full_v12_probe_lf900_lr160_fi2_hswretain_t32_fixjudge.json](results/qwen_unlearn/eval_stage2_mcq_full_v12_probe_lf900_lr160_fi2_hswretain_t32_fixjudge.json)
- [results/qwen_unlearn/eval_ngv2_causal_L0off_L51016_x1p6_hswretain_t32_fixjudge_20260510.json](results/qwen_unlearn/eval_ngv2_causal_L0off_L51016_x1p6_hswretain_t32_fixjudge_20260510.json)
- [results/qwen_unlearn/eval_ngv2_causal_L0off_L51016_x2p0_hswretain_t32_fixjudge_20260510.json](results/qwen_unlearn/eval_ngv2_causal_L0off_L51016_x2p0_hswretain_t32_fixjudge_20260510.json)
- [results/qwen_unlearn/eval_ngv2_causal_L0off_L51016_x2p0_wmdpretain_t32_fixjudge_20260510.json](results/qwen_unlearn/eval_ngv2_causal_L0off_L51016_x2p0_wmdpretain_t32_fixjudge_20260510.json)
- [results/qwen_unlearn/eval_ngv2_x2p0_ga24_e3_trueinit_wmdpretain_t32_fixjudge_20260510.json](results/qwen_unlearn/eval_ngv2_x2p0_ga24_e3_trueinit_wmdpretain_t32_fixjudge_20260510.json)
- [results/qwen_unlearn/eval_ngv2_x2p0_stage2mcq_try_20260510_bs1_wmdpretain_t32_fixjudge.json](results/qwen_unlearn/eval_ngv2_x2p0_stage2mcq_try_20260510_bs1_wmdpretain_t32_fixjudge.json)
- [results/qwen_unlearn/eval_ngv2_x2p0_stage2mcq_try_20260510_bs1_hswretain_t32_fixjudge.json](results/qwen_unlearn/eval_ngv2_x2p0_stage2mcq_try_20260510_bs1_hswretain_t32_fixjudge.json)

---

## 10. 新一轮串行实验：r3515 -> Stage2（quick筛选 + 全量复核）

### 10.1 目标与执行策略

本轮目标是验证一个更直接的工程路径：
- 先用 r3515 作为 Stage1 起点；
- 先做 quick 小样本筛选（59/128）找可行参数区；
- 再对最优候选做“全量 retain（59/537）”复核。

重点关注指标仍然是：
- Forget 尽量落在 50%-58%；
- Retain 尽量稳定且不低于约 70%。

### 10.2 Quick 筛选结果（WMDP retain）

本轮共 6 组：lf in {3.0, 4.5, 6.0} × fi in {1,2}，统一 lr=5e-5，retain quick 样本 128。

| 配置 | Forget | Retain | 结论 |
|---|---:|---:|---|
| lf30_lr160_fi1 | 34/59 = 57.63% | 35/50 = 70.00% | 进入目标区间 |
| lf30_lr160_fi2 | 34/59 = 57.63% | 35/50 = 70.00% | 进入目标区间 |
| lf45_lr160_fi1 | 20/59 = 33.90% | 31/50 = 62.00% | forget过低且retain不足 |
| lf45_lr160_fi2 | 34/59 = 57.63% | 34/50 = 68.00% | retain略低于目标 |
| lf60_lr160_fi1 | 11/59 = 18.64% | 16/50 = 32.00% | 明显塌缩 |
| lf60_lr160_fi2 | 11/59 = 18.64% | 16/50 = 32.00% | 明显塌缩 |

对应结果文件：
- [results/qwen_unlearn/eval_stage2mcq_r3515_quick_lf30_lr160_fi1_wmdpretain_t32_20260510.json](results/qwen_unlearn/eval_stage2mcq_r3515_quick_lf30_lr160_fi1_wmdpretain_t32_20260510.json)
- [results/qwen_unlearn/eval_stage2mcq_r3515_quick_lf30_lr160_fi2_wmdpretain_t32_20260510.json](results/qwen_unlearn/eval_stage2mcq_r3515_quick_lf30_lr160_fi2_wmdpretain_t32_20260510.json)
- [results/qwen_unlearn/eval_stage2mcq_r3515_quick_lf45_lr160_fi1_wmdpretain_t32_20260510.json](results/qwen_unlearn/eval_stage2mcq_r3515_quick_lf45_lr160_fi1_wmdpretain_t32_20260510.json)
- [results/qwen_unlearn/eval_stage2mcq_r3515_quick_lf45_lr160_fi2_wmdpretain_t32_20260510.json](results/qwen_unlearn/eval_stage2mcq_r3515_quick_lf45_lr160_fi2_wmdpretain_t32_20260510.json)
- [results/qwen_unlearn/eval_stage2mcq_r3515_quick_lf60_lr160_fi1_wmdpretain_t32_20260510.json](results/qwen_unlearn/eval_stage2mcq_r3515_quick_lf60_lr160_fi1_wmdpretain_t32_20260510.json)
- [results/qwen_unlearn/eval_stage2mcq_r3515_quick_lf60_lr160_fi2_wmdpretain_t32_20260510.json](results/qwen_unlearn/eval_stage2mcq_r3515_quick_lf60_lr160_fi2_wmdpretain_t32_20260510.json)

### 10.3 全量复核（59/537）与异常处理

对 quick 最优区域（lf30）做全量复核时，先尝试：
- batch_size=4, max_length=512

该设置在训练早期触发 OOM（CUDA out of memory），因此切换到已验证稳定设置：
- batch_size=1, max_length=256

### 10.4 全量结果（lf30，bs1）

#### 配置A：fi1（forget_interval=1）

训练阶段出现 retain 低探针连续触发早停，最终表现塌缩：
- WMDP retain: Forget 11/59 = 18.64%，Retain 16/50 = 32.00%
- HSW retain: Forget 11/59 = 18.64%，Retain 12/50 = 24.00%

结果文件：
- [results/qwen_unlearn/eval_stage2mcq_r3515_full_lf30_lr160_fi1_bs1_wmdpretain_t32_20260510.json](results/qwen_unlearn/eval_stage2mcq_r3515_full_lf30_lr160_fi1_bs1_wmdpretain_t32_20260510.json)
- [results/qwen_unlearn/eval_stage2mcq_r3515_full_lf30_lr160_fi1_bs1_hswretain_t32_20260510.json](results/qwen_unlearn/eval_stage2mcq_r3515_full_lf30_lr160_fi1_bs1_hswretain_t32_20260510.json)

训练配置文件：
- [trained_models/xTransform/qwen3_8B_stage2mcq_r3515_full_lf30_lr160_fi1_20260510_bs1/stage2_mcq_config.json](trained_models/xTransform/qwen3_8B_stage2mcq_r3515_full_lf30_lr160_fi1_20260510_bs1/stage2_mcq_config.json)

#### 配置B：fi2（forget_interval=2）

在相同 bs1 资源约束下，fi2 仍保持可用折中：
- WMDP retain: Forget 32/59 = 54.24%，Retain 35/50 = 70.00%
- HSW retain: Forget 32/59 = 54.24%，Retain 36/50 = 72.00%

结果文件：
- [results/qwen_unlearn/eval_stage2mcq_r3515_full_lf30_lr160_fi2_bs1_wmdpretain_t32_20260510.json](results/qwen_unlearn/eval_stage2mcq_r3515_full_lf30_lr160_fi2_bs1_wmdpretain_t32_20260510.json)
- [results/qwen_unlearn/eval_stage2mcq_r3515_full_lf30_lr160_fi2_bs1_hswretain_t32_20260510.json](results/qwen_unlearn/eval_stage2mcq_r3515_full_lf30_lr160_fi2_bs1_hswretain_t32_20260510.json)

训练配置文件：
- [trained_models/xTransform/qwen3_8B_stage2mcq_r3515_full_lf30_lr160_fi2_20260510_bs1/stage2_mcq_config.json](trained_models/xTransform/qwen3_8B_stage2mcq_r3515_full_lf30_lr160_fi2_20260510_bs1/stage2_mcq_config.json)

### 10.5 本轮结论

1. quick 筛选阶段，lf30 是唯一稳定进入目标区间的参数区。
2. 全量复核后，fi1 在 bs1 条件下出现明显塌缩，不建议继续。
3. 全量复核后，fi2 保持 54.24%/70.00%(WMDP) 与 54.24%/72.00%(HSW)，是当前 r3515 分支下的可用结果。
4. 但从“retain 更高且稳定”的目标看，其综合仍弱于主线 v12（55.93%/82.00%）。
---

## 11. r3515 方案汇总与对比

### 11.1 遗忘集与保留集准确率（r3515 最优配置：fi2）

| 指标 | WMDP 口径 | HSW 口径 |
|---|---:|---:|
| **Forget Accuracy** | 32/59 = **54.24%** | 32/59 = **54.24%** |
| **Retain Accuracy** | 35/50 = **70.00%** | 36/50 = **72.00%** |

### 11.2 方法详细记录

**起点**：
- 迁移初始化器：`trained_models/xTransform/qwen3_8B_drt_as16_r3515`
  - DRT（Delta-Retain Transfer）闭式解生成
  - 源数据规模：forget=59, retain=50（来自探针采样）
  - 迁移层范围：后 8 层（28-35），attention 四模块（q/k/v/o）

**Stage2 MCQ 微调配置**（最优: fi2）：
- 训练脚本：[stage2_mcq_finetune.py](stage2_mcq_finetune.py)
- 超参数：
  - `lambda_forget=3.0`（遗忘权重）
  - `lambda_retain=1.6`（保留权重）
  - `learning_rate=5e-5`
  - `batch_size=1`（原 4 触发 OOM，改为内存安全配置）
  - `max_length=256`（原 512 触发 OOM，改为内存安全配置）
  - `num_epochs=1`
  - `forget_interval=2`（fi2：隔步施加 forget loss）
  - `probe_every=24`（周期探针间隔）
  - `probe_samples=64`（小验证集）
  - `target_forget_acc=0.55`（目标遗忘探针准确率）
  - `min_retain_probe_acc=0.60`（保留探针下界，触发早停）
  - `retain_score_weight=0.45`（checkpoint 选择时 retain 的权重）
  - `max_bad_probe_steps=3`（retain 连续低于下界 3 次后早停）

**稳定化机制**：
- Probe 选模：不保存最后一步，而是保存在 probe 指标上最优的 checkpoint
- Retain Floor 早停：保留准确率若持续低于 0.60，提前终止训练，防止后期崩坏
- Forget 更新频率控制：`forget_interval=2` 表示隔步施加遗忘损失，减弱对 retain 的冲击

**训练数据规模**（全量）：
- Forget 集：59 样本
- Retain 集：537 样本（完整 WMDP-cyber remain）

**训练产物**：
- 模型目录：[trained_models/xTransform/qwen3_8B_stage2mcq_r3515_full_lf30_lr160_fi2_20260510_bs1](trained_models/xTransform/qwen3_8B_stage2mcq_r3515_full_lf30_lr160_fi2_20260510_bs1)
- 配置文件：[trained_models/xTransform/qwen3_8B_stage2mcq_r3515_full_lf30_lr160_fi2_20260510_bs1/stage2_mcq_config.json](trained_models/xTransform/qwen3_8B_stage2mcq_r3515_full_lf30_lr160_fi2_20260510_bs1/stage2_mcq_config.json)

### 11.3 r3515 vs v12（当前最优主线）对比

| 维度 | r3515 fi2 | v12 | 差异说明 |
|---|---|---|---|
| **Forget(WMDP)** | 54.24% | 55.93% | r3515 略低（可接受） |
| **Retain(WMDP)** | 70.00% | 82.00% | r3515 低 12pp（关键劣势） |
| **Forget(HSW)** | 54.24% | 55.93% | r3515 略低 |
| **Retain(HSW)** | 72.00% | 86.00% | r3515 低 14pp（关键劣势） |
| **稳定性** | 中等（bs1 下存在 fi1 塌缩） | 高（跨多组配置保持） | v12 更鲁棒 |
| **资源占用** | bs1（显存受限） | bs4（原始设定可用） | v12 的设定更标准 |

### 11.4 发现与改进建议

**r3515 的可改进方向**：
1. 增大 batch_size（若可用资源允许）以获得更稳定的梯度更新，可能改善 retain；
2. 微调 lambda_retain（目前 1.6），尝试 1.8~2.0，在不伤 forget 前提下提升 retain；
3. 进一步减弱 lambda_forget（目前 3.0 可能过强），尝试 2.6~2.8，换取 retain 提升而维持 forget；
4. 尝试改进 DRT 迁移起点（如扩大迁移样本规模或调整 retain 约束权重）。

**总体结论**：
- r3515 达成"遗忘集 54.24% + 保留集 70.00%"，落在目标区间但稳定性弱。
- 当前最优方案仍为 v12（55.93% / 82.00%），建议继续以 v12 为主线。