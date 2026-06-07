# DRT实现对齐说明（公式 + 代码位置 + 是否GD）

本文严格对齐当前仓库中的 delta_retain_transfer.py 实现，回答三个问题：
1. 目标函数到底是什么
2. 每层每模块的 DeltaW 怎么求
3. 这是 GD 训练还是闭式求解

## 0. 结论先行

- 当前 DRT 不是 GD 训练。
- 每层每模块的 DeltaW 是闭式线性求解器一次性求出。
- 主路径是 Cholesky 求解；失败时回退到 torch.linalg.solve，再失败回退 torch.linalg.lstsq。
- 解出 DeltaW 后做 SVD 截断，写回 LoRA A/B。

对应代码位置：
- solve_delta_retain_lora 定义: line 178
- 线性求解回退链: line 226-232
- SVD 截断: line 263

## 1. 整体流程（逐步对齐）

### 1) 读取参数与数据
- 参数定义 parse_args: line 37
- forget/retain prompt 读取: line 82
- norm_calibrate 参数: line 67
- retain nullspace 可选参数: line 55-58

### 2) 1.7B 到 8B 层映射
- 映射函数 get_layer_map: line 91
- 实际构造 target_new_layers/target_old_layers: line 317-318

映射公式：
old_layer = round(new_layer * L_old / L_new)

### 3) 旧模型侧（1.7B + 旧LoRA）构造遗忘方向监督
- 读取旧 LoRA 对 A/B: read_lora_pair, line 113
- 收集模块输入激活 collect_inputs: line 126
- 计算旧LoRA在forget样本上的输出扰动:
  delta_out = lora_scale * (B_old @ (A_old @ X_old^T))
  代码位置: line 356
- lora_scale = alpha / r: line 298

数学写法：
DeltaH_old = s * B_old * A_old * X_old_f^T,  s = alpha/r

### 4) 新模型侧（8B）收集 forget/retain 激活
- 8B forget 激活收集: line 392-397
- 8B retain 激活收集: line 400-405

记号：
- X_f in R^(N_f x d_in)
- X_r in R^(N_r x d_in)

### 5) 输出维度对齐
- 1.7B 的 DeltaH_old 通过行插值对齐到 8B 输出维度
- 函数 resize_rows: line 169
- 在求解器中调用: line 204

### 6) 每层每模块闭式求解 DeltaW（核心）
- 求解器函数 solve_delta_retain_lora: line 178
- 在主循环逐模块调用: line 426-457

先构造：
- X = [X_f; sqrt(lambda) * X_r]
- Y = [DeltaH_target, 0]

代码位置：
- X 构造: line 210-212
- Y 构造: line 216-217

对应优化目标：
min ||DeltaW * X^T - Y||_F^2 + ridge * ||DeltaW||_F^2

对应闭式解（dual/kernel形式）：
DeltaW = Y * (ridge*I + X*X^T)^(-1) * X

代码位置：
- K = X @ X.T: line 220
- A_mat = ridge*I + K: line 224-225
- Cholesky 主路径: line 227-228
- solve 回退: line 230
- lstsq 回退: line 232
- DeltaW 回代: line 234

### 7) 可选零空间投影（新增试验开关）
- 作用：将 DeltaW 投影到 retain 子空间的正交补
- 代码位置: line 236-251

形式：
DeltaW <- DeltaW * (I - V*V^T)
其中 V 是 retain 输入矩阵的 top-k 右奇异向量。

### 8) 可选范数校准
- 代码位置: line 254-260

目标：保持新模型相对扰动强度接近旧LoRA：
||DeltaW_new|| / ||W_new|| ~= ||DeltaW_old|| / ||W_old||

### 9) SVD 截断并写回 LoRA
- SVD: line 263
- 截断 rank 并组装 B/A: line 268-270
- 写入状态字典: line 467-468

关系：
DeltaW ~= U_r S_r V_r^T,
B = U_r S_r,
A = V_r^T

### 10) 保存 adapter
- 保存 adapter_model.safetensors: line 475
- 保存 adapter_config.json: line 495-497

## 2. 你最关心的问题：8B到底对齐了1.7B LoRA的哪一部分

不是对齐最终权重本身，也不是对齐完整行为分布。
当前实现对齐的是：
- 1.7B LoRA 在 forget 样本上诱导出的“模块输出扰动方向” DeltaH_old
- 同时在 retain 样本上把目标设为接近 0（抑制改动）

更直白地说：
- 你在迁移的是“每个模块对forget输入应产生怎样的输出变化方向”。
- 不是把 1.7B 的 LoRA 参数值直接拷到 8B。
- 由于 8B 空间和数据分布不同，这个方向在8B里可能仍与 retain 方向耦合，所以会出现“遗忘降了，但 retain 也掉”的现象。

## 3. 是否GD训练（最终回答）

不是 GD。
- 没有 optimizer
- 没有反向传播
- 没有训练loop
- 每个模块一次闭式求解得到 DeltaW，然后SVD成LoRA

这就是为什么脚本名里强调 zero-gradient，且主流程是线性代数求解而不是训练。
