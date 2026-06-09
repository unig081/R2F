# 方法1: ng_v2_causal_L0off_L51016 (LoRASuite/ActMap)

## 文件说明

### 核心迁移脚本

| 文件 | 说明 | 使用场景 |
|---|---|---|
| `transfer_lora_actmap.py` | ActMap v4 主脚本 | 默认推荐，使用预计算的R_l进行无梯度迁移 |
| `lora_adaption.py` | LoRASuite 迁移脚本 | 需要CKA层映射/匈牙利头匹配时使用 |

### 前置准备脚本

| 文件 | 说明 | 输出 |
|---|---|---|
| `collect_activations.py` | 采集源/目标模型每层hidden state | `R_l_{i}.pt` (每层对齐矩阵) |
| `compute_cka.py` | 计算CKA层相似度矩阵 | `*_CKA.pt` |

### 后处理精炼脚本

| 文件 | 说明 |
|---|---|
| `nonlinear_spectral_mlp.py` | 用MLP学习log-谱空间的非线性映射，精炼迁移后的LoRA |
| `nonlinear_spectral_ot.py` | 用最优传输(OT)进行谱变换 |
| `nonlinear_remap_power.py` | 简单的符号保持+power变换+B范数保持 |

## 快速开始

### 步骤1: 计算R_l矩阵 (只需执行一次)

```powershell
python collect_activations.py \
  --old_model ./modelzoo/qwen3_1_7B/ \
  --new_model ./modelzoo/qwen3_8B/ \
  --data ./ft-training_set/sampled_100_math_10k.json \
  --n_samples 64 \
  --max_length 128 \
  --output_dir ./tmp/act_align_qwen3_math64/
```

### 步骤2: 执行ActMap v4迁移

修改 `transfer_lora_actmap.py` 中的路径配置后运行：

```powershell
python transfer_lora_actmap.py
```

输出：`./trained_models/xTransform/qwen3_8B_cyber_actmap_v4/`

### 步骤3 (可选): 后处理精炼

```powershell
python nonlinear_spectral_mlp.py    # 或
python nonlinear_remap_power.py
```

### 步骤4 (可选): 二阶段微调

```powershell
python ../common/finetune.py \
  --base_model ./modelzoo/qwen3_8B/ \
  --data_path ./ft-training_set/sampled_100_math_10k.json \
  --output_dir ./trained_models/xTransform/.../xtransform/ \
  --init_lora_weights xtransform \
  --lora_r 32 --lora_alpha 32 --num_epochs 3 --learning_rate 1e-4
```

## 关键参数说明

### transfer_lora_actmap.py 配置

```python
SRC_MODEL_DIR = "./modelzoo/qwen3_1_7B/"     # 源模型路径
TGT_MODEL_DIR = "./modelzoo/qwen3_8B/"       # 目标模型路径
SRC_LORA_DIR  = "./modelzoo/checkpoints/..."  # 源LoRA路径
R_L_DIR       = "./tmp/act_align_qwen3_math64/"  # R_l矩阵目录
OUTPUT_DIR    = "./trained_models/xTransform/..." # 输出目录
```

### lora_adaption.py 参数

```
--old_model          源模型路径
--new_model          目标模型路径
--old_lora_path      源LoRA路径
--qwen_layer_mapping_mode  层映射模式: fixed / cka_monotonic / cka_hungarian
--spectral_calibrate       启用谱校准
--new_rank 32              LoRA秩
--lora_alpha_scale 1.0     alpha缩放
```

## 当前最佳结果

| 配置 | Forget | Retain | 备注 |
|---|---|---|---|
| M1: ActMap v4 | 83.05% | 80.63% | 直接映射，无微调 |
| P3-C: Zero L0 + Perp×2 L5,10,16 | 76.27% | 86.22% | 基于M2 + 选择性权重调整 |
