# 方法2: DRT (Delta-Retain Transfer)

## 文件说明

### 核心迁移脚本

| 文件 | 说明 |
|---|---|
| `delta_retain_transfer.py` | DRT 主脚本：闭式求解 + SVD截断 → LoRA A/B |

### 二阶段微调脚本

| 文件 | 说明 | 使用场景 |
|---|---|---|
| `stage2_mcq_finetune.py` | MCQ定点微调 | **推荐**，只在Answer位置优化 |
| `stage2_hybrid_finetune.py` | 混合微调 | 结合MCQ和序列级损失 |
| `stage2_lora_finetune.py` | LoRA微调 | 标准LoRA微调 |
| `stage2_simple_finetune.py` | 简单微调 | 最简实现 |

## 快速开始

### 步骤1: DRT 迁移

```powershell
python delta_retain_transfer.py \
  --old_model ./modelzoo/qwen3_1_7B/ \
  --new_model ./modelzoo/qwen3_8B/ \
  --old_lora_path ./modelzoo/checkpoints/wmdp-cyber/lr1e-4_g1.0_a2.0/ \
  --forget_file ./datasets/processed_forget_sets/1.7B_to_8B_transfer/wmdp-cyber/forget_10pct.json \
  --retain_file ./datasets/processed_forget_sets/1.7B_to_8B_transfer/dual_retain/hsw48_wmdp2.json \
  --output_dir ./trained_models/xTransform/qwen3_8B_drt_as16_r5950_l02 \
  --max_forget 59 --max_retain 50 \
  --max_input_length 512 --tokens_per_sample 4 \
  --modules attn --ridge 1e-2 --lambda_retain 0.2 \
  --lora_r 32 --alpha_scale 16.0 \
  --target_mapping linear_map \
  --norm_calibrate
```

### 步骤2: 二阶段微调

```powershell
python stage2_mcq_finetune.py \
  --base_model ./modelzoo/qwen3_8B/ \
  --lora_path trained_models/xTransform/qwen3_8B_drt_as16_r5950_l02_lmapRw02_ns0_top8 \
  --forget_file ./datasets/.../forget_10pct.json \
  --retain_file ./datasets/.../remain_10pct.json \
  --output_dir ./trained_models/xTransform/stage2_v12/ \
  --lambda_forget 900 --lambda_retain 160 \
  --lora_r 32 --learning_rate 3e-5 \
  --num_epochs 1 --batch_size 4 \
  --norm_stabilize
```

## 关键参数说明

### delta_retain_transfer.py

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--modules` | `attn` | 迁移的模块类型: attn/mlp/all |
| `--ridge` | `1e-2` | Ridge正则系数 |
| `--lambda_retain` | 测试了 {0.2, 0.5, 2.0} | retain约束权重（越大越保守） |
| `--alpha_scale` | `16.0` | alpha缩放倍数 |
| `--target_mapping` | `linear_map` | 旧→新输出维度映射方式 |
| `--target_map_use_retain` | 建议开启 | linear_map时联合拟合forget+retain |
| `--target_map_retain_weight` | `0.2` | retain在linear_map拟合中的权重 |
| `--retain_nullspace_rank` | `0`（关闭） | >0时启用零空间投影 |
| `--norm_calibrate` | 建议开启 | 范数校准 |
| `--new_layer_start/end` | 可限定 | 只迁移部分层（如top8: 28-35） |

### stage2_mcq_finetune.py

| 参数 | 说明 |
|---|---|
| `--lambda_forget` | Forget损失权重 |
| `--lambda_retain` | Retain损失权重 |
| `--norm_stabilize` | 范数稳定化（v12成功关键） |

## 命名约定解读

以 `qwen3_8B_drt_as16_r5950_l02_lmapRw02_ns0_top8` 为例：

| 字段 | 含义 |
|---|---|
| `drt` | Delta-Retain Transfer 方法 |
| `as16` | alpha_scale=16 |
| `r5950` | forget=59 + retain=50 探针样本 |
| `l02` | lambda_retain=0.2 |
| `lmapRw02` | linear_map + retain_weight=0.2 |
| `ns0` | retain_nullspace_rank=0 |
| `top8` | 仅迁移后8层 (28-35) |

## 当前最佳结果

| 配置 | Forget | Retain | 备注 |
|---|---|---|---|
| 8B Base | 86.44% | 80.45% | 无LoRA基线 |
| DRT top8 (仅迁移) | 72.88% | 64.00% | 迁移后未微调 |
| v12 full (迁移+微调) | 55.93% | 82.00% | 当前最优折中 |
| 1.7B Source LoRA | 27.12% | 86.22% | 源模型上界 |

## 方法局限

详见 `DRT_Current_Method_Analysis.md`，核心问题：
1. **Retain约束不足**：当前L2约束不能保证遗忘方向的几何结构
2. **目标构建偏置**：linear_map 仅用forget样本拟合M会导致retain外推偏移
3. **信息压缩**：8B上WMDP forget/retain基线相似度仅Δ=4.44%，强制遗忘必然伤害retain
