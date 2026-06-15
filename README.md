# R2F TOFU 模型族流程

本仓库运行了一个针对 TOFU 遗忘学习的 proxy 到 target R2F 实验。默认模型族是 Llama 3.2，也可以在 yaml 或环境变量里切换到 Phi-4 或 Qwen3。核心思路如下：

1. 训练一个 3B LoRA GA+GD 遗忘适配器。
2. 收集成对的 1B LoRA 梯度和稠密（dense）梯度坐标样本。
3. 在 1B 样本上训练一个逐坐标的梯度解码器。
4. 复用 3B LoRA 适配器，预测稠密梯度，并写入更新后的 3B 模型。
5. 评估基础模型、LoRA 和 R2F 的输出结果。

默认情况下只针对注意力投影权重。不同模型族的默认 LoRA 模块由 `src/r2f_tofu/model_families.py` 管理。

## 仓库结构

- `src/r2f_tofu/`: Python 源码包和命令行模块。
- `configs/`: 完整运行的配置以及每个流程步骤的单独配置。
- `scripts/`: 环境设置和各步骤的可执行脚本。
- `tests/`: 轻量级解码器/R2F 单元测试。
- `results/`, `logs/`, `reports/`: 生成的输出结果、日志和总结。

## 环境

远程 GPU 环境默认配置实现在 `scripts/env.sh` 中：

- 项目根目录: `/mnt/data1/zxc/R2F`
- Conda 环境: `/mnt/data1/conda_env/zxc_r2f`
- 模型族: `R2F_MODEL_FAMILY=llama`
- 源模型 (source/proxy model): Llama 3.2 1B TOFU 模型
- 目标模型 (target model): Llama 3.2 3B TOFU 模型
- 数据: TOFU 的 `forget05.json` 和 `retain95.json`

创建或刷新环境：

```bash
./scripts/setup_env.sh
```

对于已有环境，激活它并安装代码包：

```bash
source scripts/env.sh
python -m pip install --no-user -e .
```

所有路径都可以通过环境变量进行覆盖，例如 `R2F_MODEL_FAMILY`、`R2F_SOURCE_MODEL`、`R2F_TARGET_MODEL`、`R2F_FORGET_FILE`、`R2F_RETAIN_FILE` 和 `R2F_OUTPUT_DIR`。

## 模型族接口

在任意 yaml 中设置：

```yaml
model_family: llama  # llama -> Llama 3.2; phi -> Phi-4; qwen -> Qwen3
```

也可以用环境变量覆盖：

```bash
R2F_MODEL_FAMILY=phi ./scripts/run_tofu_r2f_01_lora.sh
R2F_MODEL_FAMILY=qwen ./scripts/run_tofu_r2f_full.sh
```

当前内置映射：

| family | proxy 默认目录 | target 默认目录 | Hugging Face 参考 | 默认 LoRA 模块 |
|---|---|---|---|---|
| `llama` | `model/proxy/llama3.2_1B` | `model/target/llama3.2_3B` | `meta-llama/Llama-3.2-1B-Instruct`, `meta-llama/Llama-3.2-3B-Instruct` | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| `phi` | `model/proxy/phi4_3B` | `model/target/phi4_14B` | `microsoft/Phi-4-mini-instruct`, `microsoft/phi-4` | `qkv_proj`, `o_proj` |
| `qwen` | `model/proxy/qwen3_1.7B` | `model/target/qwen3_8B` | `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-8B` | `q_proj`, `k_proj`, `v_proj`, `o_proj` |

如果 `paths.source_model` 或 `paths.target_model` 为空，配置解析会按 `model_family` 自动填入上表目录。`scripts/env.sh` 会优先寻找已经存在且含 `config.json` 的模型目录；仓库里的空占位目录不会被当成可加载模型。

## 运行流程

使用完整配置运行所有步骤：

```bash
./scripts/run_tofu_r2f_full.sh
```

单步运行：

```bash
./scripts/run_tofu_r2f_01_lora.sh
./scripts/run_tofu_r2f_02_dense_samples.sh
./scripts/run_tofu_r2f_03_train_decoder.sh
./scripts/run_tofu_r2f_04_apply_r2f.sh
./scripts/run_tofu_r2f_05_eval.sh
```

每个脚本都接受一个配置路径：

```bash
./scripts/run_tofu_r2f_03_train_decoder.sh configs/r2f_tofu_03_train_decoder.yaml
```

进行极简（sanity）运行：

```bash
./scripts/run_tofu_r2f_smoke.sh
```

## 步骤详情

### 01 LoRA GA+GD

模块: `r2f_tofu.unlearn_lora`

配置: `configs/r2f_tofu_01_lora.yaml`

输出:

- `results/lora_gagd_3b/adapter/`
- `results/lora_gagd_3b/train_stats.json`
- `results/lora_gagd_3b/lora_shape_report.json`

损失函数:

```text
loss = gamma * (-forget_ce) + alpha * retain_ce
```

`unlearning.epochs` 会重复迭代数据加载器，但 `unlearning.max_steps` 仍然是严格的硬上限。例如，`epochs: 5` 和 `max_steps: 512` 会在 512 个微批次（micro-batches）之后停止，即使 5 个完整 epoch 会更长。

### 02 稠密样本捕获 (Dense Sample Capture)

模块: `r2f_tofu.unlearn_dense`

配置: `configs/r2f_tofu_02_dense_samples.yaml`

输出:

- `results/gradients/llama1b_decoder_samples.pt`
- `results/gradients/llama1b_grad_stats.json`
- `results/gradients/llama1b_gradient_shape_report.json`

这一步仅推进 1B LoRA 的轨迹。在每一个梯度累积的边界上，它会将一个稠密标签模型同步至当前的 LoRA 有效状态：

```text
W_eff = W_base + scale * B @ A
```

使用相同的遗忘/保留批次数据不仅产生 LoRA 特征 `A/B/dA/dB`，也产生稠密标签 `dW`。该稠密模型仅记录标签；它不执行优化器更新（optimizer steps）。

### 03 解码器训练 (Decoder Training)

模块: `r2f_tofu.train_decoder`

配置: `configs/r2f_tofu_03_train_decoder.yaml`

输出:

- `results/decoder/checkpoint.pt`
- `results/decoder/stats.json`
- `reports/decoder_training_curve.json`

解码器预测归一化的稠密梯度坐标：

```text
pred_norm = pinv_mean_norm + MLP_residual(features)
dW_hat = pred_norm * grad_rms
```

只有在训练完成后，checkpoint 才会被标记为 `complete: true`。如果不开启 `r2f.allow_incomplete_decoder`，第 04 步将拒绝不完整的 checkpoint。

### 04 应用 R2F

模块: `r2f_tofu.apply_r2f`

配置: `configs/r2f_tofu_04_apply_r2f.yaml`

输入:

- 01 adapter: `results/lora_gagd_3b/adapter/`
- 03 decoder: `results/decoder/checkpoint.pt`

输出:

- `results/r2f_3b/lora_gradient_capture_stats.json`
- `results/r2f_3b/eta_*/predicted_dense_gradient_shards/`
- `results/r2f_3b/eta_*/update_stats.json`
- `results/r2f_3b/eta_*/updated_model/`

针对每个 eta:

```text
W_eff = W_base + scale * B @ A
W_out = W_eff - eta * dW_hat
```

### 05 兼容 Handoff 的 TOFU 评估

模块: `r2f_tofu.evaluate_tofu`

配置: `configs/r2f_tofu_05_eval.yaml`

输出:

- `results/eval/*/TOFU_EVAL.json`
- `results/eval/*/TOFU_SUMMARY.json`
- `results/summary.csv`
- `reports/tofu_r2f_summary.md`

这个本地评估器复刻了来自 `junior_llama_tofu_eval_bundle/eval_runtime/open-unlearning-main` 中的移交（handoff）TOFU 评估逻辑：包含模型 tokenizer 自带聊天模板标记化（tokenization）、`forget05_perturbed` / `retain_perturbed` / `real_authors_perturbed` / `world_facts_perturbed` 划分、概率、ROUGE-L 召回率、真实率 (truth-ratio)、调和平均模型效用 (harmonic-mean model utility)、min-k MIA 隐私泄露 (privleak) 和提取强度 (extraction strength)。

## 实用诊断

根据其样本文件来评估一个已训练的解码器：

```bash
python -m r2f_tofu.eval_decoder --config configs/r2f_tofu_03_train_decoder.yaml
```

运行本地测试：

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## 注意事项

- 步骤配置故意地只包含会被该步骤读取的字段。
- `configs/r2f_tofu.yaml` 仍作为 `run_tofu_r2f_full.sh` 的完整配置。
- `results/`, `logs/`, 和 `reports/` 为生成的产物。请勿将它们视为源码依赖。
