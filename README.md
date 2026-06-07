## Full Run 代码逻辑

`scripts/run_tofu_r2f_full.sh` 按顺序执行 5 个任务。脚本启用了
`set -euo pipefail`，所以任何一步失败都会停止后续流程。

1. `r2f_tofu.unlearn_lora`：训练 3B LoRA GA+GD baseline。
   - 加载已经 TOFU 微调过的 LLaMA 3B target model。
   - 构造 TOFU forget/retain paired batch。
   - 对目标 Transformer 模块注入 LoRA。
   - 使用 GA+GD loss 训练：

     ```text
     loss = gamma * (-forget_ce) + alpha * retain_ce
     ```

   - 使用 `grad_accum_steps` 做梯度累计，到累计边界才执行 `optimizer.step()`。
   - 保存 adapter 到 `results/lora_gagd_3b/adapter`。
   - 这个 adapter 同时作为 LoRA-GA+GD-3B baseline，也作为第 4 步 R2F 捕获 3B LoRA 梯度的来源。

2. `r2f_tofu.unlearn_dense`：采集 1B LoRA 梯度和 dense 梯度配对样本。
   - 加载已经 TOFU 微调过的 LLaMA 1B source model 两份。
   - 第一份注入 LoRA，只记录 LoRA 的 `A/B/dA/dB`。
   - 第二份不注入 LoRA，只让目标 Transformer dense weights 可训练并记录 `W/dW`。
   - 两份模型使用同一批 TOFU forget/retain batch 和同一个 GA+GD loss。
   - 每到梯度累计边界，按 coordinate `(o, i)` 采样并写入 decoder 训练样本：

     ```text
     A_col = A[:, i]
     B_row = B[o, :]
     dA_col = dA[:, i]
     dB_row = dB[o, :]
     target_norm = dW[o, i] / grad_rms
     ```

   - 样本保存到 `results/gradients/`。

3. `r2f_tofu.train_decoder`：训练 coordinate-wise gradient decoder。
   - 读取第 2 步生成的 1B paired gradient samples。
   - 训练一个 MLP decoder，把 LoRA coordinate 特征映射成 normalized dense gradient。
   - 输入特征包括 `A_col/B_row/dA_col/dB_row`、rank-wise interaction、RMS/dot 统计量、layer depth 特征和 module embedding。
   - loss 是 Huber loss 加一个小权重 sign loss。
   - 保存 decoder checkpoint 到 `results/decoder/checkpoint.pt`，同时保存用于反归一化的 gradient RMS 统计。

4. `r2f_tofu.apply_r2f`：复用 3B LoRA adapter，预测 3B dense gradient。
   - 加载第 1 步保存的 `results/lora_gagd_3b/adapter`，不重复训练 3B LoRA。
   - 在若干 TOFU forget/retain batch 上执行 GA+GD backward，记录 3B LoRA 的 `A/B/dA/dB`。
   - 加载第 3 步训练好的 decoder。
   - 对每个目标 3B Transformer dense weight 按 coordinate block 枚举 `(o, i)`。
   - 用 3B LoRA 梯度构造 decoder 输入，预测并反归一化：

     ```text
     dW_hat[o, i] = decoder(features) * grad_rms
     dense_delta = -eta * dW_hat
     W = W + dense_delta
     ```

   - 预测出的 dense gradient shards 保存到
     `results/r2f_3b/eta_*/predicted_dense_gradient_shards/`。
   - 每个 eta 的 R2F updated model 保存到 `results/r2f_3b/eta_*/updated_model/`。

5. `r2f_tofu.evaluate_tofu`：评估 Base、LoRA baseline 和 R2F 模型。
   - 评估原始 TOFU 微调 3B model。
   - 评估第 1 步保存的 LoRA-GA+GD-3B adapter。
   - 评估第 4 步保存的各个 R2F updated model。
   - 输出指标和样例到 `results/eval/`、`results/summary.csv` 和 `reports/tofu_r2f_summary.md`。

Outputs are written under `results/`, reports under `reports/`, and logs under
`logs/`. R2F gradient shards are stored under
`results/r2f_3b/eta_*/predicted_dense_gradient_shards/`; dense deltas are
derived as `dense_delta = -eta * dW_hat`.
