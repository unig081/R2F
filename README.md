# R2F TOFU/LLaMA Gradient Decoder

This repository implements the TOFU/LLaMA R2F reproduction plan in `plan.txt`.
The main path is a coordinate-wise gradient decoder:

```text
decoder(A[:, i], B[o, :], dA[:, i], dB[o, :], layer metadata, module type)
  -> dense Transformer gradient dW[o, i]
```

Dense gradients and R2F updates are restricted to Transformer dense weights:
`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`.
Embeddings, norm layers, and `lm_head` are not captured or updated.

## Remote Setup

On the GPU machine, pull the repository into:

```bash
/mnt/data1/zxc/R2F
```

Then create the project conda environment:

```bash
cd /mnt/data1/zxc/R2F
bash scripts/setup_env.sh
```

Activate for later runs:

```bash
source scripts/env.sh
conda activate zxc_r2f
```

The conda environment name is `zxc_r2f`.
The environment path is `/mnt/data1/conda_env/zxc_r2f` on the GPU server.

## Smoke Test

After confirming model and data paths in `configs/r2f_tofu_llama.yaml`, run:

```bash
source scripts/env.sh
conda activate zxc_r2f
bash scripts/run_tofu_r2f_smoke.sh
```

The smoke test captures 1B LoRA and dense gradient pairs, trains a tiny decoder,
runs a short 3B LoRA GA+GD pass to capture decoder inputs, predicts dense
`dW_hat`, applies one R2F update to 3B, and evaluates Base-3B / LoRA-GA+GD-3B /
R2F-3B on a small subset.

## Full Run

```bash
source scripts/env.sh
conda activate zxc_r2f
bash scripts/run_tofu_r2f_full.sh
```

Outputs are written under `results/`, reports under `reports/`, and logs under
`logs/`. R2F gradient shards are stored under
`results/r2f_3b/eta_*/predicted_dense_gradient_shards/`; dense deltas are
derived as `dense_delta = -eta * dW_hat`.
