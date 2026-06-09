# Baseline Method (Current DRT)

## Goal
Transfer 1.7B unlearning LoRA to 8B with retain-aware constraints.

## Method Summary
For each target module (attention by default), solve a closed-form constrained update:

- Match 1.7B LoRA-induced forget delta on forget prompts
- Penalize effect on retain prompts
- Convert solved full matrix to rank-r LoRA by SVD

Core objective (implemented in `delta_retain_transfer.py`):

- Build input matrix: `X = [X_forget; sqrt(lambda_retain) * X_retain]`
- Build target matrix: `Y = [Delta_forget_target, 0]`
- Solve: `DeltaW = Y * (ridge*I + X*X^T)^(-1) * X`
- Optional norm calibration to preserve relative perturbation scale
- Truncate `DeltaW` to LoRA `(A, B)` by top-r SVD

## Current Baseline Settings
- modules: `attn`
- ridge: `1e-2`
- lora_r: `32`
- alpha_scale: `16.0`
- lambda_retain: `{0.2, 0.5, 2.0}` (trade-off sweep)
- eval: hard first-letter match, `judge_model=none`

## Repro Commands (baseline)

```powershell
$py='D:/Anaconda/envs/ali/python.exe'
$base='modelzoo/qwen3_8B'
$old='modelzoo/qwen3_1_7B'
$lora='modelzoo/checkpoints/wmdp-cyber/lr1e-4_g1.0_a2.0'
$forget='datasets/processed_forget_sets/1.7B_to_8B_transfer/wmdp-cyber/forget_10pct.json'
$retain='datasets/processed_forget_sets/1.7B_to_8B_transfer/dual_retain/hsw48_wmdp2.json'
$out='trained_models/xTransform/qwen3_8B_drt_as16_r4802_l02'

& $py delta_retain_transfer.py --old_model $old --new_model $base --old_lora_path $lora --forget_file $forget --retain_file $retain --output_dir $out --max_forget 59 --max_retain 50 --max_input_length 512 --tokens_per_sample 4 --modules attn --ridge 1e-2 --lambda_retain 0.2 --lora_r 32 --alpha_scale 16.0
```

## New Optional Switch (Nullspace Trial)
A minimal orthogonalization switch is added for experiments:
- `--retain_nullspace_rank K` (default 0: disabled)
- `--retain_nullspace_center` (enabled by default)

When `K>0`, solved `DeltaW` is projected to nullspace of top-K retain input subspace:
`DeltaW <- DeltaW (I - V V^T)`.

This is optional and does not change baseline behavior when `K=0`.
