# LoRASuite
[![Arxiv](https://img.shields.io/badge/arXiv-2505.13515-B21A1B)](https://arxiv.org/abs/2505.13515)

The Official PyTorch implementation of [**LoRASuite: Efficient LoRA Adaptation Across Large Language Model Upgrades**](https://arxiv.org/abs/2505.13515).

Authors: [Yanan Li](https://scholar.google.com/citations?user=fCk0tD8AAAAJ&hl=zh-CN) $^\dagger$, [Fanxu Meng](https://fxmeng.github.io/) $^\ddagger$, [Muhan Zhang](https://muhanzhang.github.io/) $^\ddagger$, Shiai Zhu $^\nmid$, [Shangguang Wang](https://wangshangguang.github.io/) $^\dagger$, [Mengwei Xu](https://xumengwei.github.io/) $^\dagger$  
$^\dagger$ Beijing University of Posts and Telecommunications, $^\ddagger$ Peking University, $^\nmid$ Unaffiliated

## Overview
As Large Language Models (LLMs) are frequently updated, LoRA weights trained on earlier versions quickly become obsolete. The conventional practice of retraining LoRA weights from scratch on the latest model is costly, time-consuming, and environmentally detrimental, particularly as the diversity of LLMs and downstream tasks expands. This motivates a critical question: "How can we efficiently leverage existing LoRA weights to adapt to newer model versions?" To address this, we propose LoRASuite, a modular approach tailored specifically to various types of LLM updates. First, we compute a transfer matrix utilizing known parameters from both old and new LLMs. Next, we allocate corresponding layers and attention heads based on centered kernel alignment and cosine similarity metrics, respectively. A subsequent small-scale, skillful fine-tuning step ensures numerical stability. Experimental evaluations demonstrate that LoRASuite consistently surpasses small-scale vanilla LoRA methods. Notably, on backbone LLMs such as MiniCPM and Qwen, LoRASuite even exceeds the performance of full-scale LoRA retraining, with average improvements of +1.4 and +6.6 points on math tasks, respectively. Additionally, LoRASuite significantly reduces memory consumption by 5.5 GB and computational time by 78.23%.

## Experiments

### Setup

You can create the environment from the environment.yml file:

```
conda env create -f environment.yml
```


We modify the Peft `LoraLayer` to enable weight initialization from a specified path instead of the default Kaiming initialization for matrix A and zero initialization for matrix B. Please refer to our provided Peft implementation for further details.
```
def xtransform_init(self, adapter_name, init_lora_path, current_key):
        from peft import load_peft_weights

        weight = self.get_base_layer().weight
        dtype = weight.dtype
        device = weight.device

        old_lora_weights = load_peft_weights(init_lora_path)
        if f"base_model.model.{current_key}.lora_A.weight" in old_lora_weights.keys():
            self.lora_A[adapter_name].weight.data = old_lora_weights[f"base_model.model.{current_key}.lora_A.weight"].to(device, dtype)
            self.lora_B[adapter_name].weight.data = old_lora_weights[f"base_model.model.{current_key}.lora_B.weight"].to(device, dtype)
        else:
            # nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
            # nn.init.zeros_(self.lora_B[adapter_name].weight)
            nn.init.zeros_(self.lora_A[adapter_name].weight)
            nn.init.zeros_(self.lora_B[adapter_name].weight)
            self.lora_A[adapter_name].weight.requires_grad = False
            self.lora_B[adapter_name].weight.requires_grad = False
```


### Vanilla LoRA
```
CUDA_VISIBLE_DEVICES=0 python finetune.py --base_model ./modelzoo/MiniCPM-S-1B-sft-llama-format/ --data_path ./ft-training_set/math_10k.json --output_dir ./trained_models/MiniCPM-S-1B-sft-llama-format-lora-math-r32-qkvo-10k/ --init_lora_weights True --warmup_steps 100 --lora_r 32 --lora_alpha 32 --lora_dropout 0 --batch_size 16 --micro_batch_size 4 --num_epochs 3 --learning_rate 3e-4 --cutoff_len 256 --adapter_name lora --target_modules "['q_proj','k_proj', 'v_proj', 'o_proj']" 

bash evaluate.sh 0 MiniCPM-1B ./modelzoo/MiniCPM-S-1B-sft-llama-format/ ./trained_models/MiniCPM-S-1B-sft-llama-format-lora-math-r32-qkvo-10k/ 4 true 0 | tee ./logs/MiniCPM-S-1B-sft-llama-format-lora-math-r32-qkvo-10k-2025XXXX.log
```

### LoRASuite w/o LFT
```
CUDA_VISIBLE_DEVICES=0,1 python lora_adaption.py --new_model ./modelzoo/MiniCPM-2B-sft-fp32-llama-format/ --old_model ./modelzoo/MiniCPM-S-1B-sft-llama-format/ --old_lora_path ./trained_models/MiniCPM-S-1B-sft-llama-format-lora-math-r32-qkvo-10k/

bash evaluate.sh 0 MiniCPM-2B ./modelzoo/MiniCPM-2B-sft-fp32-llama-format/ ./trained_models/xTransform/MiniCPM-2B-sft-fp32-llama-format_MiniCPM-S-1B-sft-llama-format-lora-math-r32-qkvo-10k/ 4 true 0 | tee ./logs/xTransform/MiniCPM-2B-sft-fp32-llama-format_MiniCPM-S-1B-sft-llama-format-lora-math-r32-qkvo-10k-2025XXXX.log
```

### LoRASuite 
```
CUDA_VISIBLE_DEVICES=0 python finetune.py --base_model ./modelzoo/MiniCPM-2B-sft-fp32-llama-format/ --data_path ./ft-training_set/sampled_100_math_10k.json --output_dir ./trained_models/xTransform/MiniCPM-2B-sft-fp32-llama-format_MiniCPM-S-1B-sft-llama-format-lora-math-r32-qkvo-10k/ --init_lora_weights xtransform --warmup_steps 0 --lora_r 32 --lora_alpha 32 --lora_dropout 0 --batch_size 16 --micro_batch_size 4 --num_epochs 3 --learning_rate 1e-4 --cutoff_len 256 --val_set_size 0 --adapter_name lora --target_modules "['q_proj', 'k_proj', 'v_proj', 'o_proj']" 

bash evaluate.sh 0 MiniCPM-2B ./modelzoo/MiniCPM-2B-sft-fp32-llama-format/ ./trained_models/xTransform/MiniCPM-2B-sft-fp32-llama-format_MiniCPM-S-1B-sft-llama-format-lora-math-r32-qkvo-10k/xtransform/ 4 true 0 | tee ./logs/xTransform/MiniCPM-2B-sft-fp32-llama-format_MiniCPM-S-1B-sft-llama-format-lora-math-r32-qkvo-10k_100-2025XXXX.log
```

---

## Unlearning LoRA Transfer: Qwen3-1.7B → Qwen3-8B

### Task Description

Transfer an **unlearning LoRA** (wmdp-cyber forget task) trained on Qwen3-1.7B to Qwen3-8B **without any retraining or access to the forget/retain datasets**. The goal is to achieve proportional accuracy changes on 8B matching those observed on 1.7B.

**Constraint**: Pure weight-based transfer only — no labeled data, no gradient computation on the target model.

### Model Architecture Differences

| Property | Qwen3-1.7B (source) | Qwen3-8B (target) |
|---|---|---|
| Layers | 28 | 36 |
| Hidden size | 2048 | 4096 |
| Intermediate size | 6144 | 12288 |
| Q heads | 16 | 32 |
| KV heads | 8 | 8 |
| num_rep (Q/KV) | 2 | 4 |

Layer mapping: `[int(i * 36 / 28) for i in range(28)]` → 8 target layers (4,8,13,17,22,26,31,35) have no source, filled by nearest-neighbor copy.

### Source LoRA Configuration

- Path: `./modelzoo/checkpoints/wmdp-cyber/lr1e-4_g1.0_a2.0/`
- Rank: 32, Alpha: 32 (alpha/r = 1.0)
- Modules: q, k, v, o, gate, up, down projections (7 modules × 28 layers = 196 LoRA pairs)

### Baselines

| Model | Forget Acc (wmdp-cyber) | Retain Acc |
|---|---|---|
| Qwen3-1.7B base | 98.31% | 98.51% |
| Qwen3-1.7B + unlearning LoRA | **27.12%** (↓71.19pp) | **86.22%** (↓12.29pp) |
| Qwen3-8B base | 91.53% (54/59) | 87.00% |
| Qwen3-8B with 8B-trained LoRA (reference) | 86.44% | 80.63% |

**Transfer Target**: forget ↓ proportionally matching 1.7B (ideally ~27%), retain stays high (>70%).

### Experiment Results

#### Approach 1: xTransform (LoRASuite W_x method) — Pure Transfer

`lora_adaption.py` with spectral calibration, r=64, alpha_scale=2.0, `--fill_missing_layers`.

W_x = pinv(E_1.7B_common) @ E_8B_common  (from shared token embeddings).

| Variant | Forget Acc | Retain Acc | Notes |
|---|---|---|---|
| xTransform baseline (svdffn_v1) | **91.53%** | 88.00% | = base, zero effect |
| 20× alpha amplification | 93.22% | 83.00% | wrong direction |
| negated adapter | 93.22% | 87.00% | wrong direction |

**Result: Complete failure — zero effect regardless of scale or sign.**

#### Approach 2: Per-Layer W_x from Q Weight Matrices — Pure Transfer

`lora_adaption.py --per_layer_wx` estimates R_l at each layer by solving `W_old_Q @ R_l = W_new_Q` (least-squares fit on weight matrices), replacing the global embedding-derived W_x with a layer-specific version. No data required.

| Variant | Forget Acc | Retain Acc | Notes |
|---|---|---|---|
| per_layer_wx (plwx_v1) | **93.22%** | 88.00% | = base (55/59), zero effect |

**Result: Also complete failure — per-layer Q-weight alignment still produces zero unlearning effect.**

#### Why Approach 1 Succeeded on Other Tasks but Fails Here

The original LoRASuite paper demonstrated success on:
- MiniCPM-1B → 2B (math fine-tuning, same family, ~2× scale)  
- Llama-2-7B → Llama-3-8B (same scale)  
- Qwen1.5-1.8B → Qwen2.5-3B (math fine-tuning, same-ish scale)  

**Root cause of failure on our task:**

1. **Task type mismatch**: LoRASuite was designed for *fine-tuning* LoRAs (adding math/commonsense capability). The relevant features for math are partially surface-level (tokens like digits, operators) and thus partially captured by the embedding-space W_x. **Unlearning** LoRAs target deep semantic circuits for specific knowledge (cybersecurity concepts) — these circuits have no relationship to the embedding space.

2. **Scale mismatch**: 1.7B → 8B is a 5× jump. All original experiments involved models of similar scale (1B↔2B, 7B↔8B). For similar-scale same-family upgrades, the embedding-derived W_x is a reasonable approximation of the deep layer representation alignment. For 5× scale differences, the representation geometries in deep layers diverge substantially.

3. **W_x assumption**: W_x assumes the same linear transformation maps hidden states at ALL layers (derived from layer 0's embedding). This is increasingly inaccurate as depth increases, because each layer applies non-linear transformations that gradually rotate and decouple the representation geometry from the embedding space.

4. **LFT dependency**: The full LoRASuite pipeline includes a Lightweight Fine-Tuning (LFT) step on 100 labeled task samples after transfer, which corrects residual errors. "LoRASuite w/o LFT" (pure transfer) performs significantly worse. For unlearning, we cannot perform LFT because we have no labeled forget/retain data.

**Evidence**: Both embedding-based W_x and per-layer Q-weight-derived W_x give the same result: ~93% forget accuracy (essentially base model performance). Even at 20× alpha, the result is unchanged. This strongly confirms the transferred LoRA direction is truly orthogonal to the cybersecurity-forgetting subspace in 8B's weight space — the model simply ignores it.

#### Pending / Next Steps

1. **Approach 3: Activation-based alignment with generic (non-cybersecurity) text**  
   - Run both models on ~100 generic text samples (math / Wikipedia — NOT the forget/retain set)
   - Collect intermediate activations at each layer: `h_old[l]` and `h_new[l]`
   - Compute `R_l = lstsq(h_old[l].T, h_new[l].T)` per layer — a data-driven layer-wise transform
   - Use `R_l` for LoRA transfer instead of the static embedding/weight-derived W_x
   - **Why this might work**: Both Qwen3 models were trained on the same data distribution; their actual computation at each layer should be geometrically aligned. Activation-based `R_l` captures the true runtime representation correspondence, not just the static weight correspondence.
   - **Memory plan**: Run 1.7B, save activations to disk → run 8B, save activations to disk → compute R_l offline → re-run `lora_adaption.py` with `--act_align_path`
   - **Implementation needed**: Add `--act_align_path` flag to `lora_adaption.py` and a separate `collect_activations.py` script

2. **Accept fundamental limitation**: If Approach 3 also fails, document that data-free weight-based unlearning LoRA transfer is infeasible for the 5× scale jump between independent-training same-family models. The minimum viable approach requires either: (a) access to a small domain proxy dataset (even generic cybersecurity Wikipedia text, not the exact WMDP evaluation set), or (b) access to the original forget set for gradient computation.

### Key Technical Notes

- PEFT requires **absolute path** for `--lora_path` (HFValidationError with relative `./` paths)
- Result JSON key: `res['summary']['PEFT_Forget_Accuracy']` as string `"X/Y = Z%"`
- ans_tokens for Qwen3: A=32, B=33, C=34, D=35
- 8B model: 399 safetensor shards, ~16GB in float16

---

## Useful Links
- LLM-Adapters: https://github.com/AGI-Edgerunners/LLM-Adapters
- DoRA: https://github.com/NVlabs/DoRA

## Citation

Please cite our paper if it's helpful to your work!
```
@article{li2025lorasuite,
  title={LoRASuite: Efficient LoRA Adaptation Across Large Language Model Upgrades},
  author={Li, Yanan and Meng, Fanxu and Zhang, Muhan and Zhu, Shia and Wang, Shangguang and Xu, Mengwei},
  journal={arXiv preprint arXiv:2505.13515},
  year={2025}
}
```
