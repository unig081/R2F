import os
import torch
import json
import argparse
import random
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Sequence, Any
import transformers
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset
import wandb

IGNORE_INDEX = -100

class JointDataset(torch.utils.data.Dataset):
    """
    Zips forget_data and retain_data together for GA+GD training.
    """
    def __init__(self, forget_data, retain_data):
        self.forget_data = forget_data
        self.retain_data = retain_data
        
    def __len__(self):
        return len(self.forget_data)
        
    def __getitem__(self, idx):
        # Randomly sample a retain instance for each forget instance 
        retain_idx = random.randint(0, len(self.retain_data) - 1)
        return {
            "forget": self.forget_data[idx],
            "retain": self.retain_data[retain_idx]
        }

class UnlearnTrainer(Trainer):
    """
    Trainer supporting pure GA (Gradient Ascent) or GA+GD (Gradient Difference).
    """
    def __init__(self, method="ga", gamma=1.0, alpha=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = method
        self.gamma = gamma
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 1. Check if we are in joint training mode (GA+GD)
        if "forget" in inputs and "retain" in inputs:
            # GA on forget set
            forget_outputs = model(**inputs["forget"])
            forget_loss = -forget_outputs.loss
            
            # GD on retain set
            retain_outputs = model(**inputs["retain"])
            retain_loss = retain_outputs.loss
            
            final_loss = self.gamma * forget_loss + self.alpha * retain_loss
            return (final_loss, forget_outputs) if return_outputs else final_loss

        # 2. Otherwise, standard single-set training/evaluation (Pure GA or Eval)
        outputs = model(**inputs)
        loss = outputs.loss
        
        if model.training:
            if self.method == "ga":
                final_loss = -loss
            else:
                final_loss = loss
        else:
            # Evaluate using standard CrossEntropy loss
            final_loss = loss
            
        return (final_loss, outputs) if return_outputs else final_loss

@dataclass
class DataCollatorForForgetDataset(object):
    tokenizer: transformers.PreTrainedTokenizer
    max_length: int = 1024

    def _collate_single_type(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = [], []
        for instance in instances:
            prompt = instance["question"]
            answer = instance.get("gold_answers", instance.get("answers", [""])[0])
            if isinstance(answer, list):
                answer = answer[0] if len(answer) > 0 else ""
                
            prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            answer_tokens = self.tokenizer(answer, add_special_tokens=False)["input_ids"] + [self.tokenizer.eos_token_id]
            
            full_ids = prompt_tokens + answer_tokens
            label_ids = [IGNORE_INDEX] * len(prompt_tokens) + answer_tokens

            # Truncate long samples to keep VRAM usage bounded during micro-tuning.
            if len(full_ids) > self.max_length:
                full_ids = full_ids[-self.max_length:]
                label_ids = label_ids[-self.max_length:]
            
            input_ids.append(torch.tensor(full_ids))
            labels.append(torch.tensor(label_ids))
            
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

    def __call__(self, instances: Sequence[Any]) -> Dict[str, Any]:
        # Support GA+GD where each instance is a dict of {"forget": dict, "retain": dict}
        if isinstance(instances[0], dict) and "forget" in instances[0] and "retain" in instances[0]:
            forget_insts = [inst["forget"] for inst in instances]
            retain_insts = [inst["retain"] for inst in instances]
            return {
                "forget": self._collate_single_type(forget_insts),
                "retain": self._collate_single_type(retain_insts)
            }
        else:
            return self._collate_single_type(instances)

def _apply_trainable_filter(model, args):
    if not args.train_only_modules and not args.train_only_layers:
        return

    allowed_modules = set(args.train_only_modules or [])
    allowed_layers = None
    if args.train_only_layers:
        allowed_layers = set(int(x.strip()) for x in args.train_only_layers.split(",") if x.strip())

    n_total = 0
    n_active = 0
    for name, param in model.named_parameters():
        if "lora_" not in name:
            continue
        n_total += 1

        keep = True
        if allowed_modules is not None and len(allowed_modules) > 0:
            keep = any(f".{m}.lora_" in name for m in allowed_modules)

        if keep and allowed_layers is not None:
            keep = any(f".layers.{lid}." in name for lid in allowed_layers)

        param.requires_grad = keep
        if keep:
            n_active += 1

    print(f"Applied trainable filter: active_lora_tensors={n_active}/{n_total}")


def _load_full_path_lora_trainable(base_model, init_path: Path):
    """Load DRT-style full-path LoRA weights into a trainable PEFT model."""
    from safetensors import safe_open

    cfg_path = init_path / "adapter_config.json"
    st_path = init_path / "adapter_model.safetensors"
    if not cfg_path.exists() or not st_path.exists():
        return None

    with open(cfg_path, "r", encoding="utf-8-sig") as f:
        cfg = json.load(f)

    with safe_open(str(st_path), framework="pt", device="cpu") as sf:
        keys = list(sf.keys())
        has_full_path = any("model.layers." in k and ".lora_" in k for k in keys)
        if not has_full_path:
            return None
        tensors = {k: sf.get_tensor(k) for k in keys}

    print("Detected full-path LoRA keys, using direct trainable init load.")
    lora_cfg = LoraConfig(
        r=cfg.get("r", 32),
        lora_alpha=cfg.get("lora_alpha", 32),
        target_modules=cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        lora_dropout=cfg.get("lora_dropout", 0.0),
        bias=cfg.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
        layers_to_transform=cfg.get("layers_to_transform", None),
    )
    model = get_peft_model(base_model, lora_cfg)
    missing, unexpected = model.load_state_dict(tensors, strict=False)
    if unexpected:
        print(f"Warning: unexpected keys when loading init adapter: {len(unexpected)}")
    if missing:
        print(f"Warning: missing keys when loading init adapter: {len(missing)}")
    return model


def setup_lora_model(args):
    print(f"Loading baseline model from: {args.model_name_or_path}")
    
    # Auto-detect if we are running in distributed mode (e.g. torchrun) for Multi-GPU
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        device_map = {"": local_rank}
        print(f"Detected Distributed Training! Binding model strictly to Local Rank GPU: {local_rank}")
    else:
        device_map = "auto"
        print("Running in Standard/Single-GPU or Model Parallel Mode. device_map='auto'")

    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.init_lora_path:
        init_path = Path(args.init_lora_path)
        if not init_path.exists():
            raise FileNotFoundError(f"init_lora_path not found: {init_path}")

        print(f"Loading trainable adapter init from: {init_path}")
        full_path_loaded = _load_full_path_lora_trainable(model, init_path)
        if full_path_loaded is not None:
            model = full_path_loaded
        else:
            model = PeftModel.from_pretrained(model, str(init_path), is_trainable=True)
    else:
        print(f"Injecting LoRA adapters... (r={args.lora_r}, alpha={args.lora_alpha}, target_modules={args.lora_target_modules})")
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, config)

    _apply_trainable_filter(model, args)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Enabled gradient checkpointing.")

    model.print_trainable_parameters()
    
    return model, tokenizer
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--forget_data", type=str, required=True)
    parser.add_argument("--retain_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--init_lora_path", type=str, default="",
                        help="Optional adapter path used to initialize LoRA weights before GA/GA+GD training.")
    
    # Method configuration
    parser.add_argument("--method", type=str, default="ga", choices=["ga", "ga_gd"], 
                        help="Unlearning method. 'ga' is simple Gradient Ascent. 'ga_gd' adds Retain loss.")
    parser.add_argument("--gamma", type=float, default=1.0, help="Weight for forget loss (GA).")
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for retain loss (GD) in ga_gd method.")

    # General Training Hyperparams
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Steps to accumulate gradients before taking an optimizer step.")
    parser.add_argument("--eval_batch_size", type=int, default=2, help="Batch size during evaluation.")
    parser.add_argument("--max_length", type=int, default=1024, help="Max sequence length for training/eval examples.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing to reduce VRAM.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])

    # Exposed LoRA Hyperparams
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"],
                        help="List of module names to apply LoRA on, e.g., q_proj k_proj")
    parser.add_argument("--lora_bias", type=str, default="none", choices=["none", "all", "lora_only"])
    parser.add_argument("--train_only_modules", type=str, nargs="+", default=None,
                        help="Optional module filter for trainable LoRA tensors, e.g., q_proj o_proj")
    parser.add_argument("--train_only_layers", type=str, default="",
                        help="Optional comma-separated layer ids to train, e.g., 0,5,10,16")

    parser.add_argument("--run_name", type=str, default="unlearn_lora")
    parser.add_argument("--eval_steps", type=int, default=1)
    
    # Save checkpoint frequency logic
    parser.add_argument("--save_steps", type=int, default=None, 
                        help="Save checkpoint every X steps. If None, saves 10 checkpoints across the whole training via calculation.")
    parser.add_argument("--save_total_limit", type=int, default=10, 
                        help="Max checkpoints to keep. Defaults to 10.")
    
    args = parser.parse_args()

    wandb.init(project="lora-unlearning", name=args.run_name, config=vars(args))

    model, tokenizer = setup_lora_model(args)
    
    with open(args.forget_data, 'r', encoding='utf-8') as f:
        forget_list = json.load(f)
    with open(args.retain_data, 'r', encoding='utf-8') as f:
        retain_list = json.load(f)

    # Wrap data appropriately for the chosen method
    if args.method == "ga_gd":
        print(f"Using GA+GD Training. Mapping {len(forget_list)} forget samples against {len(retain_list)} retain samples.")
        train_dataset = JointDataset(forget_list, retain_list)
        dataset_len = len(forget_list)
    else:
        print(f"Using standard GA Training on {len(forget_list)} forget samples.")
        train_dataset = Dataset.from_list(forget_list)
        dataset_len = len(forget_list)

    # Need to account for multiple GPUs in total steps calculation if not handled internally
    num_processes = int(os.environ.get("WORLD_SIZE", 1))
    total_steps = math.ceil((dataset_len / (args.batch_size * args.gradient_accumulation_steps)) / num_processes) * args.epochs
    save_steps = args.save_steps if args.save_steps is not None else max(1, total_steps // 10)
    print(f"Total training steps ~ {total_steps}. Will save checkpoint every {save_steps} steps.")

    # Save training args strictly to output_dir to be used in judge eval later
    # Only save on main process to avoid collision
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "training_args.json"), "w", encoding='utf-8') as f:
            json.dump(vars(args), f, indent=4, ensure_ascii=False)

    eval_forget = Dataset.from_list(forget_list)
    eval_retain = Dataset.from_list(retain_list)

    collator = DataCollatorForForgetDataset(tokenizer, max_length=args.max_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=args.save_total_limit,
        report_to="wandb" if int(os.environ.get("LOCAL_RANK", 0)) == 0 else "none",
        bf16=True,
        # Crucial for GA+GD since we have custom dict inputs
        remove_unused_columns=False, 
        ddp_find_unused_parameters=False, # Required for some PEFT + DDP scenarios
    )

    trainer = UnlearnTrainer(
        method=args.method,
        gamma=args.gamma,
        alpha=args.alpha,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset={
            "forget": eval_forget,
            "retain": eval_retain
        },
        data_collator=collator,
    )

    print(f"Starting Training ({args.method})...")
    trainer.train()
    
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
