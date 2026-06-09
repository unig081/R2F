#!/usr/bin/env python3
"""
Stage 2 LoRA 微调 - 混合损失
目标: 在 DRT top8 基础上，通过混合损失微调达到 ~50% 遗忘, ~70% WMDP保留

损失函数:
  L_total = λ_f * L_forget + λ_r * L_retain
  
  L_forget = -log(遗忘集的错误率) = 目标让遗忘集出错
  L_retain = cross_entropy(保留集) = 保持保留集知识
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import json
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
from safetensors.torch import load_file


def record_to_text(record):
    """Normalize different JSON record schemas to a single training text string."""
    if isinstance(record, str):
        return record
    if not isinstance(record, dict):
        return str(record)

    if 'text' in record and isinstance(record['text'], str):
        return record['text']
    if 'prompt_used' in record and isinstance(record['prompt_used'], str):
        return record['prompt_used']
    if 'question' in record and isinstance(record['question'], str):
        answer = ''
        if 'gold_answers' in record and isinstance(record['gold_answers'], list) and len(record['gold_answers']) > 0:
            answer = str(record['gold_answers'][0])
        elif 'gold_option_text' in record and isinstance(record['gold_option_text'], str):
            answer = record['gold_option_text']
        if answer:
            return f"{record['question']}\n\nAnswer: {answer}"
        return record['question']

    # fallback for unknown schemas
    return json.dumps(record, ensure_ascii=False)


def remap_fullpath_lora_keys_for_peft(state_dict):
    """Map full-path LoRA keys to PEFT adapter key format with .default suffix."""
    mapped = {}
    for k, v in state_dict.items():
        new_k = k
        if new_k.endswith('.lora_A.weight'):
            new_k = new_k.replace('.lora_A.weight', '.lora_A.default.weight')
        elif new_k.endswith('.lora_B.weight'):
            new_k = new_k.replace('.lora_B.weight', '.lora_B.default.weight')
        mapped[new_k] = v
    return mapped

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
        }

def collate_fn(batch):
    """Custom collate to handle variable length sequences"""
    max_len = max(x['input_ids'].shape[0] for x in batch)
    
    input_ids = []
    attention_mask = []
    
    for item in batch:
        ids = item['input_ids']
        mask = item['attention_mask']
        
        # Pad
        pad_len = max_len - len(ids)
        ids = F.pad(ids, (0, pad_len), value=0)
        mask = F.pad(mask, (0, pad_len), value=0)
        
        input_ids.append(ids)
        attention_mask.append(mask)
    
    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_mask),
    }

def train_stage2(
    base_model_name,
    adapter_path,
    forget_file,
    retain_file,
    output_dir,
    lambda_forget=0.5,
    lambda_retain=0.5,
    learning_rate=1e-4,
    num_epochs=2,
    batch_size=2,
    max_samples=None,
    warmup_steps=100,
):
    """Stage 2 混合损失微调"""
    
    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    print("="*70)
    print(f"Stage 2 LoRA Finetune - 混合损失")
    print(f"λ_f={lambda_forget}, λ_r={lambda_retain}, LR={learning_rate}")
    print(f"Base adapter: {adapter_path}")
    print("="*70)
    
    # 加载数据
    print("\n[1/5] 加载数据...")
    with open(forget_file, encoding='utf-8') as f:
        forget_data = json.load(f)
    with open(retain_file, encoding='utf-8') as f:
        retain_data = json.load(f)
    
    forget_texts = [record_to_text(d) for d in forget_data]
    retain_texts = [record_to_text(d) for d in retain_data]
    
    if max_samples:
        forget_texts = forget_texts[:max_samples]
        retain_texts = retain_texts[:max_samples]
    
    print(f"  遗忘集: {len(forget_texts)} 样本")
    print(f"  保留集: {len(retain_texts)} 样本")
    
    # 加载模型
    print("\n[2/5] 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True,
    )
    
    # 加载 top8 LoRA
    print(f"  加载 LoRA adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True)
    # Some transferred adapters are saved with full-path LoRA keys.
    # Load them manually after remapping to PEFT's expected key format.
    adapter_weights_path = os.path.join(adapter_path, 'adapter_model.safetensors')
    if os.path.exists(adapter_weights_path):
        raw_sd = load_file(adapter_weights_path)
        remapped_sd = remap_fullpath_lora_keys_for_peft(raw_sd)
        missing, unexpected = model.load_state_dict(remapped_sd, strict=False)
        if len(missing) > 0:
            print(f"  [warn] Missing keys after remap: {len(missing)}")
        if len(unexpected) > 0:
            print(f"  [warn] Unexpected keys after remap: {len(unexpected)}")
    model.train()
    
    # 数据集和加载器
    print("\n[3/5] 准备数据...")
    forget_dataset = TextDataset(forget_texts, tokenizer)
    retain_dataset = TextDataset(retain_texts, tokenizer)
    
    forget_loader = DataLoader(
        forget_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_fn,
    )
    retain_loader = DataLoader(
        retain_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    
    # 优化器
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"  可训练参数张量数: {len(trainable_params)}")
    if len(trainable_params) == 0:
        raise RuntimeError('No trainable parameters found after adapter loading.')
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=0.01,
    )
    total_steps = num_epochs * max(len(forget_loader), len(retain_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    # 训练循环
    print("\n[4/5] 微调...")
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        pbar = tqdm(total=max(len(forget_loader), len(retain_loader)), desc=f"Epoch {epoch+1}/{num_epochs}")
        
        forget_iter = iter(forget_loader)
        retain_iter = iter(retain_loader)
        
        for step in range(max(len(forget_loader), len(retain_loader))):
            optimizer.zero_grad()
            
            loss = 0.0
            
            # 遗忘集损失: 最小化正确率 = 最大化错误率
            try:
                forget_batch = next(forget_iter)
            except StopIteration:
                forget_iter = iter(forget_loader)
                forget_batch = next(forget_iter)
            
            with torch.autocast(device_type='cuda' if 'cuda' in device else 'cpu'):
                forget_input_ids = forget_batch['input_ids'].to(device)
                forget_mask = forget_batch['attention_mask'].to(device)
                
                forget_outputs = model(
                    input_ids=forget_input_ids,
                    attention_mask=forget_mask,
                )
                forget_logits = forget_outputs.logits[:, :-1, :]
                forget_labels = forget_input_ids[:, 1:]
                
                # L_forget = -log(错误率) = log(正确率) 的反向
                # 更直接: 我们想最大化错误，即最小化置信度
                forget_probs = F.softmax(forget_logits, dim=-1)
                forget_correct_probs = forget_probs.gather(-1, forget_labels.unsqueeze(-1)).squeeze(-1)
                
                # 掩码处理
                forget_mask_2d = forget_mask[:, 1:]
                forget_correct_probs = forget_correct_probs * forget_mask_2d.float()
                valid_count = forget_mask_2d.sum()
                
                if valid_count > 0:
                    # 目标: 让模型在遗忘集上出错 -> 最小化正确 token 概率
                    forget_loss = torch.clamp(forget_correct_probs.sum() / valid_count, min=1e-6)
                    loss += lambda_forget * forget_loss
            
            # 保留集损失: 标准CLM损失
            try:
                retain_batch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                retain_batch = next(retain_iter)
            
            with torch.autocast(device_type='cuda' if 'cuda' in device else 'cpu'):
                retain_input_ids = retain_batch['input_ids'].to(device)
                retain_mask = retain_batch['attention_mask'].to(device)
                
                retain_outputs = model(
                    input_ids=retain_input_ids,
                    attention_mask=retain_mask,
                )
                retain_logits = retain_outputs.logits[:, :-1, :]
                retain_labels = retain_input_ids[:, 1:]
                
                # 标准CLM损失
                retain_loss = F.cross_entropy(
                    retain_logits.reshape(-1, retain_logits.size(-1)),
                    retain_labels.reshape(-1),
                    reduction='mean',
                )
                loss += lambda_retain * retain_loss
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            pbar.update(1)
        
        avg_loss = epoch_loss / max(len(forget_loader), len(retain_loader))
        print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save_pretrained(output_dir)
            print(f"  ✓ 保存最优模型")
    
    print("\n[5/5] 完成")
    print(f"✓ 微调完成，模型保存到: {output_dir}")
    
    # 保存配置
    config = {
        'base_model': base_model_name,
        'adapter_path': adapter_path,
        'lambda_forget': lambda_forget,
        'lambda_retain': lambda_retain,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'final_loss': best_loss,
    }
    with open(os.path.join(output_dir, 'stage2_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default='modelzoo/qwen3_8B')
    parser.add_argument('--adapter_path', default='trained_models/xTransform/qwen3_8B_drt_as16_r5950_l02_lmapRw02_ns0_top8')
    parser.add_argument('--forget_file', default='datasets/processed_forget_sets/1.7B_to_8B_transfer/wmdp-cyber/forget_10pct.json')
    parser.add_argument('--retain_file', default='datasets/processed_forget_sets/1.7B_to_8B_transfer/wmdp-cyber/remain_10pct.json')
    parser.add_argument('--output_dir', default='trained_models/xTransform/qwen3_8B_stage2_hybrid')
    parser.add_argument('--lambda_forget', type=float, default=0.5)
    parser.add_argument('--lambda_retain', type=float, default=0.5)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--max_samples', type=int, default=None)
    
    args = parser.parse_args()
    train_stage2(
        base_model_name=args.base_model,
        adapter_path=args.adapter_path,
        forget_file=args.forget_file,
        retain_file=args.retain_file,
        output_dir=args.output_dir,
        lambda_forget=args.lambda_forget,
        lambda_retain=args.lambda_retain,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )
