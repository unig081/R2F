#!/usr/bin/env python3
"""
Stage 2: LoRA 微调
目标：在top8转移(77.97%遗忘, 72% WMDP保留)基础上微调
      → 降低到~50%遗忘, 维持~70% WMDP保留

策略：
1. 使用混合损失: L = λ_f * L_forget + λ_r * L_retain
2. L_forget = 1 - (遗忘集的模型预测正确率)
3. L_retain = (保留集的模型预测错误率)
4. 参数扫描: λ_f ∈ {0.3, 0.5, 0.8}, λ_r ∈ {0.2, 0.5}
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import argparse
import numpy as np
from datetime import datetime

class UnlearningDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        text = self.samples[idx]
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
        }

def get_accuracy(logits, labels, k=1):
    """计算top-k准确率"""
    with torch.no_grad():
        topk_indices = torch.topk(logits, k, dim=-1)[1]
        labels_expanded = labels.unsqueeze(1).expand(-1, k)
        correct = (topk_indices == labels_expanded).any(dim=1)
        return correct.float().mean().item()

def train_lora_stage2(
    base_model_name,
    adapter_path,  # top8转移的LoRA
    forget_file,
    retain_file,
    output_dir,
    lambda_forget=0.5,
    lambda_retain=0.3,
    learning_rate=1e-4,
    epochs=3,
    batch_size=4,
    max_samples=None,
):
    """Stage 2 微调主函数"""
    
    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*70)
    print(f"Stage 2 LoRA 微调")
    print(f"λ_forget={lambda_forget}, λ_retain={lambda_retain}, LR={learning_rate}")
    print("="*70)
    
    # 加载数据
    print("\n[1/5] 加载数据...")
    with open(forget_file) as f:
        forget_data = json.load(f)
    with open(retain_file) as f:
        retain_data = json.load(f)
    
    forget_samples = [d['text'] if isinstance(d, dict) else d for d in forget_data]
    retain_samples = [d['text'] if isinstance(d, dict) else d for d in retain_data]
    
    if max_samples:
        forget_samples = forget_samples[:max_samples]
        retain_samples = retain_samples[:max_samples]
    
    print(f"  遗忘集: {len(forget_samples)}")
    print(f"  保留集: {len(retain_samples)}")
    
    # 加载模型
    print("\n[2/5] 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if 'cuda' in device else torch.float32,
        device_map='auto',
        trust_remote_code=True,
    )
    
    # 加载top8 LoRA
    print(f"  加载top8 LoRA from {adapter_path}...")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, adapter_path)
    model.train()
    
    # 创建数据加载器
    print("\n[3/5] 创建数据加载器...")
    forget_dataset = UnlearningDataset(forget_samples, tokenizer)
    retain_dataset = UnlearningDataset(retain_samples, tokenizer)
    
    forget_loader = DataLoader(forget_dataset, batch_size=batch_size, shuffle=True)
    retain_loader = DataLoader(retain_dataset, batch_size=batch_size, shuffle=True)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*len(forget_loader))
    
    # 训练循环
    print("\n[4/5] 微调...")
    history = {'epoch': [], 'loss': [], 'forget_acc': [], 'retain_acc': []}
    
    for epoch in range(epochs):
        epoch_loss = 0
        forget_accs = []
        retain_accs = []
        
        # 遗忘集和保留集交替
        forget_iter = iter(forget_loader)
        retain_iter = iter(retain_loader)
        
        with tqdm(total=max(len(forget_loader), len(retain_loader)), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for step in range(max(len(forget_loader), len(retain_loader))):
                optimizer.zero_grad()
                batch_loss = 0
                
                # 遗忘集损失
                try:
                    forget_batch = next(forget_iter)
                except StopIteration:
                    forget_iter = iter(forget_loader)
                    forget_batch = next(forget_iter)
                
                input_ids = forget_batch['input_ids'].to(device)
                attention_mask = forget_batch['attention_mask'].to(device)
                
                with torch.autocast(device_type='cuda' if 'cuda' in device else 'cpu'):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits[:, :-1, :]  # [B, L-1, vocab]
                    labels = input_ids[:, 1:]  # [B, L-1]
                    
                    # 计算遗忘损失: 最小化正确预测概率
                    # 即: 最大化错误预测概率
                    probs = F.softmax(logits, dim=-1)
                    correct_probs = probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
                    forget_loss = torch.clamp(correct_probs, min=0.1).mean()  # 下界0.1避免梯度爆炸
                    
                    batch_loss += lambda_forget * forget_loss
                
                # 保留集损失
                try:
                    retain_batch = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(retain_loader)
                    retain_batch = next(retain_iter)
                
                input_ids = retain_batch['input_ids'].to(device)
                attention_mask = retain_batch['attention_mask'].to(device)
                
                with torch.autocast(device_type='cuda' if 'cuda' in device else 'cpu'):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits[:, :-1, :]
                    labels = input_ids[:, 1:]
                    
                    # 计算保留损失: 标准的CLM损失
                    retain_loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        labels.reshape(-1),
                        reduction='mean'
                    )
                    
                    batch_loss += lambda_retain * retain_loss
                
                # 反向传播
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                epoch_loss += batch_loss.item()
                pbar.update(1)
        
        avg_loss = epoch_loss / max(len(forget_loader), len(retain_loader))
        print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}")
        
        history['epoch'].append(epoch + 1)
        history['loss'].append(avg_loss)
    
    print("\n[5/5] 保存模型...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # 保存训练配置
    config = {
        'base_model': base_model_name,
        'adapter_path': adapter_path,
        'lambda_forget': lambda_forget,
        'lambda_retain': lambda_retain,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'batch_size': batch_size,
        'history': history,
        'timestamp': datetime.now().isoformat(),
    }
    with open(os.path.join(output_dir, 'stage2_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✓ 微调完成，模型保存到: {output_dir}")
    return output_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default='modelzoo/qwen3_8B')
    parser.add_argument('--adapter_path', default='trained_models/xTransform/qwen3_8B_drt_as16_r5950_l02_lmapRw02_ns0_top8')
    parser.add_argument('--forget_file', default='datasets/processed_forget_sets/1.7B_to_8B_transfer/wmdp-cyber/forget_10pct.json')
    parser.add_argument('--retain_file', default='datasets/processed_forget_sets/1.7B_to_8B_transfer/wmdp-cyber/remain_10pct.json')
    parser.add_argument('--output_dir', default='trained_models/xTransform/qwen3_8B_stage2_finetune_test')
    parser.add_argument('--lambda_forget', type=float, default=0.5)
    parser.add_argument('--lambda_retain', type=float, default=0.3)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4)
    
    args = parser.parse_args()
    
    train_lora_stage2(
        base_model_name=args.base_model,
        adapter_path=args.adapter_path,
        forget_file=args.forget_file,
        retain_file=args.retain_file,
        output_dir=args.output_dir,
        lambda_forget=args.lambda_forget,
        lambda_retain=args.lambda_retain,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
