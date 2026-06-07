#!/usr/bin/env python3
"""
Stage 2 简化版：使用HF Trainer进行LoRA微调
"""

import json
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import PeftModel
from datasets import Dataset
import argparse

def main(
    base_model,
    adapter_path,
    forget_file,
    retain_file,
    output_dir,
    learning_rate=5e-5,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    max_samples=None,
):
    print("="*70)
    print(f"Stage 2 LoRA 微调 (简化版)")
    print(f"LR={learning_rate}, Epochs={num_train_epochs}")
    print("="*70)
    
    # 加载数据
    print("\n[1/4] 加载数据...")
    with open(forget_file) as f:
        forget_data = json.load(f)
    with open(retain_file) as f:
        retain_data = json.load(f)
    
    # 提取文本
    forget_texts = [d['text'] if isinstance(d, dict) else d for d in forget_data]
    retain_texts = [d['text'] if isinstance(d, dict) else d for d in retain_data]
    
    if max_samples:
        forget_texts = forget_texts[:max_samples]
        retain_texts = retain_texts[:max_samples]
    
    # 简单策略：使用"反向"样本作为遗忘目标
    # 遗忘集：让模型学到相反的回答（通过负采样或掩码）
    # 保留集：正常训练
    all_texts = retain_texts  # 先从保留集训练作为基础
    
    print(f"  使用保留集: {len(retain_texts)} 样本")
    print(f"  (遗忘集将在后续迭代中处理)")
    
    # 加载模型和分词器
    print("\n[2/4] 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True,
    )
    
    # 加载 top8 LoRA
    print(f"  加载 LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.train()
    
    # 准备数据集
    print("\n[3/4] 准备数据集...")
    def tokenize_function(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
        )
        return result
    
    dataset = Dataset.from_dict({"text": all_texts})
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        save_strategy="epoch",
        logging_steps=5,
        report_to=[],  # 不上传到wandb
        dataloader_drop_last=False,
        optim="paged_adamw_8bit",
    )
    
    # Trainer
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # 训练
    print("\n[4/4] 训练...")
    trainer.train()
    
    # 保存
    print(f"\n✓ 微调完成")
    print(f"  模型保存到: {output_dir}")
    
    # 记录配置
    config = {
        'base_model': base_model,
        'adapter_path': adapter_path,
        'learning_rate': learning_rate,
        'num_train_epochs': num_train_epochs,
        'batch_size': per_device_train_batch_size,
        'num_samples': len(retain_texts),
    }
    with open(os.path.join(output_dir, 'stage2_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default='modelzoo/qwen3_8B')
    parser.add_argument('--adapter_path', default='trained_models/xTransform/qwen3_8B_drt_as16_r5950_l02_lmapRw02_ns0_top8')
    parser.add_argument('--forget_file', default='datasets/processed_forget_sets/1.7B_to_8B_transfer/wmdp-cyber/forget_10pct.json')
    parser.add_argument('--retain_file', default='datasets/processed_forget_sets/1.7B_to_8B_transfer/wmdp-cyber/remain_10pct.json')
    parser.add_argument('--output_dir', default='trained_models/xTransform/qwen3_8B_stage2_retain_finetune')
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--num_train_epochs', type=int, default=2)
    parser.add_argument('--per_device_train_batch_size', type=int, default=2)
    
    args = parser.parse_args()
    main(**vars(args))
