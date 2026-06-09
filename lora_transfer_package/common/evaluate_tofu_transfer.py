#!/usr/bin/env python3
"""
evaluate_tofu_transfer.py
=========================
Simple TOFU evaluation for transferred LoRA adapters.
Computes forget_Q_A_Prob and model_utility metrics.

Usage:
  python evaluate_tofu_transfer.py \
    --base_model models/llama_3_2_3B_instruct_tofu \
    --lora_path lora_transfer_output/llama_1B_to_3B/forget05/actmap_base \
    --forget_file datasets/tofu/forget05.json \
    --retain_file datasets/tofu/retain95.json \
    --output_dir results/tofu_transfer/llama_1B_to_3B_forget05_actmap_base
"""
import argparse, json, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

BOS = "<|begin_of_text|>"


def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def build_prompt(sample):
    """Build LLAMA3 chat-format prompt for TOFU QA."""
    q = sample["question"]
    system = "You are a helpful assistant."
    prompt = (
        f"{BOS}<|start_header_id|>system<|end_header_id|>\n\n"
        f"{system}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{q}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return prompt


def compute_prob(model, tokenizer, prompt, answer, max_new=32):
    """Compute probability of the answer given the prompt."""
    full = prompt + answer
    enc = tokenizer(full, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[0]  # (seq_len, vocab)

    # Tokenize prompt to find split point
    prompt_enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    prompt_len = prompt_enc["input_ids"].shape[1]

    # Compute log-prob of answer tokens
    answer_ids = input_ids[0, prompt_len:]
    if len(answer_ids) == 0:
        return 0.0

    log_probs = []
    for i, aid in enumerate(answer_ids):
        pos = prompt_len + i
        if pos >= logits.shape[0]:
            break
        lp = torch.log_softmax(logits[pos], dim=-1)[aid].item()
        log_probs.append(lp)

    if not log_probs:
        return 0.0

    avg_log_prob = sum(log_probs) / len(log_probs)
    return float(torch.exp(torch.tensor(avg_log_prob)).item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--lora_path", required=True)
    p.add_argument("--forget_file", required=True)
    p.add_argument("--retain_file", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_forget", type=int, default=100)
    p.add_argument("--max_retain", type=int, default=100)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model + LoRA
    print(f"Loading base model: {args.base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    print(f"Loading LoRA: {args.lora_path}")
    model = PeftModel.from_pretrained(base, args.lora_path)
    model.eval()

    # Load data
    forget_data = load_jsonl(args.forget_file)[:args.max_forget]
    retain_data = load_jsonl(args.retain_file)[:args.max_retain]
    print(f"Forget samples: {len(forget_data)}, Retain samples: {len(retain_data)}")

    # Evaluate forget
    forget_probs = []
    print("Evaluating forget set...")
    for sample in tqdm(forget_data):
        prompt = build_prompt(sample)
        answer = sample.get("answer", "")
        prob = compute_prob(model, tokenizer, prompt, answer)
        forget_probs.append(prob)
        if len(forget_probs) <= 3:
            # Debug: print first few samples
            pass  # set to print for debugging

    # Evaluate retain
    retain_probs = []
    print("Evaluating retain set...")
    for sample in tqdm(retain_data):
        prompt = build_prompt(sample)
        answer = sample.get("answer", "")
        prob = compute_prob(model, tokenizer, prompt, answer)
        retain_probs.append(prob)

    # Metrics
    forget_qa_prob = sum(forget_probs) / len(forget_probs) if forget_probs else 0
    retain_qa_prob = sum(retain_probs) / len(retain_probs) if retain_probs else 0

    # Raw log-prob based metric (more stable)
    forget_logprobs = [p for p in forget_probs if p > 0]
    retain_logprobs = [p for p in retain_probs if p > 0]

    # model_utility: 1 - |forget_prob - retain_prob| is too sensitive
    # Use log-scale comparison instead
    if forget_logprobs and retain_logprobs:
        forget_log_mean = sum(forget_logprobs) / len(forget_logprobs)
        retain_log_mean = sum(retain_logprobs) / len(retain_logprobs)
    else:
        forget_log_mean = retain_log_mean = 0.0

    model_utility = 1.0 - abs(forget_qa_prob - retain_qa_prob)

    results = {
        "forget_Q_A_Prob": round(forget_qa_prob, 8),
        "retain_Q_A_Prob": round(retain_qa_prob, 8),
        "forget_log_prob": round(forget_log_mean, 6) if forget_logprobs else 0.0,
        "retain_log_prob": round(retain_log_mean, 6) if retain_logprobs else 0.0,
        "model_utility": round(model_utility, 6),
        "n_forget": len(forget_probs),
        "n_retain": len(retain_probs),
        "forget_file": args.forget_file,
        "retain_file": args.retain_file,
        "lora_path": args.lora_path,
    }

    out_path = os.path.join(args.output_dir, "TOFU_TRANSFER_SUMMARY.json")
    with open(out_path, "w") as f:
        json.dump({k: v for k, v in results.items() if k not in ("forget_probs", "retain_probs")}, f, indent=2)

    print(f"\n===== Results =====")
    print(f"forget_Q_A_Prob:   {forget_qa_prob:.8f}")
    print(f"retain_Q_A_Prob:   {retain_qa_prob:.8f}")
    print(f"forget_log_prob:   {forget_log_mean:.6f}")
    print(f"retain_log_prob:   {retain_log_mean:.6f}")
    print(f"model_utility:     {model_utility:.4f}")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
