#!/usr/bin/env python3
"""Stage 2 (MCQ-targeted): only optimize answer-letter token at `Answer:` position.

This avoids global language collapse caused by sequence-wide forget loss.
"""

import argparse
import json
import math
import os
from typing import List, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from safetensors.torch import load_file


def load_mcq_records(path: str) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    out = []
    for r in raw:
        if not isinstance(r, dict):
            continue
        prompt = r.get("prompt_used") or r.get("question")
        ans_list = r.get("gold_answers")
        if isinstance(ans_list, list) and len(ans_list) > 0:
            ans = str(ans_list[0]).strip()
        else:
            ans = None
        if prompt and ans in {"A", "B", "C", "D"}:
            # ensure prompt ends with Answer: cue for next-token training
            if "Answer:" not in prompt:
                prompt = prompt.rstrip() + "\n\nAnswer: "
            out.append({"prompt": prompt, "answer": ans})
    return out


class MCQDataset(Dataset):
    def __init__(self, rows: List[Dict[str, str]], tokenizer, max_length: int = 512):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        enc = self.tokenizer(
            row["prompt"],
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "answer": row["answer"],
        }


def collate_mcq(batch):
    max_len = max(x["input_ids"].shape[0] for x in batch)
    ids, masks, answers = [], [], []
    for x in batch:
        pad = max_len - x["input_ids"].shape[0]
        ids.append(F.pad(x["input_ids"], (0, pad), value=0))
        masks.append(F.pad(x["attention_mask"], (0, pad), value=0))
        answers.append(x["answer"])
    return {
        "input_ids": torch.stack(ids),
        "attention_mask": torch.stack(masks),
        "answers": answers,
    }


def remap_and_load_adapter(model, adapter_path: str):
    sd_path = os.path.join(adapter_path, "adapter_model.safetensors")
    if not os.path.exists(sd_path):
        return
    sd = load_file(sd_path)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  adapter manual load -> missing={len(missing)}, unexpected={len(unexpected)}")


def answer_token_ids(tokenizer):
    # prompt already ends with a trailing space after `Answer:`
    return {k: tokenizer.encode(k, add_special_tokens=False)[0] for k in ["A", "B", "C", "D"]}


@torch.no_grad()
def probe_answer_accuracy(model, loader, ans2id, device: str) -> float:
    model.eval()
    correct, total = 0, 0
    for b in loader:
        ids = b["input_ids"].to(device)
        mask = b["attention_mask"].to(device)
        out = model(input_ids=ids, attention_mask=mask)
        logits = out.logits
        pos = mask.sum(dim=1) - 1
        last = logits[torch.arange(logits.size(0), device=logits.device), pos, :]
        pred = last.argmax(dim=-1)
        tgt = torch.tensor([ans2id[a] for a in b["answers"]], device=last.device)
        correct += int((pred == tgt).sum().item())
        total += int(tgt.numel())
    model.train()
    return float(correct / max(total, 1))


def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("Stage2 MCQ-targeted Finetune")
    print(f"lambda_forget={args.lambda_forget}, lambda_retain={args.lambda_retain}, lr={args.learning_rate}")
    print("=" * 70)

    print("\n[1/5] Load data")
    forget_rows = load_mcq_records(args.forget_file)
    retain_rows = load_mcq_records(args.retain_file)
    if args.max_samples is not None:
        forget_rows = forget_rows[: args.max_samples]
        retain_rows = retain_rows[: args.max_samples]
    print(f"  forget={len(forget_rows)}, retain={len(retain_rows)}")

    print("\n[2/5] Load model + adapter")
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter_path, is_trainable=True)
    remap_and_load_adapter(model, args.adapter_path)
    model.train()

    ans2id = answer_token_ids(tok)
    print(f"  answer_token_ids={ans2id}")

    print("\n[3/5] Build loaders")
    f_loader = DataLoader(MCQDataset(forget_rows, tok, args.max_length), batch_size=args.batch_size, shuffle=True, collate_fn=collate_mcq)
    r_loader = DataLoader(MCQDataset(retain_rows, tok, args.max_length), batch_size=args.batch_size, shuffle=True, collate_fn=collate_mcq)

    probe_forget_rows = forget_rows[: min(args.probe_samples, len(forget_rows))]
    probe_retain_rows = retain_rows[: min(args.probe_samples, len(retain_rows))]
    pf_loader = DataLoader(MCQDataset(probe_forget_rows, tok, args.max_length), batch_size=args.batch_size, shuffle=False, collate_fn=collate_mcq)
    pr_loader = DataLoader(MCQDataset(probe_retain_rows, tok, args.max_length), batch_size=args.batch_size, shuffle=False, collate_fn=collate_mcq)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"  trainable_tensors={len(trainable)}")
    if not trainable:
        raise RuntimeError("No trainable parameters")

    opt = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)
    total_steps = args.num_epochs * max(len(f_loader), len(r_loader))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total_steps, 1))

    auto_forget_interval = max(1, len(r_loader) // max(1, len(f_loader)))
    forget_interval = args.forget_interval if args.forget_interval is not None else auto_forget_interval
    print(f"  forget_update_interval={forget_interval}")

    print("\n[4/5] Train")
    best_score = -1e9
    bad_probe_steps = 0
    for ep in range(args.num_epochs):
        ep_loss = 0.0
        f_it, r_it = iter(f_loader), iter(r_loader)
        steps = len(r_loader)
        pbar = tqdm(total=steps, desc=f"Epoch {ep+1}/{args.num_epochs}")

        for step_idx in range(steps):
            opt.zero_grad()
            loss = 0.0
            try:
                rb = next(r_it)
            except StopIteration:
                r_it = iter(r_loader)
                rb = next(r_it)

            r_ids = rb["input_ids"].to(device)
            r_mask = rb["attention_mask"].to(device)

            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                if step_idx % forget_interval == 0:
                    try:
                        fb = next(f_it)
                    except StopIteration:
                        f_it = iter(f_loader)
                        fb = next(f_it)

                    # Forget objective: minimize gold answer probability at answer position
                    f_ids = fb["input_ids"].to(device)
                    f_mask = fb["attention_mask"].to(device)
                    f_out = model(input_ids=f_ids, attention_mask=f_mask)
                    f_logits = f_out.logits  # [B, L, V]
                    f_pos = f_mask.sum(dim=1) - 1  # last real token position
                    f_last = f_logits[torch.arange(f_logits.size(0), device=f_logits.device), f_pos, :]
                    f_tgt = torch.tensor([ans2id[a] for a in fb["answers"]], device=f_last.device)
                    f_prob = F.softmax(f_last, dim=-1).gather(-1, f_tgt.unsqueeze(-1)).squeeze(-1)
                    forget_loss = torch.clamp(f_prob.mean(), min=1e-6)
                    loss = loss + args.lambda_forget * forget_loss

                # Retain objective: CE on gold answer token at answer position
                r_out = model(input_ids=r_ids, attention_mask=r_mask)
                r_logits = r_out.logits
                r_pos = r_mask.sum(dim=1) - 1
                r_last = r_logits[torch.arange(r_logits.size(0), device=r_logits.device), r_pos, :]
                r_tgt = torch.tensor([ans2id[a] for a in rb["answers"]], device=r_last.device)
                retain_loss = F.cross_entropy(r_last, r_tgt)
                loss = loss + args.lambda_retain * retain_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sch.step()

            ep_loss += float(loss.item())
            pbar.update(1)

            if (step_idx + 1) % args.probe_every == 0 or (step_idx + 1) == steps:
                f_acc = probe_answer_accuracy(model, pf_loader, ans2id, device)
                r_acc = probe_answer_accuracy(model, pr_loader, ans2id, device)
                score = -abs(f_acc - args.target_forget_acc) + args.retain_score_weight * r_acc
                pbar.set_postfix({"probe_f": f"{f_acc:.2f}", "probe_r": f"{r_acc:.2f}", "score": f"{score:.3f}"})

                if r_acc >= args.min_retain_probe_acc and score > best_score:
                    best_score = score
                    model.save_pretrained(args.output_dir)
                    print(f"\n  ✓ saved best (probe_f={f_acc:.3f}, probe_r={r_acc:.3f}, score={score:.3f})")

                if r_acc < args.min_retain_probe_acc:
                    bad_probe_steps += 1
                else:
                    bad_probe_steps = 0

                if bad_probe_steps >= args.max_bad_probe_steps:
                    print(f"\n  ! early stop: retain probe below {args.min_retain_probe_acc:.2f} for {bad_probe_steps} checks")
                    break

        avg = ep_loss / max(steps, 1)
        print(f"  Epoch {ep+1}: loss={avg:.4f}")

    if best_score <= -1e8:
        # If all checkpoints violated retain floor, keep final weights for debugging reproducibility.
        model.save_pretrained(args.output_dir)
        print("  ! no probe-qualified checkpoint, saved final weights")

    print("\n[5/5] Save config")
    with open(os.path.join(args.output_dir, "stage2_mcq_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    print(f"Done. saved -> {args.output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="modelzoo/qwen3_8B")
    p.add_argument("--adapter_path", default="trained_models/xTransform/qwen3_8B_drt_as16_r5950_l02_lmapRw02_ns0_top8")
    p.add_argument("--forget_file", default="datasets/processed_forget_sets/1.7B_to_8B_transfer/wmdp-cyber/forget_10pct.json")
    p.add_argument("--retain_file", default="datasets/processed_forget_sets/1.7B_to_8B_transfer/wmdp-cyber/remain_10pct.json")
    p.add_argument("--output_dir", default="trained_models/xTransform/qwen3_8B_stage2_mcq")
    p.add_argument("--lambda_forget", type=float, default=0.35)
    p.add_argument("--lambda_retain", type=float, default=0.75)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--probe_every", type=int, default=20)
    p.add_argument("--probe_samples", type=int, default=64)
    p.add_argument("--target_forget_acc", type=float, default=0.50)
    p.add_argument("--min_retain_probe_acc", type=float, default=0.60)
    p.add_argument("--retain_score_weight", type=float, default=0.30)
    p.add_argument("--max_bad_probe_steps", type=int, default=2)
    p.add_argument("--forget_interval", type=int, default=None)
    args = p.parse_args()
    train(args)
