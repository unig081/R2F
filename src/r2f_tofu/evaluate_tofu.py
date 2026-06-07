from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .config import apply_smoke_overrides, deep_get, load_config
from .data import build_llama3_prompt, load_tofu_file
from .metrics import generation_match, summarize_split
from .models import get_model_device, load_causal_lm, load_tokenizer
from .utils import append_jsonl, ensure_dir, move_to_device, set_seed, write_csv, write_json


def _tokenize_prompt_answer(tokenizer: Any, prompt: str, answer: str, max_length: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        answer_ids = answer_ids + [tokenizer.eos_token_id]
    if len(answer_ids) >= max_length:
        input_ids = answer_ids[-max_length:]
        prompt_len = 0
    else:
        prompt_budget = max_length - len(answer_ids)
        prompt_ids = prompt_ids[-prompt_budget:]
        prompt_len = len(prompt_ids)
        input_ids = prompt_ids + answer_ids
    attention_mask = [1] * len(input_ids)
    return (
        torch.tensor([input_ids], dtype=torch.long),
        torch.tensor([attention_mask], dtype=torch.long),
        prompt_len,
    )


def answer_probability(
    model: torch.nn.Module,
    tokenizer: Any,
    question: str,
    answer: str,
    max_length: int,
) -> tuple[float, float]:
    prompt = build_llama3_prompt(question)
    input_ids, attention_mask, prompt_len = _tokenize_prompt_answer(tokenizer, prompt, answer, max_length)
    device = get_model_device(model)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0]

    token_ids = input_ids[0]
    logps: list[float] = []
    for pos in range(max(prompt_len, 1), len(token_ids)):
        target_id = token_ids[pos]
        logp = torch.log_softmax(logits[pos - 1].float(), dim=-1)[target_id].item()
        logps.append(float(logp))
    if not logps:
        return 0.0, 0.0
    avg_logprob = sum(logps) / len(logps)
    return float(math.exp(avg_logprob)), float(avg_logprob)


def generate_answer(
    model: torch.nn.Module,
    tokenizer: Any,
    question: str,
    max_length: int,
    max_new_tokens: int,
) -> str:
    prompt = build_llama3_prompt(question)
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    device = get_model_device(model)
    enc = move_to_device(enc, device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0, enc["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _load_method_model(cfg: dict[str, Any], method: dict[str, str]) -> tuple[torch.nn.Module, Any]:
    model_path = method["model_path"]
    tokenizer_path = method.get("tokenizer_path") or model_path
    if method["type"] == "lora":
        from peft import PeftModel

        base = load_causal_lm(
            model_path,
            dtype_name=str(deep_get(cfg, "model.dtype", "bfloat16")),
            device_map=deep_get(cfg, "model.device_map", "auto"),
            trust_remote_code=bool(deep_get(cfg, "model.trust_remote_code", True)),
            attn_implementation=deep_get(cfg, "model.attn_implementation"),
        )
        model = PeftModel.from_pretrained(base, method["adapter_path"])
        tokenizer = load_tokenizer(tokenizer_path, trust_remote_code=bool(deep_get(cfg, "model.trust_remote_code", True)))
        return model.eval(), tokenizer

    model = load_causal_lm(
        model_path,
        dtype_name=str(deep_get(cfg, "model.dtype", "bfloat16")),
        device_map=deep_get(cfg, "model.device_map", "auto"),
        trust_remote_code=bool(deep_get(cfg, "model.trust_remote_code", True)),
        attn_implementation=deep_get(cfg, "model.attn_implementation"),
    )
    tokenizer = load_tokenizer(tokenizer_path, trust_remote_code=bool(deep_get(cfg, "model.trust_remote_code", True)))
    return model.eval(), tokenizer


def discover_methods(cfg: dict[str, Any]) -> list[dict[str, str]]:
    target_model = str(deep_get(cfg, "paths.target_model"))
    output_root = Path(deep_get(cfg, "paths.output_dir"))
    methods: list[dict[str, str]] = [
        {"name": "Base-3B", "type": "base", "model_path": target_model, "tokenizer_path": target_model}
    ]

    adapter_dir = output_root / "lora_gagd_3b" / "adapter"
    if (adapter_dir / "adapter_config.json").exists():
        methods.append(
            {
                "name": "LoRA-GA+GD-3B",
                "type": "lora",
                "model_path": target_model,
                "tokenizer_path": target_model,
                "adapter_path": str(adapter_dir),
            }
        )

    r2f_dir = Path(deep_get(cfg, "r2f.output_dir"))
    for updated in sorted(r2f_dir.glob("eta_*/updated_model")):
        if (updated / "config.json").exists():
            methods.append(
                {
                    "name": f"R2F-3B-{updated.parent.name}",
                    "type": "updated",
                    "model_path": str(updated),
                    "tokenizer_path": str(updated),
                }
            )
    return methods


def evaluate_method(
    cfg: dict[str, Any],
    method: dict[str, str],
    forget_samples: list[dict[str, str]],
    retain_samples: list[dict[str, str]],
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model, tokenizer = _load_method_model(cfg, method)
    max_length = int(deep_get(cfg, "model.max_length", 1024))
    max_new_tokens = int(deep_get(cfg, "evaluation.max_new_tokens", 64))
    rows: list[dict[str, Any]] = []

    for split, samples in (("forget", forget_samples), ("retain", retain_samples)):
        for idx, sample in enumerate(tqdm(samples, desc=f"eval {method['name']} {split}")):
            prob, avg_logprob = answer_probability(model, tokenizer, sample["question"], sample["answer"], max_length)
            generation = generate_answer(
                model,
                tokenizer,
                sample["question"],
                max_length=max_length,
                max_new_tokens=max_new_tokens,
            )
            rows.append(
                {
                    "method": method["name"],
                    "split": split,
                    "idx": idx,
                    "question": sample["question"],
                    "answer": sample["answer"],
                    "qa_prob": prob,
                    "avg_logprob": avg_logprob,
                    "generation": generation,
                    "generation_match": generation_match(generation, sample["answer"]),
                }
            )

    forget_summary = summarize_split(rows, "forget")
    retain_summary = summarize_split(rows, "retain")
    metrics = {
        "method": method["name"],
        **forget_summary,
        **retain_summary,
        "model_utility": retain_summary["retain_Q_A_Prob"]
        * (1.0 - forget_summary["forget_generation_match"]),
    }

    method_dir = ensure_dir(output_dir / method["name"].replace("/", "_").replace(" ", "_"))
    write_json(method_dir / "tofu_metrics.json", metrics)
    append_jsonl(method_dir / "predictions.jsonl", rows)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics, rows


def _write_report(path: Path, summary_rows: list[dict[str, Any]], prediction_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# TOFU R2F Summary",
        "",
        "## Metrics",
        "",
        "| method | forget_Q_A_Prob | forget_match | retain_Q_A_Prob | retain_match | model_utility |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['method']} | {row.get('forget_Q_A_Prob', 0):.6g} | "
            f"{row.get('forget_generation_match', 0):.4f} | {row.get('retain_Q_A_Prob', 0):.6g} | "
            f"{row.get('retain_generation_match', 0):.4f} | {row.get('model_utility', 0):.6g} |"
        )

    for split in ("forget", "retain"):
        lines.extend(["", f"## {split.title()} Examples", ""])
        subset = [row for row in prediction_rows if row["split"] == split][:5]
        for row in subset:
            lines.extend(
                [
                    f"### {row['method']} #{row['idx']}",
                    f"Q: {row['question']}",
                    f"Gold: {row['answer']}",
                    f"Generation: {row['generation']}",
                    f"Match: {row['generation_match']}",
                    "",
                ]
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def evaluate_all(cfg: dict[str, Any], smoke: bool = False) -> dict[str, Any]:
    set_seed(int(cfg.get("seed", 42)))
    max_forget = int(deep_get(cfg, "evaluation.max_forget", 100))
    max_retain = int(deep_get(cfg, "evaluation.max_retain", 100))
    forget_samples = load_tofu_file(deep_get(cfg, "paths.forget_file"), limit=max_forget)
    retain_samples = load_tofu_file(deep_get(cfg, "paths.retain_file"), limit=max_retain)
    output_dir = ensure_dir(deep_get(cfg, "evaluation.output_dir"))

    methods = discover_methods(cfg)
    summary_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for method in methods:
        metrics, rows = evaluate_method(cfg, method, forget_samples, retain_samples, output_dir)
        summary_rows.append(metrics)
        prediction_rows.extend(rows)

    write_csv(Path(deep_get(cfg, "paths.output_dir")) / "summary.csv", summary_rows)
    write_json(output_dir / "tofu_metrics.json", {"methods": summary_rows, "smoke": smoke})
    append_jsonl(output_dir / "predictions.jsonl", prediction_rows)
    _write_report(Path(deep_get(cfg, "root", ".")) / "reports" / "tofu_r2f_summary.md", summary_rows, prediction_rows)
    return {"methods": summary_rows, "output_dir": str(output_dir), "smoke": smoke}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.smoke:
        cfg = apply_smoke_overrides(cfg)
    result = evaluate_all(cfg, smoke=args.smoke)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
