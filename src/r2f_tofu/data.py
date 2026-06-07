from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100
SYSTEM_PROMPT = "You are a helpful assistant."


def _first_answer(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        if not value:
            return ""
        return _first_answer(value[0])
    if isinstance(value, dict):
        for key in ("answer", "text", "content"):
            if key in value:
                return _first_answer(value[key])
    return str(value)


def normalize_sample(raw: dict[str, Any]) -> dict[str, str]:
    question = raw.get("question") or raw.get("prompt") or raw.get("input") or raw.get("query")
    answer = raw.get("answer")
    if answer is None:
        answer = raw.get("answers")
    if answer is None:
        answer = raw.get("gold_answers")
    if answer is None:
        answer = raw.get("target") or raw.get("output") or raw.get("completion")
    if question is None:
        raise ValueError(f"TOFU sample is missing a question-like field: {raw.keys()}")
    return {"question": str(question), "answer": _first_answer(answer)}


def load_tofu_file(path: str | Path, limit: int | None = None) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []

    rows: list[Any]
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            for key in ("data", "train", "samples", "examples"):
                if key in payload and isinstance(payload[key], list):
                    rows = payload[key]
                    break
            else:
                rows = [payload]
        else:
            raise ValueError(f"Unsupported JSON payload in {path}")

    samples = [normalize_sample(row) for row in rows if isinstance(row, dict)]
    return samples[:limit] if limit is not None else samples


def build_llama3_prompt(question: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def tokenize_qa(
    tokenizer: Any,
    sample: dict[str, str],
    max_length: int,
    add_eos: bool = True,
) -> dict[str, torch.Tensor]:
    prompt = build_llama3_prompt(sample["question"])
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(sample["answer"], add_special_tokens=False)["input_ids"]
    if add_eos and tokenizer.eos_token_id is not None:
        answer_ids = answer_ids + [tokenizer.eos_token_id]

    if len(answer_ids) >= max_length:
        input_ids = answer_ids[-max_length:]
        labels = input_ids.copy()
    else:
        prompt_budget = max_length - len(answer_ids)
        prompt_ids = prompt_ids[-prompt_budget:]
        input_ids = prompt_ids + answer_ids
        labels = [IGNORE_INDEX] * len(prompt_ids) + answer_ids

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
    }


class PairedTOFUDataset(Dataset):
    def __init__(
        self,
        forget_samples: Sequence[dict[str, str]],
        retain_samples: Sequence[dict[str, str]],
        max_forget_samples: int | None = None,
        seed: int = 42,
    ) -> None:
        if not forget_samples:
            raise ValueError("forget_samples is empty")
        if not retain_samples:
            raise ValueError("retain_samples is empty")
        self.forget_samples = list(forget_samples)
        self.retain_samples = list(retain_samples)
        self.length = min(len(self.forget_samples), max_forget_samples or len(self.forget_samples))
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, dict[str, str]]:
        rng = random.Random(self.seed + idx)
        retain = self.retain_samples[rng.randrange(len(self.retain_samples))]
        return {"forget": self.forget_samples[idx], "retain": retain}


@dataclass
class DataCollatorForR2F:
    tokenizer: Any
    max_length: int

    def _pad(self, rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [row["input_ids"] for row in rows],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [row["labels"] for row in rows],
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

    def _collate_samples(self, samples: Sequence[dict[str, str]]) -> dict[str, torch.Tensor]:
        tokenized = [tokenize_qa(self.tokenizer, sample, self.max_length) for sample in samples]
        return self._pad(tokenized)

    def __call__(self, instances: Sequence[Any]) -> dict[str, Any]:
        if instances and isinstance(instances[0], dict) and "forget" in instances[0]:
            return {
                "forget": self._collate_samples([x["forget"] for x in instances]),
                "retain": self._collate_samples([x["retain"] for x in instances]),
            }
        return self._collate_samples(instances)


def build_paired_loader(
    tokenizer: Any,
    forget_file: str | Path,
    retain_file: str | Path,
    max_length: int,
    batch_size: int,
    max_forget_samples: int | None,
    seed: int,
) -> torch.utils.data.DataLoader:
    forget_samples = load_tofu_file(forget_file, limit=max_forget_samples)
    retain_samples = load_tofu_file(retain_file)
    dataset = PairedTOFUDataset(forget_samples, retain_samples, max_forget_samples, seed=seed)
    collator = DataCollatorForR2F(tokenizer=tokenizer, max_length=max_length)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
