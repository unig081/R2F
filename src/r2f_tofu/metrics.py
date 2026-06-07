from __future__ import annotations

import re
from typing import Any


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def generation_match(generation: str, answer: str) -> bool:
    answer_norm = normalize_text(answer)
    generation_norm = normalize_text(generation)
    if not answer_norm:
        return False
    if answer_norm in generation_norm:
        return True
    answer_tokens = answer_norm.split()
    if len(answer_tokens) >= 4:
        short = " ".join(answer_tokens[:4])
        return short in generation_norm
    return False


def summarize_split(rows: list[dict[str, Any]], split: str) -> dict[str, float]:
    subset = [row for row in rows if row["split"] == split]
    probs = [float(row["qa_prob"]) for row in subset]
    logprobs = [float(row["avg_logprob"]) for row in subset]
    matches = [1.0 if row["generation_match"] else 0.0 for row in subset]
    return {
        f"{split}_Q_A_Prob": mean(probs),
        f"{split}_avg_logprob": mean(logprobs),
        f"{split}_generation_match": mean(matches),
        f"n_{split}": float(len(subset)),
    }
