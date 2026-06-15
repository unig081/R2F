from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from collections import defaultdict
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import apply_smoke_overrides, deep_get, load_config
from .model_families import chat_template_kwargs
from .utils import ensure_dir, set_seed, write_json

IGNORE_INDEX = -100
SUMMARY_KEYS = [
    "extraction_strength",
    "forget_Q_A_Prob",
    "forget_Q_A_ROUGE",
    "forget_truth_ratio",
    "model_utility",
    "privleak",
]
logger = logging.getLogger("r2f_tofu.evaluate_tofu")


def _load_json_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("data", "train", "samples", "examples"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
        return [payload]
    raise ValueError(f"Unsupported JSON payload: {path}")


def _tofu_data_dir(cfg: dict[str, Any]) -> Path:
    configured = deep_get(cfg, "evaluation.tofu_data_dir")
    if configured:
        return Path(str(configured)).expanduser()
    forget_file = deep_get(cfg, "paths.forget_file")
    if forget_file:
        return Path(str(forget_file)).expanduser().parent
    return Path("datasets/tofu")


def _split_name_from_path(path: str | Path | None, default: str) -> str:
    if path is None:
        return default
    stem = Path(str(path)).stem
    match = re.search(r"(forget(?:01|05|10))", stem)
    return match.group(1) if match else default


def _resolve_forget_split(cfg: dict[str, Any]) -> str:
    return str(
        deep_get(
            cfg,
            "evaluation.forget_split",
            _split_name_from_path(deep_get(cfg, "paths.forget_file"), "forget05"),
        )
    )


def _resolve_holdout_split(cfg: dict[str, Any], forget_split: str) -> str:
    default = forget_split.replace("forget", "holdout")
    return str(deep_get(cfg, "evaluation.holdout_split", default))


def _read_tofu_split(
    data_dir: Path,
    split: str,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    rows = _load_json_rows(data_dir / f"{split}.json")
    for idx, row in enumerate(rows):
        row.setdefault("index", idx)
    return rows[:limit] if limit is not None else rows


def _as_answer(value: Any) -> str | list[str]:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        answers: list[str] = []
        for item in value:
            answer = _as_answer(item)
            if isinstance(answer, list):
                answers.extend(str(x) for x in answer)
            else:
                answers.append(str(answer))
        return answers
    if isinstance(value, dict):
        for key in ("answer", "text", "content"):
            if key in value:
                return _as_answer(value[key])
    return str(value)


def _token_list(tokenized_output: Any) -> list[int]:
    if hasattr(tokenized_output, "keys") and "input_ids" in tokenized_output:
        tokenized_output = tokenized_output["input_ids"]
    if torch.is_tensor(tokenized_output):
        tokenized_output = tokenized_output.tolist()
    if tokenized_output and isinstance(tokenized_output[0], list):
        tokenized_output = tokenized_output[0]
    return list(tokenized_output)


def _preprocess_chat_instance(
    tokenizer: Any,
    question: str,
    answer: str,
    max_length: int,
    *,
    model_family: str = "llama",
    predict_with_generate: bool = False,
) -> dict[str, torch.Tensor]:
    chat = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    template_kwargs = chat_template_kwargs(model_family)
    try:
        chat_ids = _token_list(
            tokenizer.apply_chat_template(
                chat,
                tokenize=True,
                add_generation_prompt=False,
                **template_kwargs,
            )
        )
        prompt_ids = _token_list(
            tokenizer.apply_chat_template(
                chat[:-1],
                tokenize=True,
                add_generation_prompt=True,
                **template_kwargs,
            )
        )
    except TypeError:
        chat_ids = _token_list(
            tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=False)
        )
        prompt_ids = _token_list(
            tokenizer.apply_chat_template(chat[:-1], tokenize=True, add_generation_prompt=True)
        )
    if chat_ids[-1] != tokenizer.eos_token_id:
        chat_ids += [tokenizer.eos_token_id]

    len_matched = len(prompt_ids)
    labels = chat_ids if predict_with_generate else [IGNORE_INDEX] * len_matched + chat_ids[len_matched:]
    input_ids = prompt_ids if predict_with_generate else chat_ids
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


class HandoffQADataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        question_key: str = "question",
        answer_key: str = "answer",
        tokenizer: Any,
        max_length: int = 512,
        model_family: str = "llama",
        predict_with_generate: bool = False,
    ) -> None:
        self.rows = rows
        self.question_key = question_key
        self.answer_key = answer_key
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_family = model_family
        self.predict_with_generate = predict_with_generate

    def __len__(self) -> int:
        return len(self.rows)

    def _process(self, question: str, answer: str, index: int) -> dict[str, torch.Tensor]:
        item = _preprocess_chat_instance(
            self.tokenizer,
            question,
            answer,
            self.max_length,
            model_family=self.model_family,
            predict_with_generate=self.predict_with_generate,
        )
        item["index"] = index
        return item

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        question = str(row[self.question_key])
        index = int(row.get("index", idx))
        answer = _as_answer(row.get(self.answer_key))
        if isinstance(answer, list):
            return {
                str(i): self._process(question, str(ans), index)
                for i, ans in enumerate(answer)
            }
        return self._process(question, str(answer), index)


class DataCollatorForSupervisedDataset:
    def __init__(self, tokenizer: Any, *, padding_side: str = "right", index: str | None = "index") -> None:
        self.tokenizer = tokenizer
        self.padding_side = padding_side
        self.index = index

    def _pad_tokens(self, rows: list[torch.Tensor], padding_value: int) -> torch.Tensor:
        if self.padding_side == "right":
            return torch.nn.utils.rnn.pad_sequence(rows, batch_first=True, padding_value=padding_value)
        return torch.nn.utils.rnn.pad_sequence(
            [torch.flip(row, dims=[0]) for row in rows],
            batch_first=True,
            padding_value=padding_value,
        ).flip(dims=[1])

    def __call__(self, instances: list[dict[str, Any]]) -> dict[str, Any]:
        if "input_ids" not in instances[0]:
            return {
                key: self([instance[key] for instance in instances])
                for key in instances[0]
            }
        input_ids = self._pad_tokens([x["input_ids"] for x in instances], self.tokenizer.pad_token_id)
        labels = self._pad_tokens([x["labels"] for x in instances], IGNORE_INDEX)
        batch = {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
            "labels": labels,
        }
        if self.index is not None:
            batch[self.index] = torch.tensor([int(x[self.index]) for x in instances])
        return batch


def _aggregate_to_1d(values: np.ndarray) -> np.ndarray:
    return np.mean(values, axis=tuple(range(1, values.ndim))) if values.ndim > 1 else values


def _dict_transpose(evals: dict[str, dict[int, dict[str, Any]]]) -> dict[int, dict[str, list[Any]]]:
    all_iidxs = list(evals.keys())
    all_idxs = list(evals[all_iidxs[0]].keys())
    all_stat_names = list(evals[all_iidxs[0]][all_idxs[0]].keys())
    return {
        idx: {
            stat: [evals[iidx][idx][stat] for iidx in all_iidxs]
            for stat in all_stat_names
        }
        for idx in all_idxs
    }


def _model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def _move_batch_to_device(batch: dict[str, torch.Tensor], model: torch.nn.Module) -> dict[str, torch.Tensor]:
    device = _model_device(model)
    return {k: v.to(device) for k, v in batch.items()}


def _run_batchwise_evals(
    model: torch.nn.Module,
    dataloader: DataLoader,
    batch_eval_fn: Callable[..., list[dict[str, Any]]],
    batch_eval_fn_args: dict[str, Any],
    desc: str,
) -> dict[Any, dict[str, Any]]:
    evals: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for batch in tqdm(dataloader, desc=desc, total=len(dataloader)):
        if "input_ids" in batch:
            batch = {"0": batch}
        for intra_item_idx, mini_batch in batch.items():
            data_indices = mini_batch.pop("index").cpu().numpy().tolist()
            batch_evals = batch_eval_fn(model=model, batch=mini_batch, **batch_eval_fn_args)
            indexwise = dict(zip(data_indices, batch_evals))
            if evals[intra_item_idx].keys() & indexwise.keys():
                raise RuntimeError("Data indices repeated while iterating dataloader")
            evals[intra_item_idx] |= indexwise
    if len(evals) == 1:
        return next(iter(evals.values()))
    return _dict_transpose(evals)


def _evaluate_probability(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> list[dict[str, float]]:
    batch = _move_batch_to_device(batch, model)
    with torch.no_grad():
        output = model(**batch)
    logits = output.logits[..., :-1, :].contiguous()
    shifted_labels = batch["labels"][..., 1:].contiguous()
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="none")
    losses = loss_fn(logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
    num_token_gt = (batch["labels"] != IGNORE_INDEX).sum(-1)
    avg_losses = losses / num_token_gt
    probs = torch.exp(-avg_losses)
    return [
        {"prob": float(prob), "avg_loss": float(avg_loss)}
        for prob, avg_loss in zip(
            probs.to(torch.float32).cpu().numpy().tolist(),
            avg_losses.to(torch.float32).cpu().numpy().tolist(),
        )
    ]


def _tokenwise_vocab_logprobs(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    batch = _move_batch_to_device(batch, model)
    with torch.no_grad():
        output = model(**batch)
    logits = output.logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)[:, :-1, :]
    labels_batch: list[torch.Tensor] = []
    log_probs_batch: list[torch.Tensor] = []
    for i in range(logits.shape[0]):
        labels = batch["labels"][i]
        actual_indices = (labels != IGNORE_INDEX).nonzero(as_tuple=True)[0][:-1]
        if len(actual_indices) == 0:
            labels_batch.append(torch.tensor([], device=labels.device))
            log_probs_batch.append(torch.zeros(0, logits.shape[-1], device=labels.device))
            continue
        start_idx, end_idx = actual_indices[0].item(), actual_indices[-1].item()
        log_probs_batch.append(log_probs[i, start_idx - 1 : end_idx])
        labels_batch.append(labels[actual_indices])
    return log_probs_batch, labels_batch


def _tokenwise_logprobs(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    batch = _move_batch_to_device(batch, model)
    with torch.no_grad():
        output = model(**batch)
    logits = output.logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)[:, :-1, :]
    next_tokens = batch["input_ids"][:, 1:].unsqueeze(-1)
    target_log_probs = torch.gather(log_probs, dim=2, index=next_tokens).squeeze(-1)
    log_probs_batch: list[torch.Tensor] = []
    for i in range(logits.shape[0]):
        labels = batch["labels"][i]
        actual_indices = (labels != IGNORE_INDEX).nonzero(as_tuple=True)[0][:-1]
        if actual_indices.numel() == 0:
            log_probs_batch.append(torch.tensor([], device=labels.device))
            continue
        start_idx, end_idx = actual_indices[0].item(), actual_indices[-1].item()
        log_probs_batch.append(target_log_probs[i, start_idx - 1 : end_idx])
    return log_probs_batch


def _evaluate_rouge(
    model: torch.nn.Module,
    tokenizer: Any,
    batch: dict[str, torch.Tensor],
    generation_args: dict[str, Any],
) -> list[dict[str, Any]]:
    from rouge_score import rouge_scorer

    batch = _move_batch_to_device(batch, model)
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    input_texts = tokenizer.batch_decode(
        input_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    target_tokens = [label[label != IGNORE_INDEX] for label in labels]
    full_texts = tokenizer.batch_decode(
        target_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    ground_truths = [
        full_text.replace(input_text, "").strip()
        for input_text, full_text in zip(input_texts, full_texts)
    ]
    gen_kwargs = dict(generation_args)
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
    output = model.generate(
        input_ids,
        attention_mask=batch["attention_mask"],
        **gen_kwargs,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_texts = tokenizer.batch_decode(
        output[:, input_ids.shape[-1] :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    stopwords = [tokenizer.decode([tokenizer.eos_token_id])]
    for i, text in enumerate(gen_texts):
        for word in stopwords:
            if word and word in text:
                text = text.split(word)[0]
        gen_texts[i] = text.strip()

    think_re = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
    clean_gen_texts = [think_re.sub("", text).strip() for text in gen_texts]
    clean_ground_truths = [think_re.sub("", text).strip() for text in ground_truths]
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    rows = []
    for input_text, ground_truth, gen_text, clean_gt, clean_gen in zip(
        input_texts,
        ground_truths,
        gen_texts,
        clean_ground_truths,
        clean_gen_texts,
    ):
        scores = scorer.score(clean_gt, clean_gen)
        rows.append(
            {
                "rouge1_recall": scores["rouge1"].recall,
                "rougeL_f1": scores["rougeL"].fmeasure,
                "rougeL_recall": scores["rougeL"].recall,
                "input": input_text,
                "ground_truth": clean_gt,
                "generation": clean_gen,
                "ground_truth_raw": ground_truth,
                "generation_raw": gen_text,
            }
        )
    return rows


def _make_dataset(
    rows_by_split: dict[str, list[dict[str, Any]]],
    tokenizer: Any,
    split: str,
    *,
    answer_key: str = "answer",
    question_key: str = "question",
    model_family: str = "llama",
    predict_with_generate: bool = False,
) -> HandoffQADataset:
    return HandoffQADataset(
        rows_by_split[split],
        question_key=question_key,
        answer_key=answer_key,
        tokenizer=tokenizer,
        max_length=512,
        model_family=model_family,
        predict_with_generate=predict_with_generate,
    )


def _probability(
    model: torch.nn.Module,
    tokenizer: Any,
    rows_by_split: dict[str, list[dict[str, Any]]],
    split: str,
    *,
    answer_key: str = "answer",
    batch_size: int = 32,
    question_key: str = "question",
    model_family: str = "llama",
) -> dict[str, Any]:
    dataset = _make_dataset(
        rows_by_split,
        tokenizer,
        split,
        answer_key=answer_key,
        question_key=question_key,
        model_family=model_family,
    )
    collator = DataCollatorForSupervisedDataset(tokenizer, index="index")
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    value_by_index = _run_batchwise_evals(
        model,
        loader,
        _evaluate_probability,
        {},
        f"Calculating loss {split}:{answer_key}",
    )
    values = np.array([row["prob"] for row in value_by_index.values() if row["prob"] is not None])
    return {"agg_value": float(np.mean(_aggregate_to_1d(values))), "value_by_index": value_by_index}


def _rouge(
    model: torch.nn.Module,
    tokenizer: Any,
    rows_by_split: dict[str, list[dict[str, Any]]],
    split: str,
    *,
    answer_key: str = "answer",
    batch_size: int = 32,
    question_key: str = "question",
    model_family: str = "llama",
) -> dict[str, Any]:
    dataset = _make_dataset(
        rows_by_split,
        tokenizer,
        split,
        answer_key=answer_key,
        question_key=question_key,
        model_family=model_family,
        predict_with_generate=True,
    )
    collator = DataCollatorForSupervisedDataset(tokenizer, padding_side="left", index="index")
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    generation_args = {
        "do_sample": False,
        "top_p": None,
        "temperature": None,
        "max_new_tokens": 200,
        "use_cache": True,
    }
    value_by_index = _run_batchwise_evals(
        model,
        loader,
        _evaluate_rouge,
        {"tokenizer": tokenizer, "generation_args": generation_args},
        f"Calculating text similarity {split}:{answer_key}",
    )
    values = np.array(
        [row["rougeL_recall"] for row in value_by_index.values() if row["rougeL_recall"] is not None]
    )
    return {"agg_value": float(np.mean(_aggregate_to_1d(values))), "value_by_index": value_by_index}


def _truth_ratio(
    correct: dict[str, Any],
    wrong: dict[str, Any],
    *,
    aggregator: str,
) -> dict[str, Any]:
    correct_results = correct["value_by_index"]
    wrong_results = wrong["value_by_index"]
    correct_indices = list(correct_results.keys())
    wrong_indices = list(wrong_results.keys())
    if correct_indices != wrong_indices:
        raise RuntimeError("Truth-ratio correct and wrong indices differ")
    filtered_indices = [
        idx
        for idx in correct_indices
        if correct_results[idx] is not None and wrong_results[idx] is not None
    ]
    correct_avg_losses = _aggregate_to_1d(
        np.array([correct_results[idx]["avg_loss"] for idx in filtered_indices])
    )
    wrong_avg_losses = _aggregate_to_1d(
        np.array([wrong_results[idx]["avg_loss"] for idx in filtered_indices])
    )
    correct_prob = np.exp(-correct_avg_losses)
    wrong_prob = np.exp(-wrong_avg_losses)
    if aggregator != "prob_mean":
        truth_ratios = wrong_prob / (correct_prob + 1e-10)
    else:
        truth_ratios = correct_prob / (correct_prob + wrong_prob + 1e-10)
    value_by_index = dict(zip(correct_indices, [{"score": float(v)} for v in truth_ratios]))
    stats = np.array([row["score"] for row in value_by_index.values()])
    if aggregator == "closer_to_1_better":
        agg = np.mean(np.minimum(stats, 1 / (stats + 1e-10)))
    elif aggregator == "true_better":
        agg = np.mean(np.maximum(0, 1 - stats))
    elif aggregator == "prob_mean":
        agg = np.mean(stats)
    else:
        raise ValueError(f"Invalid truth ratio aggregator: {aggregator}")
    return {"agg_value": float(agg), "value_by_index": value_by_index}


def _probability_w_options(correct: dict[str, Any], wrong: dict[str, Any]) -> dict[str, Any]:
    correct_results = correct["value_by_index"]
    wrong_results = wrong["value_by_index"]
    correct_indices = list(correct_results.keys())
    wrong_indices = list(wrong_results.keys())
    if correct_indices != wrong_indices:
        raise RuntimeError("Correct and wrong indices differ")
    filtered_indices = [
        idx
        for idx in correct_indices
        if correct_results[idx] is not None and wrong_results[idx] is not None
    ]
    correct_probs = np.array([correct_results[idx]["prob"] for idx in filtered_indices])
    all_wrong = np.array([wrong_results[idx]["prob"] for idx in filtered_indices])
    wrong_probs = np.sum(all_wrong, axis=tuple(range(1, all_wrong.ndim)))
    probs = correct_probs / (correct_probs + wrong_probs + 1e-10)
    value_by_index = dict(zip(correct_indices, [{"prob": float(v)} for v in probs]))
    return {"agg_value": float(np.mean(probs)), "value_by_index": value_by_index}


def _extraction_strength(
    model: torch.nn.Module,
    tokenizer: Any,
    rows_by_split: dict[str, list[dict[str, Any]]],
    split: str,
    *,
    answer_key: str = "answer",
    batch_size: int = 32,
    question_key: str = "question",
    model_family: str = "llama",
) -> dict[str, Any]:
    dataset = _make_dataset(
        rows_by_split,
        tokenizer,
        split,
        answer_key=answer_key,
        question_key=question_key,
        model_family=model_family,
    )
    collator = DataCollatorForSupervisedDataset(tokenizer, index="index")
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)

    def batch_eval(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> list[dict[str, float]]:
        log_probs_batch, labels_batch = _tokenwise_vocab_logprobs(model, batch)
        rows = []
        for log_probs, labels in zip(log_probs_batch, labels_batch):
            valid_len = len(labels)
            preds = torch.argmax(log_probs, dim=-1)
            k = 0
            for k in range(valid_len):
                if torch.equal(preds[k:], labels[k:]):
                    break
            rows.append({"score": 0.0 if valid_len == 0 else float(1 - (k / valid_len))})
        return rows

    value_by_index = _run_batchwise_evals(
        model,
        loader,
        batch_eval,
        {},
        f"Calculating ES {split}:{answer_key}",
    )
    values = np.array([row["score"] for row in value_by_index.values() if row["score"] is not None])
    return {"agg_value": float(np.mean(_aggregate_to_1d(values))), "value_by_index": value_by_index}


def _mia_min_k(
    model: torch.nn.Module,
    tokenizer: Any,
    rows_by_split: dict[str, list[dict[str, Any]]],
    forget_split: str,
    holdout_split: str,
    *,
    batch_size: int = 32,
    k: float = 0.4,
    question_key: str = "question",
    model_family: str = "llama",
) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score

    collator = DataCollatorForSupervisedDataset(tokenizer, index="index")

    def attack(split: str) -> dict[str, Any]:
        dataset = _make_dataset(
            rows_by_split,
            tokenizer,
            split,
            answer_key="answer",
            question_key=question_key,
            model_family=model_family,
        )
        loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
        all_scores: list[float] = []
        all_indices: list[int] = []
        for batch in tqdm(loader, total=len(loader), desc=f"MIA min-k {split}"):
            indices = batch.pop("index").cpu().numpy().tolist()
            values = _tokenwise_logprobs(model, batch)
            scores = []
            for sample_stats in values:
                lp = sample_stats.float().cpu().numpy()
                if lp.size == 0:
                    scores.append(0.0)
                    continue
                num_k = max(1, int(len(lp) * k))
                scores.append(float(-np.mean(np.sort(lp)[:num_k])))
            all_scores.extend(scores)
            all_indices.extend(indices)
        return {
            "agg_value": float(np.mean(all_scores)),
            "value_by_index": {
                str(idx): {"score": float(score)}
                for idx, score in zip(all_indices, all_scores)
            },
        }

    output = {
        "forget": attack(forget_split),
        "holdout": attack(holdout_split),
    }
    forget_scores = [row["score"] for row in output["forget"]["value_by_index"].values()]
    holdout_scores = [row["score"] for row in output["holdout"]["value_by_index"].values()]
    auc = roc_auc_score(
        np.array([0] * len(forget_scores) + [1] * len(holdout_scores)),
        np.array(forget_scores + holdout_scores),
    )
    output["auc"] = float(auc)
    output["agg_value"] = float(auc)
    return output


def _hm_aggregate(results: list[dict[str, Any]]) -> dict[str, float]:
    from scipy import stats

    values = [float(row["agg_value"]) for row in results]
    return {"agg_value": float(stats.hmean(values))}


def _privleak(mia_min_k: dict[str, Any], retain_auc: float = 0.5) -> dict[str, float]:
    score = 1 - float(mia_min_k["agg_value"])
    ref = 1 - float(retain_auc)
    return {"agg_value": float((score - ref) / (ref + 1e-10) * 100)}


def _load_model_and_tokenizer(cfg: dict[str, Any], method: dict[str, str]) -> tuple[torch.nn.Module, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_name = str(deep_get(cfg, "model.dtype", "bfloat16")).lower()
    dtype = torch.bfloat16 if dtype_name in {"bf16", "bfloat16"} else torch.float16 if dtype_name in {"fp16", "float16"} else torch.float32
    model_path = method["model_path"]
    tokenizer_path = method.get("tokenizer_path", model_path)
    model_kwargs = {
        "pretrained_model_name_or_path": model_path,
        "torch_dtype": dtype,
        "device_map": deep_get(cfg, "model.device_map", "cuda"),
    }
    attn_impl = deep_get(cfg, "model.attn_implementation")
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl
    trust_remote_code = bool(deep_get(cfg, "model.trust_remote_code", True))
    model_kwargs["trust_remote_code"] = trust_remote_code
    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    if method["type"] == "lora":
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, method["adapter_path"])
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.eos_token_id is None:
        tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def discover_methods(cfg: dict[str, Any]) -> list[dict[str, str]]:
    target_model = str(deep_get(cfg, "paths.target_model"))
    output_root = Path(deep_get(cfg, "paths.output_dir"))
    methods: list[dict[str, str]] = [
        {
            "name": "Base-3B",
            "type": "base",
            "model_path": target_model,
            "tokenizer_path": target_model,
        }
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
    extra_methods = deep_get(cfg, "evaluation.methods", [])
    if isinstance(extra_methods, list):
        for method in extra_methods:
            if isinstance(method, dict):
                methods.append({k: str(v) for k, v in method.items()})
    return methods


def _load_rows_by_split(
    cfg: dict[str, Any],
    forget_split: str,
    holdout_split: str,
    *,
    smoke: bool,
) -> dict[str, list[dict[str, Any]]]:
    data_dir = _tofu_data_dir(cfg)
    smoke_limit = 2 if smoke else None
    rows_by_split = {
        f"{forget_split}_perturbed": _read_tofu_split(data_dir, f"{forget_split}_perturbed", limit=smoke_limit),
        holdout_split: _read_tofu_split(data_dir, holdout_split, limit=smoke_limit),
        "retain_perturbed": _read_tofu_split(data_dir, "retain_perturbed", limit=smoke_limit),
        "real_authors_perturbed": _read_tofu_split(data_dir, "real_authors_perturbed", limit=smoke_limit),
        "world_facts_perturbed": _read_tofu_split(data_dir, "world_facts_perturbed", limit=smoke_limit),
    }
    return rows_by_split


def evaluate_method(
    cfg: dict[str, Any],
    method: dict[str, str],
    rows_by_split: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    forget_split: str,
    holdout_split: str,
) -> dict[str, Any]:
    model, tokenizer = _load_model_and_tokenizer(cfg, method)
    model_family = str(deep_get(cfg, "model.family", deep_get(cfg, "model_family", "llama")))
    batch_size = int(deep_get(cfg, "evaluation.batch_size", 32))
    question_key = str(deep_get(cfg, "evaluation.question_key", "question"))
    forget_pert = f"{forget_split}_perturbed"
    logs: dict[str, Any] = {}

    logs["forget_Q_A_PARA_Prob"] = _probability(
        model, tokenizer, rows_by_split, forget_pert, answer_key="paraphrased_answer", batch_size=batch_size, question_key=question_key, model_family=model_family
    )
    logs["forget_Q_A_PERT_Prob"] = _probability(
        model, tokenizer, rows_by_split, forget_pert, answer_key="perturbed_answer", batch_size=batch_size, question_key=question_key, model_family=model_family
    )
    logs["forget_truth_ratio"] = _truth_ratio(
        logs["forget_Q_A_PARA_Prob"],
        logs["forget_Q_A_PERT_Prob"],
        aggregator="closer_to_1_better",
    )
    logs["forget_quality"] = {"agg_value": None}
    logs["forget_Q_A_Prob"] = _probability(
        model, tokenizer, rows_by_split, forget_pert, answer_key="answer", batch_size=batch_size, question_key=question_key, model_family=model_family
    )
    logs["forget_Q_A_ROUGE"] = _rouge(
        model, tokenizer, rows_by_split, forget_pert, answer_key="answer", batch_size=batch_size, question_key=question_key, model_family=model_family
    )

    retain_prob = _probability(
        model, tokenizer, rows_by_split, "retain_perturbed", answer_key="answer", batch_size=batch_size, question_key=question_key, model_family=model_family
    )
    retain_rouge = _rouge(
        model, tokenizer, rows_by_split, "retain_perturbed", answer_key="answer", batch_size=batch_size, question_key=question_key, model_family=model_family
    )
    retain_para = _probability(
        model, tokenizer, rows_by_split, "retain_perturbed", answer_key="paraphrased_answer", batch_size=batch_size, question_key=question_key, model_family=model_family
    )
    retain_pert = _probability(
        model, tokenizer, rows_by_split, "retain_perturbed", answer_key="perturbed_answer", batch_size=batch_size, question_key=question_key, model_family=model_family
    )
    retain_tr = _truth_ratio(retain_para, retain_pert, aggregator="true_better")

    ra_prob = _probability(model, tokenizer, rows_by_split, "real_authors_perturbed", answer_key="answer", batch_size=batch_size, model_family=model_family)
    ra_pert = _probability(model, tokenizer, rows_by_split, "real_authors_perturbed", answer_key="perturbed_answer", batch_size=batch_size, model_family=model_family)
    ra_prob_norm = _probability_w_options(ra_prob, ra_pert)
    ra_rouge = _rouge(model, tokenizer, rows_by_split, "real_authors_perturbed", answer_key="answer", batch_size=batch_size, model_family=model_family)
    ra_tr = _truth_ratio(ra_prob, ra_pert, aggregator="true_better")

    wf_prob = _probability(model, tokenizer, rows_by_split, "world_facts_perturbed", answer_key="answer", batch_size=batch_size, model_family=model_family)
    wf_pert = _probability(model, tokenizer, rows_by_split, "world_facts_perturbed", answer_key="perturbed_answer", batch_size=batch_size, model_family=model_family)
    wf_prob_norm = _probability_w_options(wf_prob, wf_pert)
    wf_rouge = _rouge(model, tokenizer, rows_by_split, "world_facts_perturbed", answer_key="answer", batch_size=batch_size, model_family=model_family)
    wf_tr = _truth_ratio(wf_prob, wf_pert, aggregator="true_better")

    logs.update(
        {
            "retain_Q_A_Prob": retain_prob,
            "retain_Q_A_ROUGE": retain_rouge,
            "retain_Q_A_PARA_Prob": retain_para,
            "retain_Q_A_PERT_Prob": retain_pert,
            "retain_Truth_Ratio": retain_tr,
            "ra_Q_A_Prob": ra_prob,
            "ra_Q_A_PERT_Prob": ra_pert,
            "ra_Q_A_Prob_normalised": ra_prob_norm,
            "ra_Q_A_ROUGE": ra_rouge,
            "ra_Truth_Ratio": ra_tr,
            "wf_Q_A_Prob": wf_prob,
            "wf_Q_A_PERT_Prob": wf_pert,
            "wf_Q_A_Prob_normalised": wf_prob_norm,
            "wf_Q_A_ROUGE": wf_rouge,
            "wf_Truth_Ratio": wf_tr,
        }
    )
    logs["model_utility"] = _hm_aggregate(
        [
            retain_prob,
            retain_rouge,
            retain_tr,
            ra_prob_norm,
            ra_rouge,
            ra_tr,
            wf_prob_norm,
            wf_rouge,
            wf_tr,
        ]
    )
    logs["mia_min_k"] = _mia_min_k(
        model,
        tokenizer,
        rows_by_split,
        forget_pert,
        holdout_split,
        batch_size=batch_size,
        question_key=question_key,
        model_family=model_family,
    )
    logs["privleak"] = _privleak(logs["mia_min_k"])
    logs["extraction_strength"] = _extraction_strength(
        model, tokenizer, rows_by_split, forget_pert, answer_key="answer", batch_size=batch_size, question_key=question_key, model_family=model_family
    )

    method_dir = ensure_dir(output_dir / method["name"].replace("/", "_").replace(" ", "_"))
    write_json(method_dir / "TOFU_EVAL.json", logs)
    summary = {
        name: result["agg_value"]
        for name, result in sorted(logs.items())
        if result.get("agg_value") is not None
    }
    write_json(method_dir / "TOFU_SUMMARY.json", summary)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "method": method["name"],
        "model_path": method["model_path"],
        "output_dir": str(method_dir),
        **{key: summary.get(key) for key in SUMMARY_KEYS if key in summary},
    }


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# TOFU R2F Summary",
        "",
        "| method | extraction_strength | forget_Q_A_Prob | forget_Q_A_ROUGE | forget_truth_ratio | model_utility | privleak |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | "
            f"{row.get('extraction_strength', 0):.6g} | "
            f"{row.get('forget_Q_A_Prob', 0):.6g} | "
            f"{row.get('forget_Q_A_ROUGE', 0):.6g} | "
            f"{row.get('forget_truth_ratio', 0):.6g} | "
            f"{row.get('model_utility', 0):.6g} | "
            f"{row.get('privleak', 0):.6g} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def evaluate_all(cfg: dict[str, Any], smoke: bool = False) -> dict[str, Any]:
    set_seed(int(cfg.get("seed", 42)))
    forget_split = _resolve_forget_split(cfg)
    holdout_split = _resolve_holdout_split(cfg, forget_split)
    rows_by_split = _load_rows_by_split(cfg, forget_split, holdout_split, smoke=smoke)
    output_dir = ensure_dir(deep_get(cfg, "evaluation.output_dir"))
    methods = discover_methods(cfg)
    summary_rows = [
        evaluate_method(cfg, method, rows_by_split, output_dir, forget_split, holdout_split)
        for method in methods
    ]
    _write_summary_csv(Path(deep_get(cfg, "paths.output_dir")) / "summary.csv", summary_rows)
    write_json(output_dir / "tofu_metrics.json", {"methods": summary_rows, "smoke": smoke})
    _write_report(Path(deep_get(cfg, "root", ".")) / "reports" / "tofu_r2f_summary.md", summary_rows)
    return {"methods": summary_rows, "output_dir": str(output_dir), "smoke": smoke}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    cfg = load_config(args.config)
    if args.smoke:
        cfg = apply_smoke_overrides(cfg)
    result = evaluate_all(cfg, smoke=args.smoke)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
