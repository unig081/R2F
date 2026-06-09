from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .config import deep_get, load_config
from .data import load_tofu_file
from .evaluate_tofu import _write_report, evaluate_method
from .utils import append_jsonl, ensure_dir, set_seed, write_csv, write_json


DEFAULT_BUNDLE_ROOT = Path(
    os.environ.get(
        "R2F_HANDOFF_BUNDLE",
        "/mnt/data1/zxc/handoff/junior_llama_tofu_eval_bundle",
    )
)

RETAIN_BY_FORGET_SPLIT = {
    "forget01": "retain99",
    "forget05": "retain95",
    "forget10": "retain90",
}


def _resolve_bundle_path(path: str | Path | None, default: Path) -> Path:
    if path is None:
        return default
    return Path(path).expanduser()


def _require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def _load_base_cfg(config_path: str | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    return load_config(config_path)


def build_eval_cfg(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, str], Path]:
    base_cfg = _load_base_cfg(args.config)
    bundle_root = _resolve_bundle_path(args.bundle_root, DEFAULT_BUNDLE_ROOT)
    forget_split = str(args.forget_split)
    retain_split = RETAIN_BY_FORGET_SPLIT[forget_split]

    default_model_path = bundle_root / "models" / "llama_3_2_1B_instruct_tofu"
    default_adapter_path = (
        bundle_root / "checkpoints" / "tofu_gagd" / f"llama_3_2_1B_tofu_{forget_split}" / "adapter"
    )
    default_forget_file = bundle_root / "datasets" / "tofu" / f"{forget_split}.json"
    default_retain_file = bundle_root / "datasets" / "tofu" / f"{retain_split}.json"
    default_output_root = Path(os.environ.get("R2F_OUTPUT_DIR", "/mnt/data1/zxc/R2F/results"))

    model_path = _resolve_bundle_path(args.model_path, default_model_path)
    adapter_path = _resolve_bundle_path(args.adapter_path, default_adapter_path)
    forget_file = _resolve_bundle_path(args.forget_file, default_forget_file)
    retain_file = _resolve_bundle_path(args.retain_file, default_retain_file)
    output_dir = ensure_dir(
        _resolve_bundle_path(args.output_dir, default_output_root / "eval_1b_lora" / forget_split)
    )

    _require_path(model_path / "config.json", "1B model config")
    _require_path(adapter_path / "adapter_config.json", "LoRA adapter config")
    _require_path(forget_file, "forget file")
    _require_path(retain_file, "retain file")

    max_forget = int(args.max_forget)
    max_retain = int(args.max_retain)
    max_length = int(args.max_length)
    max_new_tokens = int(args.max_new_tokens)
    if args.smoke:
        max_forget = min(max_forget, 2)
        max_retain = min(max_retain, 2)
        max_length = min(max_length, 256)
        max_new_tokens = min(max_new_tokens, 32)

    cfg: dict[str, Any] = {
        "seed": int(args.seed),
        "paths": {
            "forget_file": str(forget_file),
            "retain_file": str(retain_file),
            "output_dir": str(output_dir),
        },
        "model": {
            "dtype": args.dtype or deep_get(base_cfg, "model.dtype", "bfloat16"),
            "device_map": args.device_map or deep_get(base_cfg, "model.device_map", "auto"),
            "attn_implementation": args.attn_implementation
            if args.attn_implementation is not None
            else deep_get(base_cfg, "model.attn_implementation"),
            "trust_remote_code": bool(deep_get(base_cfg, "model.trust_remote_code", True)),
            "max_length": max_length,
        },
        "evaluation": {
            "output_dir": str(output_dir),
            "max_forget": max_forget,
            "max_retain": max_retain,
            "max_new_tokens": max_new_tokens,
        },
    }

    method = {
        "name": f"LoRA-GA+GD-1B-{forget_split}",
        "type": "lora",
        "model_path": str(model_path),
        "tokenizer_path": str(model_path),
        "adapter_path": str(adapter_path),
    }
    return cfg, method, output_dir


def evaluate_1b_lora(args: argparse.Namespace) -> dict[str, Any]:
    cfg, lora_method, output_dir = build_eval_cfg(args)
    set_seed(int(cfg.get("seed", 42)))

    max_forget = int(deep_get(cfg, "evaluation.max_forget", 100))
    max_retain = int(deep_get(cfg, "evaluation.max_retain", 100))
    forget_samples = load_tofu_file(deep_get(cfg, "paths.forget_file"), limit=max_forget)
    retain_samples = load_tofu_file(deep_get(cfg, "paths.retain_file"), limit=max_retain)

    methods: list[dict[str, str]] = []
    if args.include_base:
        methods.append(
            {
                "name": f"Base-1B-{args.forget_split}",
                "type": "base",
                "model_path": lora_method["model_path"],
                "tokenizer_path": lora_method["tokenizer_path"],
            }
        )
    methods.append(lora_method)

    summary_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for method in methods:
        metrics, rows = evaluate_method(cfg, method, forget_samples, retain_samples, output_dir)
        summary_rows.append(metrics)
        prediction_rows.extend(rows)

    write_csv(output_dir / "summary.csv", summary_rows)
    write_json(
        output_dir / "tofu_metrics.json",
        {
            "methods": summary_rows,
            "forget_split": args.forget_split,
            "smoke": bool(args.smoke),
            "model_path": lora_method["model_path"],
            "adapter_path": lora_method["adapter_path"],
            "forget_file": deep_get(cfg, "paths.forget_file"),
            "retain_file": deep_get(cfg, "paths.retain_file"),
        },
    )
    append_jsonl(output_dir / "predictions.jsonl", prediction_rows)
    _write_report(output_dir / "tofu_1b_lora_summary.md", summary_rows, prediction_rows)
    return {
        "methods": summary_rows,
        "output_dir": str(output_dir),
        "forget_split": args.forget_split,
        "smoke": bool(args.smoke),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the handoff 1B LLaMA TOFU GA+GD LoRA with evaluate_tofu metrics."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional config for model dtype/device defaults.",
    )
    parser.add_argument("--bundle-root", default=None, help="Handoff bundle root.")
    parser.add_argument(
        "--forget-split",
        default="forget05",
        choices=sorted(RETAIN_BY_FORGET_SPLIT),
        help="TOFU forget split and matching LoRA adapter to evaluate.",
    )
    parser.add_argument("--model-path", default=None, help="Base 1B model path.")
    parser.add_argument("--adapter-path", default=None, help="GA+GD LoRA adapter path.")
    parser.add_argument("--forget-file", default=None, help="Forget QA file.")
    parser.add_argument("--retain-file", default=None, help="Retain QA file.")
    parser.add_argument("--output-dir", default=None, help="Evaluation output directory.")
    parser.add_argument("--max-forget", type=int, default=100)
    parser.add_argument("--max-retain", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument(
        "--dtype",
        default=None,
        choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
    )
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-base",
        action="store_true",
        help="Also evaluate the base 1B model.",
    )
    parser.add_argument("--smoke", action="store_true", help="Run a tiny 2-example sanity check.")
    return parser.parse_args()


def main() -> None:
    result = evaluate_1b_lora(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
