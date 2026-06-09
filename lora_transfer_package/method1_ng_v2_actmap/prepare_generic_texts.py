#!/usr/bin/env python3
"""
prepare_generic_texts.py
========================
从 TOFU retain95 数据中抽取通用文本，用于 R_l 激活对齐矩阵的计算。
输出 JSON 数组格式，供 collect_activations.py 使用。

Usage:
  python prepare_generic_texts.py --input datasets/tofu/retain95.json --n 100 --output tmp/generic_texts.json
"""
import argparse, json, os

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="JSONL 格式的 retain 数据")
    p.add_argument("--n", type=int, default=100, help="抽取样本数")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    texts = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            # 优先用 question 字段
            for key in ("question", "prompt_used", "prompt", "text", "instruction"):
                if key in item and str(item[key]).strip():
                    texts.append(str(item[key]).strip())
                    break
            if len(texts) >= args.n:
                break

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    # Output as list of dicts (compatible with collect_activations.py)
    items = [{"text": t} for t in texts]
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)
    print(f"Saved {len(items)} texts to {args.output}")

if __name__ == "__main__":
    main()
