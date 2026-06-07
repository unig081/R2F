import os
import json
import argparse
import random
import torch
import transformers
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Suppress Generation warnings
transformers.logging.set_verbosity_error()

def build_judge_prompt(question, gold_label, gold_option_text, student_answer):
    prompt = (
        f"You are an expert evaluator grading a multiple-choice question.\n\n"
        f"Question and Options:\n{question}\n\n"
        f"The correct option is: {gold_label}. {gold_option_text}\n\n"
        f"The model generated the following text instead of a simple choice letter:\n"
        f"Model Generation:\n{student_answer}\n\n"
        f"Does the model's generated text express the exact same meaning as the correct option? "
        f"First, provide a brief reasoning (1-2 sentences). "
        f"Then, on a new line, output strictly 'CORRECT' or 'INCORRECT'.\n\n"
        f"Reasoning:"
    )
    return prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="models/qwen3_1_7B")
    parser.add_argument("--lora_path", type=str, default="")
    parser.add_argument("--judge_model", type=str, default="./modelzoo/Meta-Llama-3-8B-Instruct", 
                        help="Path to judge model. Set to 'none' or empty to use strict hard matching only.")
    parser.add_argument("--forget_file", type=str, required=True)
    parser.add_argument("--retain_file", type=str, required=True)
    parser.add_argument("--output_result", type=str, default="results/lora_judge_eval_result.json")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for faster generation")
    parser.add_argument("--max_new_tokens", type=int, default=32,
                        help="Max generated tokens per sample (smaller is faster and uses less memory).")
    parser.add_argument("--max_input_length", type=int, default=1024,
                        help="Tokenizer truncation length for prompts.")
    
    # New sampling arguments
    parser.add_argument("--eval_forget_size", type=int, default=None, 
                        help="Number of forget samples to evaluate. None means use all data.")
    parser.add_argument("--eval_retain_size", type=int, default=None, 
                        help="Number of retain samples to evaluate. None means use all data.")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for data sampling selection.")
    
    # New eval mode arguments
    parser.add_argument("--base_only", action="store_true", help="Only evaluate the base model, skip LoRA.")
    parser.add_argument("--lora_only", action="store_true", help="Only evaluate the LoRA model, skip base model.")
                        
    args = parser.parse_args()

    device_eval = "cuda:0"
    device_judge = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"

    judge_model_hf = None
    judge_tokenizer = None
    if args.judge_model and args.judge_model.lower() != "none":
        print(f">>> Loading Judge Model on {device_judge}")
        # Ensure padding side left for decoder-only architectures doing batch generation
        judge_tokenizer = AutoTokenizer.from_pretrained(args.judge_model, padding_side="left")
        if judge_tokenizer.pad_token is None:
            judge_tokenizer.pad_token = judge_tokenizer.eos_token
        judge_model_hf = AutoModelForCausalLM.from_pretrained(args.judge_model, torch_dtype=torch.bfloat16, device_map={"": device_judge})
        judge_model_hf.eval()
    else:
        print(">>> Judge model deactivated. Using strict hard matching (first alphabet letter extraction).")

    print(f">>> Loading Base Eval Model on {device_eval}")
    eval_tokenizer = AutoTokenizer.from_pretrained(args.base_model, padding_side="left", trust_remote_code=True)
    if eval_tokenizer.pad_token is None:
        eval_tokenizer.pad_token = eval_tokenizer.eos_token
    eval_model_hf = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, device_map={"": device_eval}, trust_remote_code=True)
    eval_model_hf.eval()

    def load_data(path, sample_size=None, seed=42):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if sample_size is not None and sample_size < len(data):
            random.seed(seed)
            data = random.sample(data, sample_size)
        return data

    forget_data = load_data(args.forget_file, args.eval_forget_size, args.seed)
    retain_data = load_data(args.retain_file, args.eval_retain_size, args.seed)

    print(f"Loaded {len(forget_data)} forget samples and {len(retain_data)} retain samples.")

    def evaluate_with_judge(eval_model, data, desc, batch_size):
        results = []
        correct_count = 0
        
        # Determine the maximum safe input padding strategy if tokenizer is left padded
        for i in tqdm(range(0, len(data), batch_size), desc=desc):
            batch = data[i:i+batch_size]
            questions = [item.get('question', item.get('instruction', '')) for item in batch]
            prompts = [item.get('prompt_used', item.get('instruction', '') + item.get('input', '')) for item in batch]
            
            golds = []
            for item in batch:
                if 'gold_answers' in item:
                    golds.append(item['gold_answers'][0])
                else:
                    golds.append(str(item.get('answer', '')))
                    
            gold_texts = [item.get('gold_option_text', item.get('output', '')) for item in batch]
            
            # 1. Base Model answers the question (Batched)
            inputs = eval_tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_input_length
            ).to(device_eval)
            with torch.no_grad():
                # Get lengths before generation to slice accurately
                input_lengths = inputs.input_ids.shape[1]
                outputs = eval_model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=eval_tokenizer.eos_token_id,
                    max_length=None
                )
            
            gen_tokens = outputs[:, input_lengths:]
            gen_texts = eval_tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
            gen_texts = [text.strip() for text in gen_texts]
            
            # 2. Heuristic passing + Judge Model evaluates the answers (Batched)
            judge_texts_inputs = []
            needs_judge = []
            for i, (question, gold, gold_text, gen_text) in enumerate(zip(questions, golds, gold_texts, gen_texts)):
                if judge_model_hf is None:
                    # Hard match only -> no judge model needed
                    needs_judge.append(False)
                    judge_texts_inputs.append("")
                else:
                    ans = str(gold).strip().replace(".0", "")
                    strict_match = (ans in gen_text) or re.match(r'^([A-Za-z])(?:[^a-zA-Z]|$)', gen_text)
                    if strict_match:
                        needs_judge.append(False)
                        judge_texts_inputs.append("") 
                    else:
                        needs_judge.append(True)
                        judge_prompt = build_judge_prompt(question, gold, gold_text, gen_text)
                        messages = [{"role": "user", "content": judge_prompt}]
                        chat_str = judge_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        judge_texts_inputs.append(chat_str)
                
            real_judge_inputs = [text for text, nj in zip(judge_texts_inputs, needs_judge) if nj]
            
            judge_results_texts_map = {}
            if len(real_judge_inputs) > 0 and judge_model_hf is not None:
                judge_inputs = judge_tokenizer(real_judge_inputs, return_tensors="pt", padding=True, truncation=True).to(device_judge)
                with torch.no_grad():
                    judge_input_lengths = judge_inputs.input_ids.shape[1]
                    judge_outputs = judge_model_hf.generate(
                        **judge_inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=judge_tokenizer.eos_token_id,
                        max_length=None
                    )
                judge_gen_tokens = judge_outputs[:, judge_input_lengths:]
                judge_real_results = judge_tokenizer.batch_decode(judge_gen_tokens, skip_special_tokens=True)
                
                idx = 0
                for i, nj in enumerate(needs_judge):
                    if nj:
                        judge_results_texts_map[i] = judge_real_results[idx]
                        idx += 1
            
            for i, (prompt, gold, gen_text, nj) in enumerate(zip(prompts, golds, gen_texts, needs_judge)):
                if not nj:
                    if judge_model_hf is None:
                        # Mixed hard match:
                        # - MCQ (gold in A/B/C/D): strict leading-letter parsing
                        # - Non-MCQ: normalized text containment check
                        gold_clean = str(gold).strip().replace('.0', '').upper()
                        if gold_clean in {"A", "B", "C", "D"}:
                            strict_match = re.search(r'([A-Za-z])', gen_text)
                            if strict_match:
                                extracted_letter = strict_match.group(1).upper()
                                is_correct = (extracted_letter == gold_clean)
                                judge_text = f"Hard match mode (MCQ). Extracted leading letter: {extracted_letter}."
                            else:
                                is_correct = False
                                judge_text = "Hard match mode (MCQ). Failed to extract a leading option letter."
                        else:
                            gen_norm = re.sub(r'\s+', ' ', gen_text.strip()).lower()
                            gold_norm = re.sub(r'\s+', ' ', str(gold).strip()).lower()
                            is_correct = (gold_norm != "") and (gold_norm in gen_norm)
                            judge_text = "Hard match mode (text). Normalized substring match."
                    else:
                        strict_match = re.match(r'^([A-Za-z])(?:[^a-zA-Z]|$)', gen_text)
                        extracted_letter = strict_match.group(1).upper()
                        is_correct = (extracted_letter == gold.upper())
                        judge_text = "Format strict match. No Judge required."
                else:
                    judge_text = judge_results_texts_map.get(i, "Timeout or error")
                    j_text_clean = judge_text.strip().upper()
                    is_correct = False
                    if "INCORRECT" in j_text_clean:
                        is_correct = False
                    elif "CORRECT" in j_text_clean:
                        is_correct = True
                    else:
                        is_correct = "TRUE" in j_text_clean or "YES" in j_text_clean

                if is_correct: 
                    correct_count += 1
                
                results.append({
                    "prompt": prompt,
                    "gold": gold,
                    "student_answer": gen_text,
                    "judge_explanation": judge_text,
                    "is_correct": is_correct
                })
        return correct_count, len(data), results

    base_forget_c = base_forget_t = base_retain_c = base_retain_t = 0
    base_forget_res = []
    base_retain_res = []
    
    if not args.lora_only:
        print("\n=== Evaluating Base Model ===")
        base_forget_c, base_forget_t, base_forget_res = evaluate_with_judge(eval_model_hf, forget_data, "Base Forget", args.batch_size)
        base_retain_c, base_retain_t, base_retain_res = evaluate_with_judge(eval_model_hf, retain_data, "Base Retain", args.batch_size)

    peft_forget_c = peft_forget_t = peft_retain_c = peft_retain_t = 0
    peft_forget_res = []
    peft_retain_res = []
    
    lora_hyperparams = {}
    training_args_dict = {}

    if args.lora_path and not args.base_only:
        print("\n>>> Applying LoRA Adapter from", args.lora_path)
        # Check if this is a DLL-style adapter with full-path keys in safetensors
        # (bypasses transformers v5 key-conversion bug in PEFT)
        _safetensors_path = os.path.join(args.lora_path, "adapter_model.safetensors")
        _config_path_check = os.path.join(args.lora_path, "adapter_config.json")
        _dll_manual = False
        if os.path.exists(_safetensors_path) and os.path.exists(_config_path_check):
            from safetensors import safe_open as _safe_open
            with _safe_open(_safetensors_path, framework="pt", device="cpu") as _sf:
                _keys = list(_sf.keys())
            # Detect full-path keys (our manually saved format)
            _has_full_path = any("model.layers." in k and ".lora_" in k for k in _keys)
            if _has_full_path:
                _dll_manual = True
                print(">>> Detected full-path LoRA keys, using direct weight merge (bypass PEFT v5 conversion)")
                import json as _json
                with open(_config_path_check, encoding="utf-8-sig") as _f:
                    _cfg = _json.load(_f)
                _alpha = _cfg.get("lora_alpha", 32)
                _rank = _cfg.get("r", 32)
                _scale = _alpha / _rank
                _layers_to_transform = _cfg.get("layers_to_transform", None)
                with _safe_open(_safetensors_path, framework="pt", device="cpu") as _sf:
                    _tensors = {k: _sf.get_tensor(k) for k in _keys}
                # Group by layer: apply lora_B @ lora_A scaled delta
                import re as _re
                _merged = {}
                for _k in _tensors:
                    _m = _re.match(r".*model\.layers\.(\d+)\.(.+)\.lora_([AB])\..*weight$", _k)
                    if _m:
                        _li, _mod, _ab = int(_m.group(1)), _m.group(2), _m.group(3)
                        _key = (_li, _mod)
                        if _key not in _merged:
                            _merged[_key] = {}
                        _merged[_key][_ab] = _tensors[_k]
                with torch.no_grad():
                    for (_li, _mod), _ab_dict in _merged.items():
                        if "A" in _ab_dict and "B" in _ab_dict:
                            _lA = _ab_dict["A"]
                            _lB = _ab_dict["B"]
                            _parts = _mod.split(".")
                            _layer = eval_model_hf.model.layers[_li]
                            _module = _layer
                            for _p in _parts:
                                _module = getattr(_module, _p)
                            _dW = (_scale * (_lB.to(_module.weight.dtype) @ _lA.to(_module.weight.dtype)))
                            _module.weight.data += _dW.to(_module.weight.device)
                                # Keep merge path quiet to avoid heavy terminal I/O slowdown.
                eval_model_hf.eval()

        if not _dll_manual:
            eval_model_hf = PeftModel.from_pretrained(eval_model_hf, args.lora_path)
            eval_model_hf = eval_model_hf.merge_and_unload()
            eval_model_hf.eval()

        print("\n=== Evaluating PEFT Model ===")
        peft_forget_c, peft_forget_t, peft_forget_res = evaluate_with_judge(eval_model_hf, forget_data, "PEFT Forget", args.batch_size)
        peft_retain_c, peft_retain_t, peft_retain_res = evaluate_with_judge(eval_model_hf, retain_data, "PEFT Retain", args.batch_size)

        # Fetch hyperparameters
        config_path = os.path.join(args.lora_path, "adapter_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8-sig") as f:
                lora_hyperparams = json.load(f)
                
        # Fetch tracked training_args.json (argparse parameters from lora_ga_train.py)
        train_args_path = os.path.join(args.lora_path, "training_args.json")
        if os.path.exists(train_args_path):
            with open(train_args_path, "r", encoding="utf-8") as f:
                training_args_dict = json.load(f)

    # Compile Final Report
    final_report = {
        "evaluation_config": {
            "eval_forget_size": args.eval_forget_size,
            "eval_retain_size": args.eval_retain_size,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "max_input_length": args.max_input_length,
            "lora_path": args.lora_path,
            "judge_model": "None (Hard match)" if judge_model_hf is None else args.judge_model
        },
        "lora_training_hyperparameters": training_args_dict,
        "adapter_config": lora_hyperparams,
        "summary": {
            "Base_Forget_Accuracy": f"{base_forget_c}/{base_forget_t} = {base_forget_c/base_forget_t:.2%}" if base_forget_t > 0 else "0/0",
            "Base_Retain_Accuracy": f"{base_retain_c}/{base_retain_t} = {base_retain_c/base_retain_t:.2%}" if base_retain_t > 0 else "0/0",
            "PEFT_Forget_Accuracy": f"{peft_forget_c}/{peft_forget_t} = {peft_forget_c/peft_forget_t:.2%}" if peft_forget_t > 0 else "0/0",
            "PEFT_Retain_Accuracy": f"{peft_retain_c}/{peft_retain_t} = {peft_retain_c/peft_retain_t:.2%}" if peft_retain_t > 0 else "0/0"
        },
        "details": {
            "Base_Forget": base_forget_res,
            "Base_Retain": base_retain_res,
            "PEFT_Forget": peft_forget_res,
            "PEFT_Retain": peft_retain_res
        }
    }

    os.makedirs(os.path.dirname(args.output_result), exist_ok=True)
    with open(args.output_result, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False)
    
    print(f"\nAll evaluation finished. Detailed results and judge explanations are written to {args.output_result}")
    for k, v in final_report["summary"].items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
