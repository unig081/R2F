#!/usr/bin/env bash
# This file is meant to be sourced. Do not mutate the caller's shell options.

export R2F_ROOT="${R2F_ROOT:-/mnt/data1/zxc/R2F}"
export R2F_CONDA_ENV_NAME="${R2F_CONDA_ENV_NAME:-zxc_r2f}"
export R2F_CONDA_ENV="${R2F_CONDA_ENV:-/mnt/data1/conda_env/${R2F_CONDA_ENV_NAME}}"
export R2F_OUTPUT_DIR="${R2F_OUTPUT_DIR:-${R2F_ROOT}/results}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${R2F_ROOT}/cache/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${R2F_ROOT}/cache/transformers}"
export HF_HOME="${HF_HOME:-${R2F_ROOT}/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${R2F_ROOT}/cache/torch}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${R2F_ROOT}/cache/pip}"
export PYTHONNOUSERSITE=1
export PIP_USER=false
export PYTHONPATH="${R2F_ROOT}/src:${PYTHONPATH:-}"
export R2F_MODEL_FAMILY="${R2F_MODEL_FAMILY:-llama}"

set_model_path_if_present() {
  var_name="$1"
  shift
  for p in "$@"; do
    eval "current_value=\${${var_name}:-}"
    if [ -f "${p}/config.json" ] && [ -z "${current_value}" ]; then
      export "${var_name}=${p}"
    fi
  done
}

case "${R2F_MODEL_FAMILY}" in
  llama|llama3|llama3.2|llama_3_2)
    set_model_path_if_present R2F_SOURCE_MODEL \
      "${R2F_ROOT}/model/proxy/llama3.2_1B" \
      "${R2F_ROOT}/models/llama_3_2_1B_instruct_tofu" \
      "/mnt/data1/zxc/models/llama_3_2_1B_instruct_tofu" \
      "/mnt/data1/zxc/handoff/junior_llama_tofu_eval_bundle/models/llama_3_2_1B_instruct_tofu" \
      "/mnt/data1/zxc/zxc_r2f/models/llama_3_2_1B_instruct_tofu"
    set_model_path_if_present R2F_TARGET_MODEL \
      "${R2F_ROOT}/model/target/llama3.2_3B" \
      "${R2F_ROOT}/models/llama_3_2_3B_instruct_tofu" \
      "/mnt/data1/zxc/models/llama_3_2_3B_instruct_tofu" \
      "/mnt/data1/zxc/handoff/junior_llama_tofu_eval_bundle/models/llama_3_2_3B_instruct_tofu" \
      "/mnt/data1/zxc/zxc_r2f/models/llama_3_2_3B_instruct_tofu"
    ;;
  phi|phi4)
    set_model_path_if_present R2F_SOURCE_MODEL \
      "${R2F_ROOT}/model/proxy/phi4_3B" \
      "/mnt/data1/zxc/models/phi4_3B" \
      "/mnt/data1/zxc/handoff/models/phi4_3B"
    set_model_path_if_present R2F_TARGET_MODEL \
      "${R2F_ROOT}/model/target/phi4_14B" \
      "/mnt/data1/zxc/models/phi4_14B" \
      "/mnt/data1/zxc/handoff/models/phi4_14B"
    ;;
  qwen|qwen3)
    set_model_path_if_present R2F_SOURCE_MODEL \
      "${R2F_ROOT}/model/proxy/qwen3_1.7B" \
      "/mnt/data1/zxc/models/qwen3_1.7B" \
      "/mnt/data1/zxc/handoff/models/qwen3_1.7B"
    set_model_path_if_present R2F_TARGET_MODEL \
      "${R2F_ROOT}/model/target/qwen3_8B" \
      "/mnt/data1/zxc/models/qwen3_8B" \
      "/mnt/data1/zxc/handoff/models/qwen3_8B"
    ;;
  *)
    echo "Unsupported R2F_MODEL_FAMILY=${R2F_MODEL_FAMILY}; expected llama, phi, or qwen" >&2
    return 1 2>/dev/null || exit 1
    ;;
esac

for p in \
  "${R2F_ROOT}/dataset/TOFU/forget05.json" \
  "/mnt/data1/zxc/datasets/tofu/forget05.json" \
  "/mnt/data1/zxc/handoff/junior_llama_tofu_eval_bundle/datasets/tofu/forget05.json" \
  "/mnt/data1/zxc/zxc_r2f/datasets/tofu/forget05.json"; do
  if [ -f "${p}" ] && [ -z "${R2F_FORGET_FILE:-}" ]; then
    export R2F_FORGET_FILE="${p}"
  fi
done

for p in \
  "${R2F_ROOT}/dataset/TOFU/retain95.json" \
  "/mnt/data1/zxc/datasets/tofu/retain95.json" \
  "/mnt/data1/zxc/handoff/junior_llama_tofu_eval_bundle/datasets/tofu/retain95.json" \
  "/mnt/data1/zxc/zxc_r2f/datasets/tofu/retain95.json"; do
  if [ -f "${p}" ] && [ -z "${R2F_RETAIN_FILE:-}" ]; then
    export R2F_RETAIN_FILE="${p}"
  fi
done

mkdir -p \
  "${HF_DATASETS_CACHE}" \
  "${TRANSFORMERS_CACHE}" \
  "${HF_HOME}" \
  "${TORCH_HOME}" \
  "${PIP_CACHE_DIR}" \
  "${R2F_ROOT}/logs" \
  "${R2F_ROOT}/reports" \
  "${R2F_ROOT}/results"
