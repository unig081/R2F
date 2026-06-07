#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R2F_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export R2F_ROOT
source "${SCRIPT_DIR}/env.sh"

CONFIG="${1:-${R2F_ROOT}/configs/r2f_tofu_llama.yaml}"
LOG_DIR="${R2F_ROOT}/logs"
mkdir -p "${LOG_DIR}"

python -m r2f_tofu.unlearn_lora --config "${CONFIG}" --smoke \
  2>&1 | tee "${LOG_DIR}/smoke_01_lora_3b.log"
python -m r2f_tofu.unlearn_dense --config "${CONFIG}" --smoke \
  2>&1 | tee "${LOG_DIR}/smoke_02_dense_1b_samples.log"
python -m r2f_tofu.train_decoder --config "${CONFIG}" --smoke \
  2>&1 | tee "${LOG_DIR}/smoke_03_train_decoder.log"
python -m r2f_tofu.apply_r2f --config "${CONFIG}" --smoke \
  2>&1 | tee "${LOG_DIR}/smoke_04_apply_r2f.log"
python -m r2f_tofu.evaluate_tofu --config "${CONFIG}" --smoke \
  2>&1 | tee "${LOG_DIR}/smoke_05_eval.log"
