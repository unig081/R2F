#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R2F_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export R2F_ROOT
source "${SCRIPT_DIR}/env.sh"

CONFIG="${1:-${R2F_ROOT}/configs/r2f_tofu_05_eval.yaml}"
LOG_DIR="${R2F_ROOT}/logs"
mkdir -p "${LOG_DIR}"

python -m r2f_tofu.evaluate_tofu --config "${CONFIG}" \
  2>&1 | tee "${LOG_DIR}/full_05_eval.log"

