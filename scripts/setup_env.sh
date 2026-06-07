#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R2F_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export R2F_ROOT
source "${SCRIPT_DIR}/env.sh"
export PYTHONNOUSERSITE=1
export PIP_USER=false

CONDA_SH="${CONDA_SH:-/opt/anaconda3/etc/profile.d/conda.sh}"
if [ ! -f "${CONDA_SH}" ]; then
  CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
fi
source "${CONDA_SH}"

if ! conda env list | awk -v env="${R2F_CONDA_ENV_NAME}" '$1 == env {found=1} END {exit !found}'; then
  conda create -y -n "${R2F_CONDA_ENV_NAME}" python=3.11 pip
fi

conda env update -n "${R2F_CONDA_ENV_NAME}" -f "${R2F_ROOT}/environment.yml" --prune
set +u
conda activate "${R2F_CONDA_ENV_NAME}"
set -u
python -m pip install --no-user -e "${R2F_ROOT}"
python - <<'PY'
import torch
print("python ok")
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device_count", torch.cuda.device_count())
PY
