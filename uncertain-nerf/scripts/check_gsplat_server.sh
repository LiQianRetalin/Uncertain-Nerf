#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "Usage: bash scripts/check_gsplat_server.sh DATA_DIR GPU_ID"
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$1"
GPU_ID="$2"
AUDIT_VENV="${PURI_AUDIT_VENV:-${ROOT_DIR}/.venv-v7}"

if ! [[ "${GPU_ID}" =~ ^[0-9]+$ ]]; then
  echo "GPU_ID must be one non-negative integer"
  exit 2
fi
if [[ ! -x "${AUDIT_VENV}/bin/python" ]]; then
  echo "Existing audit environment is missing: ${AUDIT_VENV}/bin/python"
  exit 2
fi
for command_name in git nvidia-smi nvcc g++; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Required server command is missing: ${command_name}"
    exit 2
  fi
done

cd "${ROOT_DIR}"
"${AUDIT_VENV}/bin/python" run_v8_dataset_audit.py \
  --data "${DATA_DIR}" \
  --output tmp/fern_dataset_audit.json

nvidia-smi -i "${GPU_ID}" \
  --query-gpu=index,name,memory.total,memory.free,driver_version \
  --format=csv,noheader
nvcc --version | tail -n 1
echo "g++=$(g++ -dumpfullversion -dumpversion)"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${AUDIT_VENV}/bin/python" - <<'PY'
import sys

import torch

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access the selected CUDA GPU")
print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"visible_gpu={torch.cuda.get_device_name(0)}")
PY
