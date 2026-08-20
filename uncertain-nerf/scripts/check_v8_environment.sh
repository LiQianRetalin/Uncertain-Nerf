#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-6}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv-v7}"

cd "${ROOT_DIR}"
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "STOP_ENVIRONMENT: Python is missing at ${VENV_DIR}/bin/python"
  echo "Set VENV_DIR to the verified historical environment; do not install into it yet."
  exit 3
fi

source "${VENV_DIR}/bin/activate"
echo "repository=${ROOT_DIR}"
echo "python=$(command -v python)"
python --version
python - <<'PY'
import importlib

required = (
    "torch", "numpy", "cv2", "imageio", "configargparse", "tqdm", "pytest",
    "skimage", "lpips"
)
for name in required:
    module = importlib.import_module(name)
    print(f"{name}={getattr(module, '__version__', 'installed')}")
PY

echo "physical_gpu=${GPU_ID}"
nvidia-smi -i "${GPU_ID}" \
  --query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu \
  --format=csv
GPU_STATE="$(nvidia-smi -i "${GPU_ID}" \
  --query-gpu=memory.used,utilization.gpu \
  --format=csv,noheader,nounits)"
IFS=',' read -r GPU_MEMORY_USED GPU_UTILIZATION <<< "${GPU_STATE}"
GPU_MEMORY_USED="${GPU_MEMORY_USED//[[:space:]]/}"
GPU_UTILIZATION="${GPU_UTILIZATION//[[:space:]]/}"
if (( GPU_MEMORY_USED > 1024 || GPU_UTILIZATION > 5 )); then
  echo "STOP_GPU_BUSY: physical GPU ${GPU_ID} is not idle " \
       "(${GPU_MEMORY_USED} MiB, ${GPU_UTILIZATION}% utilization)"
  exit 5
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" python - <<'PY'
import os
import torch

print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"torch_cuda_available={torch.cuda.is_available()}")
print(f"torch_visible_device_count={torch.cuda.device_count()}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("STOP_GPU_MAPPING: the process must see exactly one CUDA device")
print(f"logical_cuda_0={torch.cuda.get_device_name(0)}")
print(f"torch_cuda={torch.version.cuda}")
PY

echo "V8 diagnostic environment check passed."
