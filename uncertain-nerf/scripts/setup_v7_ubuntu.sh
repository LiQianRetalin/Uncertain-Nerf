#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv-v7}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
sudo apt update
sudo apt install -y python3-venv python3-dev build-essential git ffmpeg
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision --index-url "${TORCH_INDEX_URL}"
python -m pip install -r "${ROOT_DIR}/requirements.txt"
python -m pip install -r "${ROOT_DIR}/requirements-eval-v6.txt"
if [[ "${INSTALL_TCNN:-0}" == "1" ]]; then
  command -v nvcc >/dev/null 2>&1 || { echo "INSTALL_TCNN=1 requires nvcc" >&2; exit 3; }
  python -m pip install ninja
  python -m pip install "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
fi
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available(): print("GPU:", torch.cuda.get_device_name(0))
PY
echo "V7 environment ready: ${VENV_DIR}"
