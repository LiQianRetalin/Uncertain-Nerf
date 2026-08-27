#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: bash scripts/setup_gsplat_robot_env.sh GPU_ID"
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${ROOT_DIR}/.venv-gsplat153"
PYTHON_BIN="${ENV_DIR}/bin/python"
GPU_ID="$1"
OFFLINE_DIR="${PURI_GSPLAT_OFFLINE_DIR:-${ROOT_DIR}/tmp/gsplat153-offline}"
WHEEL_NAME="gsplat-1.5.3+pt24cu121-cp310-cp310-linux_x86_64.whl"
WHEEL_PATH="${OFFLINE_DIR}/${WHEEL_NAME}"
WHEEL_SHA256="0493bab68ed5fc71f4ce8bfc2be03b584d8a41a06a6d9362e09a795340f8c488"

if ! [[ "${GPU_ID}" =~ ^[0-9]+$ ]]; then
  echo "GPU_ID must be one non-negative integer"
  exit 2
fi
for command_name in conda g++ nvcc sha256sum; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Required command is missing: ${command_name}"
    exit 2
  fi
done
for required_path in \
  "${WHEEL_PATH}" \
  "${OFFLINE_DIR}/pycolmap/pyproject.toml" \
  "${OFFLINE_DIR}/nerfview/pyproject.toml" \
  "${OFFLINE_DIR}/fused-ssim/setup.py"; do
  if [[ ! -f "${required_path}" ]]; then
    echo "Offline bundle is incomplete: ${required_path}"
    exit 2
  fi
done
if ! echo "${WHEEL_SHA256}  ${WHEEL_PATH}" | sha256sum --check --status; then
  echo "Offline gsplat wheel checksum failed"
  exit 2
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  conda create --prefix "${ENV_DIR}" python=3.10 pip -y
fi
if [[ "$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != "3.10" ]]; then
  echo "Existing isolated environment is not Python 3.10: ${ENV_DIR}"
  exit 2
fi

"${PYTHON_BIN}" -m pip install \
  "setuptools==75.1.0" "wheel==0.44.0" "ninja==1.11.1.1"
"${PYTHON_BIN}" -m pip install \
  "torch==2.4.0" "torchvision==0.19.0" \
  --index-url https://download.pytorch.org/whl/cu121
"${PYTHON_BIN}" -m pip install \
  -r "${ROOT_DIR}/configs/gsplat153_robot_requirements.txt"
"${PYTHON_BIN}" -m pip install \
  "${OFFLINE_DIR}/pycolmap" "${OFFLINE_DIR}/nerfview"

CUDA_VISIBLE_DEVICES="${GPU_ID}" MAX_JOBS=4 \
  "${PYTHON_BIN}" -m pip install --no-build-isolation \
  "${OFFLINE_DIR}/fused-ssim"

"${PYTHON_BIN}" -m pip install --no-deps "${WHEEL_PATH}"
"${PYTHON_BIN}" -m pip check

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" \
  "${ROOT_DIR}/scripts/verify_gsplat_robot_install.py"

echo "Prepared isolated environment: ${ENV_DIR}"
