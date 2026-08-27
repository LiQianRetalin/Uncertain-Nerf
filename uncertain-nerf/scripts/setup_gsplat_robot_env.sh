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
WHEEL_NAME="gsplat-1.5.3+pt24cu121-cp310-cp310-linux_x86_64.whl"
WHEEL_PATH="${ROOT_DIR}/tmp/${WHEEL_NAME}"
WHEEL_URL="https://github.com/nerfstudio-project/gsplat/releases/download/v1.5.3/gsplat-1.5.3%2Bpt24cu121-cp310-cp310-linux_x86_64.whl"
WHEEL_SHA256="0493bab68ed5fc71f4ce8bfc2be03b584d8a41a06a6d9362e09a795340f8c488"
FUSED_SSIM="git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5"

if ! [[ "${GPU_ID}" =~ ^[0-9]+$ ]]; then
  echo "GPU_ID must be one non-negative integer"
  exit 2
fi
for command_name in conda curl git nvcc sha256sum; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Required command is missing: ${command_name}"
    exit 2
  fi
done

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

CUDA_VISIBLE_DEVICES="${GPU_ID}" MAX_JOBS=4 \
  "${PYTHON_BIN}" -m pip install --no-build-isolation "${FUSED_SSIM}"

mkdir -p "${ROOT_DIR}/tmp"
if [[ -f "${WHEEL_PATH}" ]] && \
  echo "${WHEEL_SHA256}  ${WHEEL_PATH}" | sha256sum --check --status; then
  echo "Reusing verified gsplat wheel: ${WHEEL_PATH}"
else
  curl --fail --location --retry 2 --output "${WHEEL_PATH}.part" "${WHEEL_URL}"
  echo "${WHEEL_SHA256}  ${WHEEL_PATH}.part" | sha256sum --check --status
  mv "${WHEEL_PATH}.part" "${WHEEL_PATH}"
fi
"${PYTHON_BIN}" -m pip install --no-deps "${WHEEL_PATH}"
"${PYTHON_BIN}" -m pip check

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" \
  "${ROOT_DIR}/scripts/verify_gsplat_robot_install.py"

echo "Prepared isolated environment: ${ENV_DIR}"
