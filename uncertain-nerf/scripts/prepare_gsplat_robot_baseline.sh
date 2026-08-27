#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: bash scripts/prepare_gsplat_robot_baseline.sh TARGET_DIR"
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="$1"
PATCH_PATH="${ROOT_DIR}/patches/gsplat_v1.5.3_robot_screen.patch"
EXPECTED_COMMIT="937e29912570c372bed6747a5c9bf85fed877bae"

if [[ ! -e "${TARGET_DIR}" ]]; then
  git clone --depth 1 --branch v1.5.3 --recurse-submodules \
    https://github.com/nerfstudio-project/gsplat.git "${TARGET_DIR}"
elif [[ ! -d "${TARGET_DIR}/.git" ]]; then
  echo "TARGET_DIR exists but is not a git checkout: ${TARGET_DIR}"
  exit 2
fi

ACTUAL_COMMIT="$(git -C "${TARGET_DIR}" rev-parse HEAD)"
if [[ "${ACTUAL_COMMIT}" != "${EXPECTED_COMMIT}" ]]; then
  echo "Expected gsplat ${EXPECTED_COMMIT}, got ${ACTUAL_COMMIT}"
  exit 2
fi

git -C "${TARGET_DIR}" submodule update --init --recursive --depth 1
if [[ ! -f "${TARGET_DIR}/gsplat/cuda/csrc/third_party/glm/glm/glm.hpp" ]]; then
  echo "Required gsplat GLM submodule is incomplete"
  exit 2
fi

if git -C "${TARGET_DIR}" apply --reverse --check --ignore-whitespace \
    "${PATCH_PATH}" >/dev/null 2>&1; then
  echo "Robot screen patch is already applied."
else
  git -C "${TARGET_DIR}" apply --check --ignore-whitespace "${PATCH_PATH}"
  git -C "${TARGET_DIR}" apply --ignore-whitespace "${PATCH_PATH}"
  echo "Applied robot screen split patch."
fi

echo "Prepared gsplat v1.5.3 at ${TARGET_DIR}"
