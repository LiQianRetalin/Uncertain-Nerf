#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: bash scripts/prepare_puri_gs_ru.sh TARGET_DIR"
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="$1"
RU_PATCH="${ROOT_DIR}/patches/gsplat_v1.5.3_puri_gs_ru.patch"
CAUSAL_PATCH="${ROOT_DIR}/patches/gsplat_v1.5.3_puri_gs_garden_causal.patch"
EXPECTED_COMMIT="937e29912570c372bed6747a5c9bf85fed877bae"

if [[ -d "${TARGET_DIR}/.git" ]] && \
  [[ "$(git -C "${TARGET_DIR}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] && \
  git -C "${TARGET_DIR}" apply --reverse --check --ignore-whitespace \
    "${CAUSAL_PATCH}" >/dev/null 2>&1; then
  echo "PURI-GS-GARDEN-CAUSAL-PATCH-READY"
  exit 0
fi

# The historical efficiency/On-the-go stacks are RU supersets, so the complete
# RU patch is no longer reverse-applicable even though all RU capabilities exist.
# The causal patch has deliberately isolated hunks and can be added directly.
if [[ -d "${TARGET_DIR}/.git" ]] && \
  [[ "$(git -C "${TARGET_DIR}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] && \
  grep -q "puri_gs_ru_enabled" \
    "${TARGET_DIR}/examples/simple_trainer.py" 2>/dev/null && \
  git -C "${TARGET_DIR}" apply --check --ignore-whitespace \
    "${CAUSAL_PATCH}" >/dev/null 2>&1; then
  git -C "${TARGET_DIR}" apply --ignore-whitespace "${CAUSAL_PATCH}"
  git -C "${TARGET_DIR}" apply --reverse --check --ignore-whitespace \
    "${CAUSAL_PATCH}"
  echo "PURI-GS-GARDEN-CAUSAL-PATCH-READY"
  exit 0
fi

if ! [[ -d "${TARGET_DIR}/.git" ]] || \
  [[ "$(git -C "${TARGET_DIR}" rev-parse HEAD)" != "${EXPECTED_COMMIT}" ]] || \
  ! git -C "${TARGET_DIR}" apply --reverse --check --ignore-whitespace \
    "${RU_PATCH}" >/dev/null 2>&1; then
  bash "${ROOT_DIR}/scripts/prepare_gsplat_robot_baseline.sh" "${TARGET_DIR}"

  if git -C "${TARGET_DIR}" apply --check --ignore-whitespace \
    "${RU_PATCH}" >/dev/null 2>&1; then
    git -C "${TARGET_DIR}" apply --ignore-whitespace "${RU_PATCH}"
  else
    echo "PURI-GS-RU patch cannot be applied cleanly; stop and inspect TARGET_DIR."
    exit 3
  fi
fi

git -C "${TARGET_DIR}" apply --reverse --check --ignore-whitespace "${RU_PATCH}"
if git -C "${TARGET_DIR}" apply --check --ignore-whitespace \
  "${CAUSAL_PATCH}" >/dev/null 2>&1; then
  git -C "${TARGET_DIR}" apply --ignore-whitespace "${CAUSAL_PATCH}"
else
  echo "Garden causal patch cannot be applied cleanly; stop and inspect TARGET_DIR."
  exit 3
fi

git -C "${TARGET_DIR}" apply --reverse --check --ignore-whitespace \
  "${CAUSAL_PATCH}"
echo "PURI-GS-GARDEN-CAUSAL-PATCH-READY"
