#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: bash scripts/prepare_puri_gs_ontogo.sh GSPLAT_DIR"
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GSPLAT_DIR="$1"
ONTOGO_PATCH="${ROOT_DIR}/patches/gsplat_v1.5.3_puri_gs_ontogo.patch"
EXPECTED_COMMIT="937e29912570c372bed6747a5c9bf85fed877bae"

if [[ ! -d "${GSPLAT_DIR}/.git" ]] || \
  [[ "$(git -C "${GSPLAT_DIR}" rev-parse HEAD)" != "${EXPECTED_COMMIT}" ]]; then
  echo "ONTOGO-PREPARE-INVALID-GSPLAT"
  exit 3
fi

if git -C "${GSPLAT_DIR}" apply --reverse --check --ignore-whitespace \
  "${ONTOGO_PATCH}" >/dev/null 2>&1; then
  echo "ONTOGO-PATIO-HIGH-PATCH-READY"
  exit 0
fi

bash "${ROOT_DIR}/scripts/prepare_puri_gs_efficiency_audit.sh" "${GSPLAT_DIR}"

if git -C "${GSPLAT_DIR}" apply --check --ignore-whitespace \
  "${ONTOGO_PATCH}" >/dev/null 2>&1; then
  git -C "${GSPLAT_DIR}" apply --ignore-whitespace "${ONTOGO_PATCH}"
else
  echo "ONTOGO-PREPARE-UNRECOGNIZED-PATCH-STATE"
  exit 3
fi

git -C "${GSPLAT_DIR}" apply --reverse --check --ignore-whitespace \
  "${ONTOGO_PATCH}"
echo "ONTOGO-PATIO-HIGH-PATCH-READY"
