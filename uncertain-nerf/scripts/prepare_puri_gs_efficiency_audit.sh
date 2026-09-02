#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: bash scripts/prepare_puri_gs_efficiency_audit.sh GSPLAT_DIR"
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GSPLAT_DIR="$1"
FULL_PATCH="${ROOT_DIR}/patches/gsplat_v1.5.3_puri_gs_efficiency_audit.patch"
UPGRADE_PATCH="${ROOT_DIR}/patches/gsplat_v1.5.3_puri_gs_efficiency_no_images_upgrade.patch"
EXPECTED_COMMIT="937e29912570c372bed6747a5c9bf85fed877bae"

if [[ ! -d "${GSPLAT_DIR}/.git" ]] || \
  [[ "$(git -C "${GSPLAT_DIR}" rev-parse HEAD)" != "${EXPECTED_COMMIT}" ]]; then
  echo "EFFICIENCY-PREPARE-INVALID-GSPLAT"
  exit 3
fi

if git -C "${GSPLAT_DIR}" apply --reverse --check --ignore-whitespace \
  "${FULL_PATCH}" >/dev/null 2>&1; then
  echo "EFFICIENCY-AUDIT-PATCH-READY"
  exit 0
fi

if git -C "${GSPLAT_DIR}" apply --check --ignore-whitespace \
  "${UPGRADE_PATCH}" >/dev/null 2>&1; then
  git -C "${GSPLAT_DIR}" apply --ignore-whitespace "${UPGRADE_PATCH}"
elif git -C "${GSPLAT_DIR}" apply --check --ignore-whitespace \
  "${FULL_PATCH}" >/dev/null 2>&1; then
  git -C "${GSPLAT_DIR}" apply --ignore-whitespace "${FULL_PATCH}"
else
  echo "EFFICIENCY-PREPARE-UNRECOGNIZED-PATCH-STATE"
  exit 3
fi

git -C "${GSPLAT_DIR}" apply --reverse --check --ignore-whitespace "${FULL_PATCH}"
echo "EFFICIENCY-AUDIT-PATCH-READY"
