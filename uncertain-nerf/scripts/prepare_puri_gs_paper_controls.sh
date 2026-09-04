#!/usr/bin/env bash
set -euo pipefail
if [[ "$#" -ne 1 ]]; then
  echo "Usage: bash scripts/prepare_puri_gs_paper_controls.sh EXISTING_GSPLAT_DIR"
  exit 2
fi
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="$1"
PATCH="${ROOT_DIR}/patches/gsplat_v1.5.3_puri_gs_paper_controls.patch"
test -d "${TARGET_DIR}/.git"
test "$(git -C "${TARGET_DIR}" rev-parse HEAD)" = "937e29912570c372bed6747a5c9bf85fed877bae"
if git -C "${TARGET_DIR}" apply --reverse --check --ignore-whitespace "$PATCH" >/dev/null 2>&1; then
  echo "PURI-GS-PAPER-CONTROLS-PATCH-READY"
  exit 0
fi
# The existing pinned checkout must already contain the audited causal stack.
git -C "${TARGET_DIR}" apply --reverse --check --ignore-whitespace \
  "${ROOT_DIR}/patches/gsplat_v1.5.3_puri_gs_garden_causal.patch"
git -C "${TARGET_DIR}" apply --check --ignore-whitespace "$PATCH"
git -C "${TARGET_DIR}" apply --ignore-whitespace "$PATCH"
git -C "${TARGET_DIR}" apply --reverse --check --ignore-whitespace "$PATCH"
echo "PURI-GS-PAPER-CONTROLS-PATCH-READY"
