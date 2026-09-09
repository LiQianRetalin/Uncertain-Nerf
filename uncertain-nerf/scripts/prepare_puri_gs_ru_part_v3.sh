#!/usr/bin/env bash
set -euo pipefail
if [[ "$#" -ne 1 ]]; then
  echo 'Usage: bash scripts/prepare_puri_gs_ru_part_v3.sh GSPLAT_V3_DIR'
  exit 2
fi
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$1"
PATCH="$ROOT_DIR/patches/gsplat_v1.5.3_ru_part_v3.patch"
if [[ -f "$TARGET/examples/simple_trainer.py" ]] && grep -q 'ru_v3_mode:' "$TARGET/examples/simple_trainer.py"; then
  git -C "$TARGET" apply --reverse --check --ignore-whitespace "$PATCH"
  echo V3_PATCH_READY
  exit 0
fi
# A separate pinned example checkout keeps the old RU/PART example behavior intact.
if [[ -f "$TARGET/examples/simple_trainer.py" ]] && grep -q 'puri_gs_ru_part_enabled' "$TARGET/examples/simple_trainer.py"; then
  echo 'Refusing to overlay V3 onto an old PART checkout; use the dedicated V3 directory.'
  exit 3
fi
bash "$ROOT_DIR/scripts/prepare_puri_gs_ru.sh" "$TARGET"
bash "$ROOT_DIR/scripts/prepare_puri_gs_efficiency_audit.sh" "$TARGET"
git -C "$TARGET" apply --check --ignore-whitespace "$PATCH"
git -C "$TARGET" apply --ignore-whitespace "$PATCH"
git -C "$TARGET" apply --reverse --check --ignore-whitespace "$PATCH"
echo V3_PATCH_READY
