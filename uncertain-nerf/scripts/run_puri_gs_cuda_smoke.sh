#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "Usage: bash scripts/run_puri_gs_cuda_smoke.sh DATA_DIR RESULT_ROOT GPU_ID"
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PURI_GSPLAT_PYTHON:-${ROOT_DIR}/.venv-gsplat153/bin/python}"
GSPLAT_DIR="${PURI_GSPLAT_DIR:-${ROOT_DIR}/external/gsplat-v1.5.3}"
if [[ ! -d "$1" ]]; then
  echo "Dataset directory is missing: $1"
  exit 3
fi
DATA_DIR="$(realpath "$1")"
RESULT_ROOT_INPUT="$2"
GPU_ID="$3"

if [[ "${RESULT_ROOT_INPUT}" = /* ]]; then
  RESULT_ROOT="${RESULT_ROOT_INPUT}"
else
  RESULT_ROOT="${ROOT_DIR}/${RESULT_ROOT_INPUT#./}"
fi

if ! [[ "${GPU_ID}" =~ ^[0-9]+$ ]]; then
  echo "GPU_ID must be one non-negative integer"
  exit 2
fi
if [[ "$(git -C "${ROOT_DIR}" branch --show-current)" != "dev" ]]; then
  echo "STOP_BRANCH: server checkout is not on dev"
  exit 3
fi
if [[ -n "$(git -C "${ROOT_DIR}" status --short)" ]]; then
  echo "STOP_DIRTY_WORKTREE: inspect server changes before running smoke"
  git -C "${ROOT_DIR}" status --short
  exit 3
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Pinned Python is missing: ${PYTHON_BIN}"
  exit 3
fi
if [[ ! -d "${GSPLAT_DIR}" ]]; then
  echo "Pinned gsplat source is missing: ${GSPLAT_DIR}"
  exit 3
fi
if [[ ! -d "${DATA_DIR}/images_4" ]]; then
  echo "Dataset input is missing: ${DATA_DIR}/images_4"
  exit 3
fi
for required in \
  "${DATA_DIR}/sparse/0/cameras.bin" \
  "${DATA_DIR}/sparse/0/images.bin" \
  "${DATA_DIR}/sparse/0/points3D.bin"; do
  if [[ ! -s "${required}" ]]; then
    echo "Dataset input is missing: ${required}"
    exit 3
  fi
done
IMAGE_COUNT="$(find "${DATA_DIR}/images_4" -maxdepth 1 -type f | wc -l)"
if [[ "${IMAGE_COUNT}" -ne 20 ]]; then
  echo "Expected Fern images_4 count 20, got ${IMAGE_COUNT}"
  exit 3
fi
if [[ -e "${RESULT_ROOT}" ]]; then
  echo "Result root already exists; refusing to overwrite: ${RESULT_ROOT}"
  exit 3
fi

GPU_NAME="$(
  nvidia-smi -i "${GPU_ID}" --query-gpu=name --format=csv,noheader | xargs
)"
GPU_STATE="$(
  nvidia-smi -i "${GPU_ID}" \
    --query-gpu=memory.used,utilization.gpu \
    --format=csv,noheader,nounits | tr -d ' '
)"
IFS=',' read -r GPU_MEMORY_USED GPU_UTILIZATION <<<"${GPU_STATE}"
echo "physical_gpu=${GPU_ID} name=${GPU_NAME} memory_used_mib=${GPU_MEMORY_USED} utilization_percent=${GPU_UTILIZATION}"
if [[ "${GPU_NAME}" != *"NVIDIA L20"* ]]; then
  echo "STOP_GPU_MODEL: expected NVIDIA L20"
  exit 3
fi
if (( GPU_MEMORY_USED > 1024 || GPU_UTILIZATION > 5 )); then
  echo "STOP_GPU_BUSY: select an idle L20 before smoke"
  exit 3
fi

mkdir -p "${RESULT_ROOT}"
git -C "${ROOT_DIR}" rev-parse HEAD >"${RESULT_ROOT}/git_commit.txt"
printf '%s\n' "${GPU_ID}" >"${RESULT_ROOT}/physical_gpu.txt"

echo "Preparing the pinned external gsplat source without overwriting conflicts..."
bash "${ROOT_DIR}/scripts/prepare_gsplat_robot_baseline.sh" "${GSPLAT_DIR}" \
  2>&1 | tee "${RESULT_ROOT}/gsplat_patch.log"

echo "Verifying pinned environment and real CUDA forward/backward..."
CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH="${ROOT_DIR}" "${PYTHON_BIN}" \
  "${ROOT_DIR}/scripts/verify_gsplat_robot_install.py" \
  --expected-gpu "NVIDIA L20" \
  2>&1 | tee "${RESULT_ROOT}/environment_verify.log"

GSPLAT_RUNTIME_PATH="$(
  CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH="${ROOT_DIR}" "${PYTHON_BIN}" \
    -c 'import pathlib, gsplat; print(pathlib.Path(gsplat.__file__).resolve())'
)"
echo "gsplat_runtime_path=${GSPLAT_RUNTIME_PATH}" | \
  tee "${RESULT_ROOT}/gsplat_runtime_path.txt"
if [[ "${GSPLAT_RUNTIME_PATH}" == "${GSPLAT_DIR}"/* ]]; then
  echo "STOP_GSPLAT_SHADOWED: trainer would import source instead of the fixed wheel"
  exit 3
fi

run_training() {
  local label="$1"
  local config="$2"
  shift 2
  local result_dir="${RESULT_ROOT}/${label}"
  echo "Running ${label} 10-step training..."
  PYTHONUNBUFFERED=1 "${PYTHON_BIN}" "${ROOT_DIR}/run_puri_gs.py" \
    --config "${ROOT_DIR}/configs/${config}" \
    --gsplat-dir "${GSPLAT_DIR}" \
    --data-dir "${DATA_DIR}" \
    --result-dir "${result_dir}" \
    --gpu "${GPU_ID}" \
    --max-steps 10 \
    --data-factor 4 \
    "$@" \
    2>&1 | tee "${RESULT_ROOT}/${label}.train.log"
  test -s "${result_dir}/ckpts/ckpt_9_rank0.pt"
  test -s "${result_dir}/stats/train_step0009_rank0.json"
  test -s "${result_dir}/dataset_split.json"
}

run_evaluation() {
  local label="$1"
  local config="$2"
  local train_dir="${RESULT_ROOT}/${label}"
  local eval_dir="${RESULT_ROOT}/${label}_eval"
  echo "Loading and evaluating ${label} checkpoint..."
  PYTHONUNBUFFERED=1 "${PYTHON_BIN}" "${ROOT_DIR}/run_puri_gs.py" \
    --config "${ROOT_DIR}/configs/${config}" \
    --gsplat-dir "${GSPLAT_DIR}" \
    --data-dir "${DATA_DIR}" \
    --result-dir "${eval_dir}" \
    --gpu "${GPU_ID}" \
    --data-factor 4 \
    --checkpoint "${train_dir}/ckpts/ckpt_9_rank0.pt" \
    2>&1 | tee "${RESULT_ROOT}/${label}.eval.log"
  test -s "${eval_dir}/stats/test_step0009.json"
  test -s "${eval_dir}/dataset_split.json"
}

run_training b0 puri_gs_b0_default.yaml
run_training b1 puri_gs_b1_absgrad.yaml
run_training a1 puri_gs_a1_responsibility.yaml \
  --responsibility-start-step 3

test -s "${RESULT_ROOT}/a1/renders/responsibility_step0009.png"

run_evaluation b0 puri_gs_b0_default.yaml
run_evaluation b1 puri_gs_b1_absgrad.yaml
run_evaluation a1 puri_gs_a1_responsibility.yaml

echo "Validating checkpoints, splits, metrics, and responsibility map..."
CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH="${ROOT_DIR}" "${PYTHON_BIN}" - \
  "${RESULT_ROOT}" <<'PY' | tee "${RESULT_ROOT}/smoke_summary.stdout"
import json
import math
import sys
from pathlib import Path

import gsplat
import numpy as np
import torch
from PIL import Image


root = Path(sys.argv[1])
summary = {
    "decision": "PASS",
    "steps": 10,
    "gpu": torch.cuda.get_device_name(0),
    "gpu_capability": list(torch.cuda.get_device_capability(0)),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "gsplat": gsplat.__version__,
    "profiles": {},
}


def require_finite(value, path="root"):
    if isinstance(value, dict):
        for key, child in value.items():
            require_finite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            require_finite(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"non-finite value at {path}: {value}")


reference_split = None
for label in ("b0", "b1", "a1"):
    train_dir = root / label
    eval_dir = root / f"{label}_eval"
    checkpoint_path = train_dir / "ckpts" / "ckpt_9_rank0.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["step"] != 9:
        raise RuntimeError(f"unexpected checkpoint step for {label}")
    for name, tensor in checkpoint["splats"].items():
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"non-finite checkpoint tensor: {label}.{name}")

    split = json.loads((train_dir / "dataset_split.json").read_text())
    if len(split["train"]) != 17 or len(split["test"]) != 3:
        raise RuntimeError(f"unexpected Fern split for {label}")
    split_signature = (split["train"], split["test"])
    if reference_split is None:
        reference_split = split_signature
    elif split_signature != reference_split:
        raise RuntimeError(f"dataset split differs for {label}")

    train_stats = json.loads(
        (train_dir / "stats" / "train_step0009_rank0.json").read_text()
    )
    test_metrics = json.loads(
        (eval_dir / "stats" / "test_step0009.json").read_text()
    )
    require_finite(train_stats, f"{label}.train")
    require_finite(test_metrics, f"{label}.test")
    summary["profiles"][label] = {
        "train": train_stats,
        "test": test_metrics,
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "train_images": len(split["train"]),
        "test_images": len(split["test"]),
    }

responsibility_path = root / "a1" / "renders" / "responsibility_step0009.png"
responsibility = np.asarray(Image.open(responsibility_path))
summary["profiles"]["a1"]["responsibility_map"] = {
    "path": str(responsibility_path),
    "shape": list(responsibility.shape),
    "min": int(responsibility.min()),
    "max": int(responsibility.max()),
    "unique_values": int(np.unique(responsibility).size),
}

output = json.dumps(summary, indent=2)
(root / "smoke_summary.json").write_text(output + "\n")
print(output)
PY

echo "decision=PASS"
echo "summary=${RESULT_ROOT}/smoke_summary.json"
