#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "Usage: bash scripts/run_puri_gs_android_short_screen.sh DATA_DIR RESULT_ROOT GPU_ID"
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PURI_GSPLAT_PYTHON:-${ROOT_DIR}/.venv-gsplat153/bin/python}"
GSPLAT_DIR="${PURI_GSPLAT_DIR:-${ROOT_DIR}/external/gsplat-v1.5.3}"
MAX_STEPS=10000
FINAL_STEP=9999
TRAIN_KEYWORD="clutter"
TEST_KEYWORD="extra"
GPU_ID="$3"

if [[ ! -d "$1" ]]; then
  echo "Dataset directory is missing: $1"
  exit 3
fi
DATA_DIR="$(realpath "$1")"
RESULT_ROOT_INPUT="$2"
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
  echo "STOP_DIRTY_WORKTREE: inspect server changes before running the short screen"
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
for image_dir in "${DATA_DIR}/images" "${DATA_DIR}/images_4"; do
  if [[ ! -d "${image_dir}" ]]; then
    echo "Dataset input is missing: ${image_dir}"
    exit 3
  fi
done
for required in \
  "${DATA_DIR}/sparse/0/cameras.bin" \
  "${DATA_DIR}/sparse/0/images.bin" \
  "${DATA_DIR}/sparse/0/points3D.bin"; do
  if [[ ! -s "${required}" ]]; then
    echo "Dataset input is missing: ${required}"
    exit 3
  fi
done

DATA_COUNTS="$("${PYTHON_BIN}" - "${DATA_DIR}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
full_names = [path.name.casefold() for path in (root / "images").iterdir() if path.is_file()]
factor_names = [path.name.casefold() for path in (root / "images_4").iterdir() if path.is_file()]
train_count = sum("clutter" in name for name in factor_names)
test_count = sum("extra" in name for name in factor_names)
print(f"{len(full_names)},{len(factor_names)},{train_count},{test_count}")
PY
)"
IFS=',' read -r FULL_COUNT FACTOR_COUNT TRAIN_COUNT TEST_COUNT <<<"${DATA_COUNTS}"
echo "dataset=android images=${FULL_COUNT} images_4=${FACTOR_COUNT} train=${TRAIN_COUNT} test=${TEST_COUNT}"
if [[ "${FULL_COUNT}" -ne 263 || "${FACTOR_COUNT}" -ne 263 ]]; then
  echo "STOP_DATA_COUNT: expected 263 files in both images and images_4"
  exit 3
fi
if [[ "${TRAIN_COUNT}" -ne 122 || "${TEST_COUNT}" -ne 19 ]]; then
  echo "STOP_DATA_SPLIT: expected clutter train=122 and extra test=19"
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
  echo "STOP_GPU_BUSY: select an idle L20 before the short screen"
  exit 3
fi

mkdir -p "${RESULT_ROOT}"
git -C "${ROOT_DIR}" rev-parse HEAD >"${RESULT_ROOT}/git_commit.txt"
printf '%s\n' "${GPU_ID}" >"${RESULT_ROOT}/physical_gpu.txt"
printf '%s\n' "${DATA_DIR}" >"${RESULT_ROOT}/dataset_path.txt"

echo "Preparing the pinned external gsplat source without overwriting conflicts..."
bash "${ROOT_DIR}/scripts/prepare_gsplat_robot_baseline.sh" "${GSPLAT_DIR}" \
  2>&1 | tee "${RESULT_ROOT}/gsplat_patch.log"

echo "Verifying the fixed environment and real CUDA forward/backward..."
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
GSPLAT_DIR_REAL="$(realpath "${GSPLAT_DIR}")"
if [[ "${GSPLAT_RUNTIME_PATH}" == "${GSPLAT_DIR_REAL}"/* ]]; then
  echo "STOP_GSPLAT_SHADOWED: trainer would import source instead of the fixed wheel"
  exit 3
fi

run_training() {
  local label="$1"
  local config="$2"
  local result_dir="${RESULT_ROOT}/${label}"
  echo "Running ${label} ${MAX_STEPS}-step training..."
  PYTHONUNBUFFERED=1 "${PYTHON_BIN}" "${ROOT_DIR}/run_puri_gs.py" \
    --config "${ROOT_DIR}/configs/${config}" \
    --gsplat-dir "${GSPLAT_DIR}" \
    --data-dir "${DATA_DIR}" \
    --result-dir "${result_dir}" \
    --gpu "${GPU_ID}" \
    --max-steps "${MAX_STEPS}" \
    --data-factor 4 \
    --train-keyword "${TRAIN_KEYWORD}" \
    --test-keyword "${TEST_KEYWORD}" \
    2>&1 | tee "${RESULT_ROOT}/${label}.train.log"
  test -s "${result_dir}/ckpts/ckpt_${FINAL_STEP}_rank0.pt"
  test -s "${result_dir}/stats/train_step${FINAL_STEP}_rank0.json"
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
    --train-keyword "${TRAIN_KEYWORD}" \
    --test-keyword "${TEST_KEYWORD}" \
    --checkpoint "${train_dir}/ckpts/ckpt_${FINAL_STEP}_rank0.pt" \
    2>&1 | tee "${RESULT_ROOT}/${label}.eval.log"
  test -s "${eval_dir}/stats/test_step${FINAL_STEP}.json"
  test -s "${eval_dir}/dataset_split.json"
  test "$(find "${eval_dir}/renders" -maxdepth 1 -type f -name "test_step${FINAL_STEP}_*.png" | wc -l)" -eq 19
}

run_training b0 puri_gs_b0_default.yaml
run_training b1 puri_gs_b1_absgrad.yaml
run_training a1 puri_gs_a1_responsibility.yaml
test -s "${RESULT_ROOT}/a1/renders/responsibility_step${FINAL_STEP}.png"

run_evaluation b0 puri_gs_b0_default.yaml
run_evaluation b1 puri_gs_b1_absgrad.yaml
run_evaluation a1 puri_gs_a1_responsibility.yaml

echo "Validating checkpoints, splits, metrics, renders, and the responsibility map..."
CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH="${ROOT_DIR}" "${PYTHON_BIN}" - \
  "${RESULT_ROOT}" "${DATA_DIR}" <<'PY' | tee "${RESULT_ROOT}/short_screen_summary.stdout"
import json
import math
import sys
from pathlib import Path

import gsplat
import numpy as np
import torch
from PIL import Image


root = Path(sys.argv[1])
data_dir = Path(sys.argv[2])
summary = {
    "decision": "PASS_EXECUTION",
    "algorithm_decision": "PENDING_VISUAL_AND_GATE_REVIEW",
    "scene": "android",
    "steps": 10000,
    "seed": 42,
    "data_dir": str(data_dir),
    "split_protocol": {"train_keyword": "clutter", "test_keyword": "extra"},
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
    checkpoint_path = train_dir / "ckpts" / "ckpt_9999_rank0.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["step"] != 9999:
        raise RuntimeError(f"unexpected checkpoint step for {label}")
    for name, tensor in checkpoint["splats"].items():
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"non-finite checkpoint tensor: {label}.{name}")

    split = json.loads((train_dir / "dataset_split.json").read_text())
    eval_split = json.loads((eval_dir / "dataset_split.json").read_text())
    if len(split["train"]) != 122 or len(split["test"]) != 19:
        raise RuntimeError(f"unexpected Android split for {label}")
    split_signature = (split["train"], split["test"])
    if split_signature != (eval_split["train"], eval_split["test"]):
        raise RuntimeError(f"train/eval split differs for {label}")
    if reference_split is None:
        reference_split = split_signature
    elif split_signature != reference_split:
        raise RuntimeError(f"dataset split differs across profiles: {label}")

    train_metrics = json.loads(
        (train_dir / "stats" / "train_step9999_rank0.json").read_text()
    )
    test_metrics = json.loads(
        (eval_dir / "stats" / "test_step9999.json").read_text()
    )
    require_finite(train_metrics, f"{label}.train")
    require_finite(test_metrics, f"{label}.test")
    render_count = len(list((eval_dir / "renders").glob("test_step9999_*.png")))
    if render_count != 19:
        raise RuntimeError(f"expected 19 test renders for {label}, got {render_count}")

    efficiency = {
        "train_seconds": train_metrics["ellipse_time"],
        "peak_vram_gib": train_metrics["mem"],
        "num_GS": test_metrics["num_GS"],
        "render_seconds_per_image_mean": test_metrics["ellipse_time"],
        "render_fps_mean": 1.0 / test_metrics["ellipse_time"],
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "latency_p50_ms": None,
        "latency_p95_ms": None,
        "note": "The short-screen trainer reports mean latency only; p50/p95 remain for the formal efficiency gate.",
    }
    (train_dir / "train_metrics.json").write_text(
        json.dumps(train_metrics, indent=2) + "\n"
    )
    (train_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2) + "\n"
    )
    (train_dir / "efficiency_metrics.json").write_text(
        json.dumps(efficiency, indent=2) + "\n"
    )
    summary["profiles"][label] = {
        "train": train_metrics,
        "test": test_metrics,
        "efficiency": efficiency,
        "checkpoint": str(checkpoint_path),
        "train_images": len(split["train"]),
        "test_images": len(split["test"]),
        "test_render_count": render_count,
    }

responsibility_path = root / "a1" / "renders" / "responsibility_step9999.png"
responsibility = np.asarray(Image.open(responsibility_path))
summary["profiles"]["a1"]["responsibility_map"] = {
    "path": str(responsibility_path),
    "shape": list(responsibility.shape),
    "min": int(responsibility.min()),
    "max": int(responsibility.max()),
    "mean": float(responsibility.mean()),
    "unique_values": int(np.unique(responsibility).size),
}

b1 = summary["profiles"]["b1"]
a1 = summary["profiles"]["a1"]
summary["a1_vs_b1"] = {
    "psnr_delta_db": a1["test"]["psnr"] - b1["test"]["psnr"],
    "ssim_delta": a1["test"]["ssim"] - b1["test"]["ssim"],
    "lpips_relative_improvement": (
        b1["test"]["lpips"] - a1["test"]["lpips"]
    ) / b1["test"]["lpips"],
    "gaussian_count_ratio": a1["test"]["num_GS"] / b1["test"]["num_GS"],
    "peak_vram_ratio": a1["train"]["mem"] / b1["train"]["mem"],
    "train_time_ratio": a1["train"]["ellipse_time"] / b1["train"]["ellipse_time"],
    "render_fps_ratio": (
        b1["test"]["ellipse_time"] / a1["test"]["ellipse_time"]
    ),
}
require_finite(summary)
output = json.dumps(summary, indent=2)
(root / "short_screen_summary.json").write_text(output + "\n")
print(output)
PY

echo "decision=PASS_EXECUTION"
echo "algorithm_decision=PENDING_VISUAL_AND_GATE_REVIEW"
echo "summary=${RESULT_ROOT}/short_screen_summary.json"
