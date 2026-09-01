#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 7 ]]; then
  echo "Usage: bash scripts/run_puri_gs_room_efficiency_audit.sh GSPLAT_DIR DATA_DIR B1_CHECKPOINT RU_CHECKPOINT OUTPUT_ROOT REPORT_DIR GPU_ID"
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PURI_GSPLAT_PYTHON:-${ROOT_DIR}/.venv-gsplat153/bin/python}"
GSPLAT_DIR="$1"
DATA_DIR="$2"
B1_CHECKPOINT="$3"
RU_CHECKPOINT="$4"
OUTPUT_ROOT="$5"
REPORT_DIR="$6"
GPU_ID="$7"
WARMUP_RENDERS=10

for path in "${PYTHON_BIN}" "${B1_CHECKPOINT}" "${RU_CHECKPOINT}"; do
  if [[ ! -f "${path}" ]]; then
    echo "AUDIT-REQUIRED-FILE-MISSING: ${path}"
    exit 3
  fi
done
if [[ ! -d "${GSPLAT_DIR}" || ! -d "${DATA_DIR}" ]]; then
  echo "AUDIT-REQUIRED-DIRECTORY-MISSING"
  exit 3
fi
if [[ -e "${OUTPUT_ROOT}" || -e "${REPORT_DIR}" ]]; then
  echo "AUDIT-OUTPUT-ALREADY-EXISTS"
  exit 4
fi

mkdir -p "${OUTPUT_ROOT}/console"

run_evaluation() {
  local method="$1"
  local run_index="$2"
  local config="$3"
  local checkpoint="$4"
  local name="${method}_run${run_index}"
  local result_dir="${OUTPUT_ROOT}/${name}"
  local log_path="${OUTPUT_ROOT}/console/${name}.log"

  echo "AUDIT-EVALUATION-START method=${method} run=${run_index}"
  set +e
  "${PYTHON_BIN}" "${ROOT_DIR}/run_puri_gs.py" \
    --config "${ROOT_DIR}/configs/${config}" \
    --gsplat-dir "${GSPLAT_DIR}" \
    --data-dir "${DATA_DIR}" \
    --result-dir "${result_dir}" \
    --gpu "${GPU_ID}" \
    --data-factor 4 \
    --checkpoint "${checkpoint}" \
    --eval-warmup-renders "${WARMUP_RENDERS}" \
    2>&1 | tee "${log_path}"
  local evaluation_code=${PIPESTATUS[0]}
  set -e
  if [[ "${evaluation_code}" -ne 0 ]]; then
    echo "AUDIT-EVALUATION-FAIL method=${method} run=${run_index} exit=${evaluation_code}"
    return "${evaluation_code}"
  fi

  "${PYTHON_BIN}" - "${result_dir}" "${method}" "${run_index}" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

result = Path(sys.argv[1])
method = sys.argv[2]
run_index = int(sys.argv[3])

def read_json(name):
    path = result / name
    assert path.is_file() and path.stat().st_size > 0, path
    return json.loads(path.read_text(encoding="utf-8"))

split = read_json("dataset_split.json")
efficiency = read_json("efficiency_metrics.json")
validation = read_json("ru_validation.json")
metrics = read_json("test_metrics.json")

assert split["protocol"] == "every-nth-test"
assert len(split["train"]) == 272
assert len(split["test"]) == 39
assert efficiency["warmup_render_count"] == 10
assert efficiency["raw_latency_sample_count"] == 39
assert validation["evaluation_warmup_render_count"] == 10
assert validation["raw_latency_sample_count"] == 39
assert validation["standard_checkpoint_load_pass"] is True
assert validation["evaluation_imported_dino"] is False
assert validation["evaluation_loaded_mask_head"] is False
assert validation["evaluation_rasterization_count_ratio"] == 1.0

latency_path = result / "per_image_latency.csv"
with latency_path.open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
assert len(rows) == 39
assert [int(row["image_index"]) for row in rows] == list(range(39))
assert {row["image_name"] for row in rows} == set(split["test"])
assert all(
    math.isfinite(float(row["latency_ms"]))
    and float(row["latency_ms"]) > 0
    for row in rows
)

renders = list((result / "renders").glob("test_step29999_*.png"))
assert len(renders) == 39

print(json.dumps({
    "status": "ROOM_EFFICIENCY_AUDIT_RUN_VALIDATED",
    "method": method,
    "run_index": run_index,
    "gaussian_count": int(efficiency["gaussian_count"]),
    "psnr": float(metrics["psnr"]),
    "ssim": float(metrics["ssim"]),
    "lpips": float(metrics["lpips"]),
    "render_fps": float(efficiency["render_fps"]),
    "latency_p50_ms": float(efficiency["latency_p50_ms"]),
    "latency_p95_ms": float(efficiency["latency_p95_ms"]),
}, indent=2))
PY
  echo "AUDIT-EVALUATION-PASS method=${method} run=${run_index}"
}

# Counterbalanced order: each method starts first in at least one pair.
run_evaluation b1 1 puri_gs_b1_full30k.yaml "${B1_CHECKPOINT}"
run_evaluation ru 1 puri_gs_ru_full30k.yaml "${RU_CHECKPOINT}"
run_evaluation ru 2 puri_gs_ru_full30k.yaml "${RU_CHECKPOINT}"
run_evaluation b1 2 puri_gs_b1_full30k.yaml "${B1_CHECKPOINT}"
run_evaluation b1 3 puri_gs_b1_full30k.yaml "${B1_CHECKPOINT}"
run_evaluation ru 3 puri_gs_ru_full30k.yaml "${RU_CHECKPOINT}"

"${PYTHON_BIN}" "${ROOT_DIR}/tools/summarize_puri_gs_ru_efficiency_audit.py" \
  --b1-run1 "${OUTPUT_ROOT}/b1_run1" \
  --b1-run2 "${OUTPUT_ROOT}/b1_run2" \
  --b1-run3 "${OUTPUT_ROOT}/b1_run3" \
  --ru-run1 "${OUTPUT_ROOT}/ru_run1" \
  --ru-run2 "${OUTPUT_ROOT}/ru_run2" \
  --ru-run3 "${OUTPUT_ROOT}/ru_run3" \
  --output-dir "${REPORT_DIR}"

test -s "${REPORT_DIR}/PHASE_R_PURI_GS_RU_EFFICIENCY_AUDIT.md"
test -s "${REPORT_DIR}/phase_r_puri_gs_ru_efficiency_audit.json"

echo "AUDIT-REPORT-HASHES"
sha256sum \
  "${REPORT_DIR}/PHASE_R_PURI_GS_RU_EFFICIENCY_AUDIT.md" \
  "${REPORT_DIR}/phase_r_puri_gs_ru_efficiency_audit.json"
echo "ROOM-EFFICIENCY-AUDIT-ALL-RUNS-PASS"
