#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 10 && "$#" -ne 12 ]]; then
  echo "Usage: bash scripts/run_puri_gs_efficiency_audit.sh SCENE GSPLAT_DIR DATA_DIR B1_CHECKPOINT RU_CHECKPOINT OUTPUT_ROOT REPORT_DIR GPU_ID DATA_FACTOR EXPECTED_TEST_IMAGES [TRAIN_KEYWORD TEST_KEYWORD]"
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PURI_GSPLAT_PYTHON:-${ROOT_DIR}/.venv-gsplat153/bin/python}"
SCENE="$1"
GSPLAT_DIR="$2"
DATA_DIR="$3"
B1_CHECKPOINT="$4"
RU_CHECKPOINT="$5"
OUTPUT_ROOT="$6"
REPORT_DIR="$7"
GPU_ID="$8"
DATA_FACTOR="$9"
EXPECTED_TEST_IMAGES="${10}"
TRAIN_KEYWORD=""
TEST_KEYWORD=""
EXPECTED_SPLIT_PROTOCOL="every-nth-test"

if [[ "$#" -eq 12 ]]; then
  TRAIN_KEYWORD="${11}"
  TEST_KEYWORD="${12}"
  EXPECTED_SPLIT_PROTOCOL="filename-keyword"
fi

case "${SCENE}" in
  android)
    REPORT_JSON="${REPORT_DIR}/android_ru_efficiency_audit.json"
    REPORT_MARKDOWN="${REPORT_DIR}/ANDROID_RU_EFFICIENCY_AUDIT.md"
    RAW_CSV="${ROOT_DIR}/analysis/android_ru_latency_raw.csv"
    ;;
  garden)
    REPORT_JSON="${REPORT_DIR}/garden_ru_efficiency_audit.json"
    REPORT_MARKDOWN="${REPORT_DIR}/GARDEN_RU_EFFICIENCY_AUDIT.md"
    RAW_CSV="${ROOT_DIR}/analysis/garden_ru_latency_raw.csv"
    ;;
  ontogo)
    REPORT_JSON="${REPORT_DIR}/ontogo_ru_efficiency_audit.json"
    REPORT_MARKDOWN="${REPORT_DIR}/ONTOGO_RU_EFFICIENCY_AUDIT.md"
    RAW_CSV="${ROOT_DIR}/analysis/ontogo_ru_latency_raw.csv"
    ;;
  *) echo "AUDIT-UNSUPPORTED-SCENE: ${SCENE}"; exit 2 ;;
esac
if [[ ! "${DATA_FACTOR}" =~ ^[1-9][0-9]*$ ]]; then
  echo "AUDIT-INVALID-DATA-FACTOR"
  exit 2
fi
if [[ ! "${EXPECTED_TEST_IMAGES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "AUDIT-INVALID-TEST-COUNT"
  exit 2
fi
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
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "AUDIT-OUTPUT-ALREADY-EXISTS: ${OUTPUT_ROOT}"
  exit 4
fi
for output in "${REPORT_JSON}" "${REPORT_MARKDOWN}" "${RAW_CSV}"; do
  if [[ -e "${output}" ]]; then
    echo "AUDIT-OUTPUT-ALREADY-EXISTS: ${output}"
    exit 4
  fi
done
if [[ "${SCENE}" == "android" ]]; then
  "${PYTHON_BIN}" - "${REPORT_DIR}/android_ru_pairing_audit.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit("ANDROID-EFFICIENCY-BLOCKED: pairing report is missing")
report = json.loads(path.read_text(encoding="utf-8"))
if report.get("decision") != "ANDROID_PAIRING_VALID":
    raise SystemExit("ANDROID-EFFICIENCY-BLOCKED: pairing is not valid")
print("ANDROID-PAIRING-PRECONDITION-PASS")
PY
fi

B1_CHECKPOINT_SHA256="$(sha256sum "${B1_CHECKPOINT}" | awk '{print $1}')"
RU_CHECKPOINT_SHA256="$(sha256sum "${RU_CHECKPOINT}" | awk '{print $1}')"
mkdir -p "${OUTPUT_ROOT}/console"

run_evaluation() {
  local method="$1"
  local run_index="$2"
  local config="$3"
  local checkpoint="$4"
  local checkpoint_sha256="$5"
  local name="${method}_run${run_index}"
  local result_dir="${OUTPUT_ROOT}/${name}"
  local log_path="${OUTPUT_ROOT}/console/${name}.log"
  local args=(
    --config "${ROOT_DIR}/configs/${config}"
    --gsplat-dir "${GSPLAT_DIR}"
    --data-dir "${DATA_DIR}"
    --result-dir "${result_dir}"
    --gpu "${GPU_ID}"
    --data-factor "${DATA_FACTOR}"
    --checkpoint "${checkpoint}"
    --eval-warmup-renders 10
    --eval-disable-image-save
  )
  if [[ -n "${TRAIN_KEYWORD}" ]]; then
    args+=(--train-keyword "${TRAIN_KEYWORD}" --test-keyword "${TEST_KEYWORD}")
  fi

  echo "AUDIT-EVALUATION-START scene=${SCENE} method=${method} run=${run_index}"
  set +e
  "${PYTHON_BIN}" "${ROOT_DIR}/run_puri_gs.py" "${args[@]}" 2>&1 | tee "${log_path}"
  local evaluation_code=${PIPESTATUS[0]}
  set -e
  if [[ "${evaluation_code}" -ne 0 ]]; then
    echo "AUDIT-EVALUATION-FAIL scene=${SCENE} method=${method} run=${run_index} exit=${evaluation_code}"
    return "${evaluation_code}"
  fi
  printf '%s\n' "${checkpoint_sha256}" >"${result_dir}/checkpoint_sha256.txt"

  "${PYTHON_BIN}" - "${result_dir}" "${method}" "${run_index}" "${EXPECTED_TEST_IMAGES}" "${EXPECTED_SPLIT_PROTOCOL}" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

result = Path(sys.argv[1])
method = sys.argv[2]
run_index = int(sys.argv[3])
expected_test_images = int(sys.argv[4])
expected_split_protocol = sys.argv[5]

def read_json(name):
    path = result / name
    assert path.is_file() and path.stat().st_size > 0, path
    return json.loads(path.read_text(encoding="utf-8"))

split = read_json("dataset_split.json")
efficiency = read_json("efficiency_metrics.json")
validation = read_json("ru_validation.json")
metrics = read_json("test_metrics.json")
assert split["protocol"] == expected_split_protocol
assert len(split["test"]) == expected_test_images
assert efficiency["warmup_render_count"] == 10
assert efficiency["raw_latency_sample_count"] == expected_test_images
assert validation["evaluation_warmup_render_count"] == 10
assert validation["raw_latency_sample_count"] == expected_test_images
assert validation["standard_checkpoint_load_pass"] is True
assert validation["evaluation_imported_dino"] is False
assert validation["evaluation_loaded_mask_head"] is False
assert validation["evaluation_rasterization_count_ratio"] == 1.0
assert validation["evaluation_image_save_disabled"] is True

with (result / "per_image_latency.csv").open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
assert len(rows) == expected_test_images
assert [int(row["image_index"]) for row in rows] == list(range(expected_test_images))
assert [row["image_name"] for row in rows] == split["test"]
assert all(math.isfinite(float(row["latency_ms"])) and float(row["latency_ms"]) > 0 for row in rows)
assert len(list((result / "renders").glob("test_step29999_*.png"))) == 0
assert len((result / "checkpoint_sha256.txt").read_text().strip()) == 64
assert all(math.isfinite(float(metrics[key])) for key in ("psnr", "ssim", "lpips"))
print(json.dumps({
    "status": "EFFICIENCY_AUDIT_RUN_VALIDATED",
    "method": method,
    "run_index": run_index,
    "test_images": expected_test_images,
    "gaussian_count": int(efficiency["gaussian_count"]),
    "render_fps": float(efficiency["render_fps"]),
    "latency_p50_ms": float(efficiency["latency_p50_ms"]),
    "latency_p95_ms": float(efficiency["latency_p95_ms"]),
}, indent=2))
PY
  echo "AUDIT-EVALUATION-PASS scene=${SCENE} method=${method} run=${run_index}"
}

# Fixed counterbalanced, non-concurrent order.
run_evaluation b1 1 puri_gs_b1_full30k.yaml "${B1_CHECKPOINT}" "${B1_CHECKPOINT_SHA256}"
run_evaluation ru 1 puri_gs_ru_full30k.yaml "${RU_CHECKPOINT}" "${RU_CHECKPOINT_SHA256}"
run_evaluation ru 2 puri_gs_ru_full30k.yaml "${RU_CHECKPOINT}" "${RU_CHECKPOINT_SHA256}"
run_evaluation b1 2 puri_gs_b1_full30k.yaml "${B1_CHECKPOINT}" "${B1_CHECKPOINT_SHA256}"
run_evaluation b1 3 puri_gs_b1_full30k.yaml "${B1_CHECKPOINT}" "${B1_CHECKPOINT_SHA256}"
run_evaluation ru 3 puri_gs_ru_full30k.yaml "${RU_CHECKPOINT}" "${RU_CHECKPOINT_SHA256}"

summary_args=(
  --b1-run1 "${OUTPUT_ROOT}/b1_run1"
  --b1-run2 "${OUTPUT_ROOT}/b1_run2"
  --b1-run3 "${OUTPUT_ROOT}/b1_run3"
  --ru-run1 "${OUTPUT_ROOT}/ru_run1"
  --ru-run2 "${OUTPUT_ROOT}/ru_run2"
  --ru-run3 "${OUTPUT_ROOT}/ru_run3"
  --output-dir "${REPORT_DIR}"
  --scene "${SCENE}"
  --expected-test-images "${EXPECTED_TEST_IMAGES}"
  --expected-split-protocol "${EXPECTED_SPLIT_PROTOCOL}"
  --require-checkpoint-sha
  --analysis-dir "${ROOT_DIR}/analysis"
)
if [[ "${SCENE}" == "android" ]]; then
  summary_args+=(--expected-train-images 122 --p95-ratio-gate 1.05)
elif [[ "${SCENE}" == "garden" ]]; then
  summary_args+=(--expected-train-images 161)
else
  summary_args+=(--expected-train-images 221)
fi

"${PYTHON_BIN}" "${ROOT_DIR}/tools/summarize_puri_gs_ru_efficiency_audit.py" "${summary_args[@]}"
echo "EFFICIENCY-AUDIT-ALL-RUNS-PASS scene=${SCENE}"
