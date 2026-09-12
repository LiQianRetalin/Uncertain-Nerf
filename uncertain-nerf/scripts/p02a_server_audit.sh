#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 6 ]]; then
  echo "用法: bash scripts/p02a_server_audit.sh REPO_ROOT WORK_ROOT ANDROID_DATA PATIO_DATA EXPECTED_COMMIT PACKAGE_TAG" >&2
  exit 2
fi

repo_root="$(realpath "$1")"
work_root="$(realpath "$2")"
android_data="$(realpath "$3")"
patio_data="$(realpath "$4")"
expected_commit="$5"
package_tag="$6"
code_root="$repo_root/uncertain-nerf"
output="$work_root/p02a"
robust_commit="a130281d6d0c004032a9a57e8d6a14962d9836d3"
spotless_commit="0caae3cc45bb1fddf86bd47e4a521888f5c49889"
robust_source="$work_root/sources/RobustSplat-$robust_commit"
spotless_source="$work_root/sources/SpotLessSplats-$spotless_commit"
robust_env="$work_root/envs/robustsplat"
spotless_env="$work_root/envs/spotless"
eval_env="$code_root/.venv-gsplat153"
android_internal_data="$code_root/data/nerf_robustnerf/robustnerf/android"
patio_internal_data="$code_root/data/nerf_on-the-go/patio_high_v1"
log="$work_root/p02a-launcher.log"
state="$work_root/p02a-state.json"

write_state() {
  local stage="$1" status="$2" detail="$3" percent="$4"
  python3 - "$state" "$stage" "$status" "$detail" "$percent" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "schema": "puri-gs-p02a-live-state-v1", "stage": sys.argv[2],
    "status": sys.argv[3], "detail": sys.argv[4], "percent": int(sys.argv[5]),
    "updated_utc": datetime.now(timezone.utc).isoformat(),
}, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

fail_trap() {
  local code=$?
  write_state "p02a" "FAILED" "查看 $log" 100 || true
  echo "P02A_FAILED_EXIT=$code"
  exit "$code"
}
trap fail_trap ERR

[[ -d "$repo_root/.git" ]] || { echo "不是Git仓库: $repo_root" >&2; exit 3; }
[[ "$(git -C "$repo_root" branch --show-current)" == "ru-part" ]] || { echo "当前分支不是ru-part" >&2; exit 3; }
[[ "$(git -C "$repo_root" rev-parse HEAD)" == "$expected_commit" ]] || { echo "HEAD不等于预期提交" >&2; exit 3; }
[[ "$(git -C "$repo_root" rev-parse origin/ru-part)" == "$expected_commit" ]] || { echo "origin/ru-part未同步" >&2; exit 3; }
[[ -z "$(git -C "$repo_root" status --porcelain)" ]] || { echo "服务器工作树不干净" >&2; exit 3; }
[[ -f "$work_root/report/status.json" ]] || { echo "缺少P02最终报告" >&2; exit 3; }
[[ ! -e "$output" ]] || { echo "拒绝覆盖现有P02-A目录: $output" >&2; exit 4; }
[[ -x "$robust_env/bin/python" && -x "$spotless_env/bin/python" && -x "$eval_env/bin/python" ]] || { echo "缺少既有隔离环境" >&2; exit 3; }
[[ -d "$android_internal_data/images_4" && -d "$android_internal_data/sparse/0" ]] || { echo "缺少P01 Android原评测数据" >&2; exit 3; }
[[ -d "$patio_internal_data/images_4" && -f "$patio_internal_data/puri_gs_protocol.json" ]] || { echo "缺少P01 Patio-High原评测数据" >&2; exit 3; }

mkdir -p "$output"/{runtime,loader,features,checkpoints,representatives,timing,logs,historical/p01,historical/p02,saved_configs}
write_state "preflight" "RUNNING" "只读核对P01/P02与边界" 3
cp -a "$code_root/reports/p01/." "$output/historical/p01/"
cp -a "$work_root/report/." "$output/historical/p02/"
cp -a "$work_root/state/pipeline.json" "$output/historical/p02/pipeline.json"

write_state "runtime_provenance" "RUNNING" "核对运行提交、源码diff、配置和命令" 8
"$eval_env/bin/python" "$code_root/tools/p02a_repository_audit.py" \
  --repo-root "$repo_root" --work-root "$work_root" \
  --robust-source "$robust_source" --spotless-source "$spotless_source" \
  --expected-commit "$expected_commit" --output "$output/runtime/repository_provenance.json"
git -C "$robust_source" status --porcelain=v1 > "$output/runtime/robustsplat.status"
git -C "$robust_source" diff --no-ext-diff > "$output/runtime/robustsplat.diff"
git -C "$robust_source" submodule status --recursive > "$output/runtime/robustsplat.submodules"
git -C "$robust_source/submodules/fused-ssim" diff --no-ext-diff > "$output/runtime/robustsplat-fused-ssim.diff"
git -C "$robust_source/submodules/simple-knn" diff --no-ext-diff > "$output/runtime/robustsplat-simple-knn.diff"
git -C "$spotless_source" status --porcelain=v1 > "$output/runtime/spotless.status"
git -C "$spotless_source" diff --no-ext-diff > "$output/runtime/spotless.diff"
cp "$code_root/scripts/p02_server_pipeline.sh" "$output/runtime/"
cp "$code_root/tools/p02_extract_sls_features.py" "$output/runtime/"
cp "$code_root/tools/p02_evaluate_predictions.py" "$output/runtime/"
cp "$code_root/tools/p02_finalize_report.py" "$output/runtime/"
cp "$code_root/patches/p02_robustsplat_seed_float_timing.patch" "$output/runtime/"
cp "$code_root/patches/p02_robustsplat_pin_dinov2.patch" "$output/runtime/"
cp "$code_root/patches/p02_spotless_seed_float_timing.patch" "$output/runtime/"

python3 - "$output/runtime/runtime_diff_classification.csv" <<'PY'
import csv, sys
rows = [
    ("RobustSplat", "path/seed", "seed CLI and safe_state forwarding", "runtime/robustsplat.diff; runtime/p02_robustsplat_seed_float_timing.patch", "no algorithm change"),
    ("RobustSplat", "loader/split", "keyword split clutter/extra", "runtime/robustsplat.diff; loader/robustsplat-*.json", "identity-relevant and verified"),
    ("RobustSplat", "float export", "clamped float32 NPY by source image name", "runtime/robustsplat.diff", "evaluation adapter only"),
    ("RobustSplat", "timing", "10 warmups and three native render repeats", "runtime/robustsplat.diff", "timing adapter only"),
    ("RobustSplat", "dependency compatibility", "fixed DINO source revision and CUDA build compatibility", "runtime/robustsplat.diff; runtime/robustsplat-*.diff", "no loss/model hyperparameter change"),
    ("SLS-MLP", "path/seed", "explicit seed field preserving prior 42", "runtime/spotless.diff", "no algorithm change"),
    ("SLS-MLP", "loader/split", "keyword split and train-only semantic feature load", "runtime/spotless.diff; loader/sls-mlp-*.json", "identity-relevant and verified"),
    ("SLS-MLP", "float export", "clamped float32 NPY by source image name", "runtime/spotless.diff", "evaluation adapter only"),
    ("SLS-MLP", "timing", "10 warmups and three native rasterize_splats repeats", "runtime/spotless.diff", "timing adapter only"),
    ("SLS-MLP", "algorithm/hyperparameters", "robust+semantics+no-cluster+no-UBP; fixed scene bounds", "runtime/actual_commands.csv; saved_configs/", "actual adopted recipe; Android defaults and Patio benchmark bounds"),
]
with open(sys.argv[1], "w", encoding="utf-8-sig", newline="") as f:
    w=csv.writer(f); w.writerow(["method","category","change","evidence","classification"]); w.writerows(rows)
PY

python3 - "$output/runtime/actual_commands.csv" "$repo_root" "$work_root" "$android_data" "$patio_data" "$robust_source" "$spotless_source" <<'PY'
import csv, sys
out, repo, work, android, patio, robust, spotless = sys.argv[1:]
rows=[]
for scene, data, rbounds in (("android", android, ""), ("patio_high", patio, "--lower_bound 0.3 --upper_bound 0.8")):
    view_r=f"{work}/dataset_views/robustsplat/{scene}"; view_s=f"{work}/dataset_views/spotless/{scene}"
    rid=f"P02-{scene}-robustsplat"; dest=f"{work}/outputs/{rid}"
    rows.append((rid,"training","reconstructed exactly from frozen runner",f"CUDA_VISIBLE_DEVICES=<ledger_gpu> conda run --prefix {work}/envs/robustsplat python {robust}/train.py -s {view_r} -m {dest} --iterations 30000 --seed 42 --resolution 1 --eval --disable_viewer --test_iterations 30000 --save_iterations 30000 --checkpoint_iterations 30000 --quiet"))
    rows.append((rid,"render/evaluate","frozen runner plus independent evaluator",f"P02_FLOAT_PREDICTIONS=1 P02_TIMING_WARMUP=10 P02_TIMING_REPEATS=3 python {robust}/render.py -m {dest} --iteration 30000 --skip_train --quiet"))
    rows.append((rid,"independent_evaluation","reconstructed exactly from frozen runner",f"{repo}/uncertain-nerf/.venv-gsplat153/bin/python {repo}/uncertain-nerf/tools/p02_evaluate_predictions.py --run-id {rid} --method robustsplat --scene {scene} --data-dir {data} --prediction-dir {dest}/test/ours_30000/float_predictions --checkpoint-sha256 <FULL_SHA256_FROM_RUNTIME> --output-dir {work}/report/runs/{rid}"))
    rid=f"P02-{scene}-sls-mlp"; dest=f"{work}/outputs/{rid}"
    bounds="--lower_bound 0.5 --upper_bound 0.9" if scene=="android" else rbounds
    rows.append((rid,"training","reconstructed exactly from frozen runner",f"CUDA_VISIBLE_DEVICES=<ledger_gpu> conda run --prefix {work}/envs/spotless python {spotless}/examples/spotless_trainer.py --data_dir {view_s} --data_factor 1 --result_dir {dest} --loss_type robust --semantics --no-cluster --train_keyword clutter --test_keyword extra --seed 42 --max_steps 30000 --eval_steps 30000 --save_steps 30000 --disable_viewer {bounds}"))
    rows.append((rid,"render/evaluate","frozen runner plus independent evaluator",f"P02_FLOAT_PREDICTIONS=1 P02_TIMING_WARMUP=10 P02_TIMING_REPEATS=3 python {spotless}/examples/spotless_trainer.py --data_dir {view_s} --data_factor 1 --result_dir {dest} --loss_type robust --semantics --no-cluster --train_keyword clutter --test_keyword extra --seed 42 --max_steps 30000 --eval_steps 30000 --save_steps 30000 --disable_viewer --ckpt {dest}/ckpts/ckpt_29999.pt {bounds}"))
    rows.append((rid,"independent_evaluation","reconstructed exactly from frozen runner",f"{repo}/uncertain-nerf/.venv-gsplat153/bin/python {repo}/uncertain-nerf/tools/p02_evaluate_predictions.py --run-id {rid} --method sls-mlp --scene {scene} --data-dir {data} --prediction-dir {dest}/renders/float_step29999 --checkpoint-sha256 <FULL_SHA256_FROM_RUNTIME> --output-dir {work}/report/runs/{rid}"))
with open(out,"w",encoding="utf-8-sig",newline="") as f:
    w=csv.writer(f); w.writerow(["run_id","phase","evidence_status","command"]); w.writerows(rows)
PY

for run_path in "$work_root"/outputs/P02-*; do
  run_name="$(basename "$run_path")"
  find "$run_path" -maxdepth 2 -type f \( -name 'cfg_args' -o -name 'config.yml' -o -name 'config.yaml' -o -name 'config.json' \) -print0 | while IFS= read -r -d '' config; do
    cp "$config" "$output/saved_configs/${run_name}-$(basename "$config")"
  done
done
python3 - "$output/saved_configs/manifest.json" "$output/saved_configs" <<'PY'
import hashlib, json, sys
from pathlib import Path
out=Path(sys.argv[1]); root=Path(sys.argv[2])
files=[]
for path in sorted(root.iterdir()):
    if path.is_file() and path.name != out.name:
        files.append({"name":path.name,"bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
out.write_text(json.dumps({
    "schema":"puri-gs-p02a-saved-config-manifest-v1",
    "files":files,
    "spotless_saved_config_status":"UPSTREAM_RUNNER_DID_NOT_EMIT_A_STANDALONE_CONFIG; complete realized command is frozen in runtime/actual_commands.csv",
},indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
PY

write_state "loaders" "RUNNING" "实例化四套实际loader并逐名核验" 16
"$robust_env/bin/python" "$code_root/tools/p02a_loader_audit.py" --method robustsplat --source "$robust_source" --data-dir "$work_root/dataset_views/robustsplat/android" --output "$output/loader/robustsplat-android.json" > "$output/logs/loader-robustsplat-android.log" 2>&1
"$robust_env/bin/python" "$code_root/tools/p02a_loader_audit.py" --method robustsplat --source "$robust_source" --data-dir "$work_root/dataset_views/robustsplat/patio_high" --output "$output/loader/robustsplat-patio_high.json" > "$output/logs/loader-robustsplat-patio_high.log" 2>&1
"$spotless_env/bin/python" "$code_root/tools/p02a_loader_audit.py" --method sls-mlp --source "$spotless_source" --data-dir "$work_root/dataset_views/spotless/android" --output "$output/loader/sls-mlp-android.json" > "$output/logs/loader-sls-mlp-android.log" 2>&1
"$spotless_env/bin/python" "$code_root/tools/p02a_loader_audit.py" --method sls-mlp --source "$spotless_source" --data-dir "$work_root/dataset_views/spotless/patio_high" --output "$output/loader/sls-mlp-patio_high.json" > "$output/logs/loader-sls-mlp-patio_high.log" 2>&1

write_state "features" "RUNNING" "核验SLS/DINO固定权重与特征身份" 28
"$eval_env/bin/python" "$code_root/tools/p02a_weight_audit.py" --work-root "$work_root" --code-root "$code_root" --robust-source "$robust_source" --spotless-source "$spotless_source" --output "$output/features/weight_provenance.json" > "$output/logs/weight-provenance.log" 2>&1
cp "$work_root/features/android.json" "$output/features/sls-android-original-status.json"
cp "$work_root/features/patio_high.json" "$output/features/sls-patio_high-original-status.json"

write_state "checkpoints" "RUNNING" "只读加载8个既有checkpoint并核对完整哈希" 36
declare -a internal_specs=(
  "P02A-android-b1|$code_root/logs-puri/ru-generalization-rerun-9e292309/android_b1_30k/ckpts/ckpt_29999_rank0.pt"
  "P02A-android-ru|$code_root/logs-puri/ru-generalization-rerun-9e292309/android_ru_30k/ckpts/ckpt_29999_rank0.pt"
  "P02A-patio_high-b1|$code_root/logs-puri/ru-generalization-ontogo-749d584b/patio_high_b1_30k/ckpts/ckpt_29999_rank0.pt"
  "P02A-patio_high-ru|$code_root/logs-puri/ru-generalization-ontogo-749d584b/patio_high_ru_30k/ckpts/ckpt_29999_rank0.pt"
)
for spec in "${internal_specs[@]}"; do
  IFS='|' read -r rid checkpoint <<<"$spec"
  "$eval_env/bin/python" "$code_root/tools/p02a_checkpoint_audit.py" --checkpoint "$checkpoint" --family internal --run-id "$rid" --output "$output/checkpoints/$rid.json" >> "$output/logs/checkpoints.log" 2>&1
done
for scene in android patio_high; do
  rid="P02A-$scene-robustsplat"; checkpoint="$work_root/outputs/P02-$scene-robustsplat/chkpnt30000.pth"
  "$robust_env/bin/python" "$code_root/tools/p02a_checkpoint_audit.py" --checkpoint "$checkpoint" --family robustsplat --run-id "$rid" --output "$output/checkpoints/$rid.json" >> "$output/logs/checkpoints.log" 2>&1
  rid="P02A-$scene-sls-mlp"; checkpoint="$work_root/outputs/P02-$scene-sls-mlp/ckpts/ckpt_29999.pt"
  "$spotless_env/bin/python" "$code_root/tools/p02a_checkpoint_audit.py" --checkpoint "$checkpoint" --family sls-mlp --run-id "$rid" --output "$output/checkpoints/$rid.json" >> "$output/logs/checkpoints.log" 2>&1
done

write_state "representatives" "RUNNING" "导出共同首中末与Patio固定诊断图" 44
"$eval_env/bin/python" "$code_root/tools/p02a_export_representatives.py" --repo-root "$repo_root" --work-root "$work_root" --android-data "$android_data" --patio-data "$patio_data" --output-dir "$output/representatives" > "$output/logs/representatives.log" 2>&1

write_state "gpu_selection" "RUNNING" "按6→7→0→1→2→3→4→5选择单一空闲L20" 50
gpu="$($eval_env/bin/python "$code_root/tools/p02_select_gpu.py" --output "$output/runtime/gpu_selection.json")"
if [[ -n "${P02A_GPU_OVERRIDE:-}" ]]; then
  gpu="$($eval_env/bin/python - "$output/runtime/gpu_selection.json" "$P02A_GPU_OVERRIDE" <<'PY'
import json, sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    requested = int(sys.argv[2])
except ValueError as exc:
    raise SystemExit("P02A_GPU_OVERRIDE must be an integer from 0 through 7") from exc
if requested not in range(8):
    raise SystemExit("P02A_GPU_OVERRIDE must be an integer from 0 through 7")
data = json.loads(path.read_text(encoding="utf-8"))
selected = next((row for row in data["inventory"] if row["physical_index"] == requested), None)
if selected is None or not selected["idle"]:
    raise SystemExit(f"requested P02-A GPU is not idle: {selected}")
data["default_priority_selected"] = data["selected"]
data["selected"] = selected
data["selection_policy"] = "explicit_user_override_for_this_p02a_run"
data["requested_physical_gpu"] = requested
path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(requested)
PY
)"
fi
[[ "$gpu" =~ ^[0-7]$ ]] || { echo "没有空闲GPU: $gpu" >&2; exit 5; }
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv,noheader > "$output/runtime/gpu-before-timing.csv"

retime_internal() {
  local scene="$1" method="$2" config="$3" checkpoint="$4" data="$5" dataset_format="$6"
  local rid="P02A-$scene-$method"
  for repeat in 1 2 3; do
    local result="$output/timing/$rid/repeat-$repeat"
    write_state "timing_${scene}_${method}_${repeat}" "RUNNING" "既有checkpoint同设备计时；无参数更新" $((50 + repeat))
    "$eval_env/bin/python" "$code_root/tools/p02a_gpu_guard.py" --physical-gpu "$gpu" --label "$rid-repeat-$repeat" --append-jsonl "$output/runtime/gpu-guard.jsonl" >> "$output/logs/gpu-guard.log" 2>&1
    "$eval_env/bin/python" "$code_root/run_puri_gs.py" \
      --config "$code_root/configs/$config" --gsplat-dir "$code_root/external/gsplat-v1.5.3-ru" \
      --data-dir "$data" --result-dir "$result" --gpu "$gpu" --data-factor 4 \
      --dataset-format "$dataset_format" \
      --checkpoint "$checkpoint" --eval-warmup-renders 10 --eval-disable-image-save \
      --train-keyword clutter --test-keyword extra > "$output/logs/timing-$scene-$method-repeat-$repeat.log" 2>&1
  done
}

write_state "timing_internal" "RUNNING" "B1/RU四模型各3遍" 54
retime_internal android b1 puri_gs_b1_full30k.yaml "$code_root/logs-puri/ru-generalization-rerun-9e292309/android_b1_30k/ckpts/ckpt_29999_rank0.pt" "$android_internal_data" colmap
retime_internal android ru puri_gs_ru_full30k.yaml "$code_root/logs-puri/ru-generalization-rerun-9e292309/android_ru_30k/ckpts/ckpt_29999_rank0.pt" "$android_internal_data" colmap
retime_internal patio_high b1 puri_gs_b1_full30k.yaml "$code_root/logs-puri/ru-generalization-ontogo-749d584b/patio_high_b1_30k/ckpts/ckpt_29999_rank0.pt" "$patio_internal_data" ontogo-patio-high
retime_internal patio_high ru puri_gs_ru_full30k.yaml "$code_root/logs-puri/ru-generalization-ontogo-749d584b/patio_high_ru_30k/ckpts/ckpt_29999_rank0.pt" "$patio_internal_data" ontogo-patio-high

write_state "timing_external" "RUNNING" "RobustSplat/SLS四模型各3遍" 72
for scene in android patio_high; do
  source_model="$work_root/outputs/P02-$scene-robustsplat"
  audit_model="$output/timing/P02A-$scene-robustsplat"
  mkdir -p "$audit_model"
  cp "$source_model/cfg_args" "$audit_model/cfg_args"
  ln -s "$source_model/point_cloud" "$audit_model/point_cloud"
  "$eval_env/bin/python" "$code_root/tools/p02a_gpu_guard.py" --physical-gpu "$gpu" --label "P02A-$scene-robustsplat" --append-jsonl "$output/runtime/gpu-guard.jsonl" >> "$output/logs/gpu-guard.log" 2>&1
  env CUDA_VISIBLE_DEVICES="$gpu" P02_FLOAT_PREDICTIONS=0 P02_TIMING_WARMUP=10 P02_TIMING_REPEATS=3 \
    conda run --prefix "$robust_env" python "$robust_source/render.py" -m "$audit_model" --iteration 30000 --skip_train --quiet > "$output/logs/timing-$scene-robustsplat.log" 2>&1

  lower="0.5"; upper="0.9"
  if [[ "$scene" == "patio_high" ]]; then
    lower="0.3"; upper="0.8"
  fi
  "$eval_env/bin/python" "$code_root/tools/p02a_gpu_guard.py" --physical-gpu "$gpu" --label "P02A-$scene-sls-mlp" --append-jsonl "$output/runtime/gpu-guard.jsonl" >> "$output/logs/gpu-guard.log" 2>&1
  env CUDA_VISIBLE_DEVICES="$gpu" P02_FLOAT_PREDICTIONS=0 P02_TIMING_WARMUP=10 P02_TIMING_REPEATS=3 \
    conda run --prefix "$spotless_env" python "$code_root/tools/p02a_sls_eval_only.py" \
      --source "$spotless_source" --data-dir "$work_root/dataset_views/spotless/$scene" \
      --checkpoint "$work_root/outputs/P02-$scene-sls-mlp/ckpts/ckpt_29999.pt" \
      --result-dir "$output/timing/P02A-$scene-sls-mlp" --lower-bound "$lower" --upper-bound "$upper" > "$output/logs/timing-$scene-sls-mlp.log" 2>&1
done
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv,noheader > "$output/runtime/gpu-after-timing.csv"

write_state "finalize" "RUNNING" "生成8行/256逐图/24计时及验收报告" 91
"$eval_env/bin/python" "$code_root/tools/p02a_finalize.py" --repo-root "$repo_root" --work-root "$work_root" --output-dir "$output" --expected-runtime-commit "$expected_commit" > "$output/logs/finalize.log" 2>&1

write_state "package" "RUNNING" "生成小型P02-A证据包" 97
archive="$work_root/P02A-final-$package_tag.tar.gz"
[[ ! -e "$archive" ]] || { echo "拒绝覆盖归档: $archive" >&2; exit 4; }
tar -czf "$archive" \
  --exclude='p02a/timing/*/renders' \
  --exclude='p02a/timing/*/test/*/renders' \
  --exclude='p02a/timing/*/test/*/gt' \
  --exclude='p02a/timing/*/point_cloud' \
  -C "$work_root" p02a
sha256sum "$archive" > "$archive.sha256"
write_state "p02a" "COMPLETE" "P02-A验收补全完成并停止；未启动下一包" 100
echo "P02A_FINAL_ARCHIVE=$archive"
echo "P02A_FINAL_ARCHIVE_SHA256=$(cut -d' ' -f1 "$archive.sha256")"
echo "P02A_COMPLETE_STOP"
