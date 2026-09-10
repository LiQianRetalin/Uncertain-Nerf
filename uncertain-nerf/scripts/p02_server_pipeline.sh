#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
用法:
  bash scripts/p02_server_pipeline.sh all REPO_ROOT WORK_ROOT ANDROID_DATA PATIO_DATA EXPECTED_COMMIT

说明:
  REPO_ROOT      ru-part 仓库根目录（其下应有 uncertain-nerf/）
  WORK_ROOT      P02 独立工作目录，不能位于 REPO_ROOT 内
  ANDROID_DATA   Android common factor4 数据目录
  PATIO_DATA     Patio-High common 数据目录
  EXPECTED_COMMIT 必须为 P01 已推送提交

流水线严格顺序：预检 -> 两套隔离环境 -> SLS 训练集特征 -> 4 项各 100 步 smoke
-> 4 项 seed42/30k 正式训练 -> 独立评测 -> 原生渲染计时。任一 smoke 失败时不会开正式训练。
EOF
}

if [[ $# -ne 6 || "$1" != "all" ]]; then
  usage
  exit 2
fi

mode="$1"
repo_root="$(realpath "$2")"
work_root="$(realpath -m "$3")"
android_data="$(realpath "$4")"
patio_data="$(realpath "$5")"
expected_commit="$6"
code_root="$repo_root/uncertain-nerf"

robust_commit="a130281d6d0c004032a9a57e8d6a14962d9836d3"
spotless_commit="0caae3cc45bb1fddf86bd47e4a521888f5c49889"
robust_src="$work_root/sources/RobustSplat-$robust_commit"
spotless_src="$work_root/sources/SpotLessSplats-$spotless_commit"
robust_env="$work_root/envs/robustsplat"
spotless_env="$work_root/envs/spotless"
eval_env="$repo_root/uncertain-nerf/.venv-gsplat153"
state_dir="$work_root/state"
log_dir="$work_root/logs"
output_dir="$work_root/outputs"
smoke_dir="$work_root/smoke"
feature_root="$work_root/features"
report_dir="$work_root/report"

mkdir -p "$state_dir" "$log_dir" "$output_dir" "$smoke_dir" "$feature_root" "$report_dir"
if [[ ! -f "$state_dir/stage_timing.csv" ]]; then
  printf 'stage,started_epoch,ended_epoch,exit_code\n' > "$state_dir/stage_timing.csv"
fi
if [[ ! -f "$state_dir/smoke_ledger.csv" ]]; then
  printf 'run_id,status,updates,gpu\n' > "$state_dir/smoke_ledger.csv"
fi
if [[ ! -f "$state_dir/formal_ledger.csv" ]]; then
  printf 'run_id,status,seed,updates,gpu,started_epoch,ended_epoch\n' > "$state_dir/formal_ledger.csv"
fi

json_escape() {
  python3 -c 'import json,sys; print(json.dumps(sys.argv[1], ensure_ascii=False))' "$1"
}

write_state() {
  local stage="$1"
  local status="$2"
  local detail="$3"
  printf '{"stage":%s,"status":%s,"detail":%s,"updated_utc":%s}\n' \
    "$(json_escape "$stage")" "$(json_escape "$status")" "$(json_escape "$detail")" \
    "$(json_escape "$(date -u +%Y-%m-%dT%H:%M:%SZ)")" > "$state_dir/pipeline.json"
}

run_logged() {
  local label="$1"
  shift
  local log_path="$log_dir/$label.log"
  local started_epoch
  local ended_epoch
  local exit_code=0
  started_epoch="$(date +%s)"
  printf '%s\n' "$log_path" > "$state_dir/current_log.txt"
  write_state "$label" "RUNNING" "查看 $log_path"
  "$@" > "$log_path" 2>&1 || exit_code=$?
  ended_epoch="$(date +%s)"
  printf '%s,%s,%s,%s\n' "$label" "$started_epoch" "$ended_epoch" "$exit_code" >> "$state_dir/stage_timing.csv"
  return "$exit_code"
}

fail() {
  local message="$1"
  write_state "preflight" "FAILED" "$message"
  echo "错误: $message" >&2
  exit 1
}

[[ "$mode" == "all" ]] || fail "未知模式"
[[ -d "$repo_root/.git" ]] || fail "REPO_ROOT 不是 Git 仓库根目录"
[[ -d "$code_root" ]] || fail "REPO_ROOT 下没有 uncertain-nerf"
[[ "$work_root" != "$repo_root" && "$work_root" != "$repo_root/"* ]] || fail "WORK_ROOT 必须位于仓库外，避免污染或覆盖历史资产"
[[ -x "$eval_env/bin/python" ]] || fail "缺少独立评测环境 $eval_env/bin/python"
[[ "$(git -C "$repo_root" branch --show-current)" == "ru-part" ]] || fail "当前分支不是 ru-part"
[[ "$(git -C "$repo_root" rev-parse HEAD)" == "$expected_commit" ]] || fail "服务器 HEAD 与 P01 提交不一致"
[[ "$(git -C "$repo_root" rev-parse origin/ru-part)" == "$expected_commit" ]] || fail "服务器 origin/ru-part 与 P01 提交不一致，请先 fetch/pull"
[[ -z "$(git -C "$repo_root" status --porcelain)" ]] || fail "服务器 ru-part 工作区不干净"

write_state "common_data" "RUNNING" "校验共同数据、拆分、哈希与固定源代码约束"
"$eval_env/bin/python" "$code_root/tools/p02_validate_common_inputs.py" \
  --protocol-manifest "$code_root/reports/protocol_manifest.json" \
  --run-plan "$code_root/reports/run_plan_P02.csv" \
  --android-root "$android_data" \
  --patio-root "$patio_data" \
  --execution-scope server \
  --output "$report_dir/preflight_data_only.json"

write_state "sources" "RUNNING" "获取两个固定提交并应用最小适配补丁"
if [[ ! -d "$robust_src/.git" ]]; then
  git clone --recurse-submodules https://github.com/fcyycf/RobustSplat.git "$robust_src"
fi
git -C "$robust_src" fetch --depth 1 origin "$robust_commit"
git -C "$robust_src" checkout --detach "$robust_commit"
git -C "$robust_src" submodule update --init --recursive

if [[ ! -d "$spotless_src/.git" ]]; then
  git clone https://github.com/lilygoli/SpotLessSplats.git "$spotless_src"
fi
git -C "$spotless_src" fetch --depth 1 origin "$spotless_commit"
git -C "$spotless_src" checkout --detach "$spotless_commit"

apply_exact_patch() {
  local source_dir="$1"
  local patch_file="$2"
  if git -C "$source_dir" apply --reverse --check "$patch_file" >/dev/null 2>&1; then
    return
  fi
  git -C "$source_dir" apply --check "$patch_file"
  git -C "$source_dir" apply "$patch_file"
}

apply_exact_patch "$robust_src" "$code_root/patches/p02_robustsplat_seed_float_timing.patch"
apply_exact_patch "$spotless_src" "$code_root/patches/p02_spotless_seed_float_timing.patch"

"$eval_env/bin/python" "$code_root/tools/p02_validate_common_inputs.py" \
  --protocol-manifest "$code_root/reports/protocol_manifest.json" \
  --run-plan "$code_root/reports/run_plan_P02.csv" \
  --android-root "$android_data" \
  --patio-root "$patio_data" \
  --robust-source "$robust_src" \
  --spotless-source "$spotless_src" \
  --execution-scope server \
  --output "$report_dir/preflight.json"
cp "$code_root/reports/p02/protocol_diff.md" "$report_dir/protocol_diff.md"

write_state "environments" "RUNNING" "创建 RobustSplat 与 SpotLessSplats 隔离环境"
if [[ ! -x "$robust_env/bin/python" ]]; then
  conda env create --prefix "$robust_env" --file "$robust_src/environment.yml"
fi
if [[ ! -x "$spotless_env/bin/python" ]]; then
  conda create --yes --prefix "$spotless_env" python=3.10 pip=24.0
  conda run --prefix "$spotless_env" python -m pip install \
    torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
  conda run --prefix "$spotless_env" python -m pip install -r "$spotless_src/examples/requirements.txt"
  conda run --prefix "$spotless_env" python -m pip install --no-build-isolation -e "$spotless_src"
fi

run_logged "verify_robust_env" conda run --prefix "$robust_env" python -c \
  "import torch; import diff_gaussian_rasterization; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda)"
run_logged "verify_spotless_env" conda run --prefix "$spotless_env" python -c \
  "import torch, gsplat, diffusers, transformers; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda)"
run_logged "verify_eval_env" "$eval_env/bin/python" -c \
  "import torch, torchmetrics; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda)"

select_gpu() {
  "$eval_env/bin/python" "$code_root/tools/p02_select_gpu.py" --output "$state_dir/gpu_selection.json"
}

make_dataset_view() {
  local dataset="$1"
  local scene="$2"
  local method="$3"
  local view="$work_root/dataset_views/$method/$scene"
  mkdir -p "$view/sparse/0"
  if [[ ! -e "$view/images" ]]; then
    ln -s "$dataset/images" "$view/images"
  fi
  for name in cameras.bin images.bin points3D.bin; do
    if [[ ! -e "$view/sparse/0/$name" ]]; then
      ln -s "$dataset/sparse/0/$name" "$view/sparse/0/$name"
    fi
  done
  for name in common_input_protocol.json common_input_validation.json; do
    if [[ ! -e "$view/$name" ]]; then
      ln -s "$dataset/$name" "$view/$name"
    fi
  done
  if [[ "$method" == "spotless" ]]; then
    mkdir -p "$view/SD"
  fi
}

make_dataset_view "$android_data" "android" "robustsplat"
make_dataset_view "$patio_data" "patio_high" "robustsplat"
make_dataset_view "$android_data" "android" "spotless"
make_dataset_view "$patio_data" "patio_high" "spotless"

prepare_features() {
  local dataset="$1"
  local scene="$2"
  local view="$work_root/dataset_views/spotless/$scene"
  local gpu
  gpu="$(select_gpu)"
  run_logged "features_$scene" env CUDA_VISIBLE_DEVICES="$gpu" \
    conda run --prefix "$spotless_env" python "$code_root/tools/p02_extract_sls_features.py" \
      --spotless-source "$spotless_src" --data-dir "$view" \
      --feature-dir "$view/SD" --output-status "$feature_root/$scene.json" --seed 42
}

prepare_features "$android_data" "android"
prepare_features "$patio_data" "patio_high"

declare -a run_ids=(
  "P02-android-robustsplat"
  "P02-android-sls-mlp"
  "P02-patio_high-robustsplat"
  "P02-patio_high-sls-mlp"
)

dataset_for() {
  case "$1" in
    P02-android-*) printf '%s\n' "$android_data" ;;
    P02-patio_high-*) printf '%s\n' "$patio_data" ;;
    *) return 2 ;;
  esac
}

method_for() {
  case "$1" in
    *-robustsplat) printf '%s\n' "robustsplat" ;;
    *-sls-mlp) printf '%s\n' "sls-mlp" ;;
    *) return 2 ;;
  esac
}

run_robust_train() {
  local dataset="$1"
  local destination="$2"
  local steps="$3"
  local gpu="$4"
  local robust_dataset
  if [[ "$dataset" == "$patio_data" ]]; then
    robust_dataset="$work_root/dataset_views/robustsplat/patio_high"
  else
    robust_dataset="$work_root/dataset_views/robustsplat/android"
  fi
  env CUDA_VISIBLE_DEVICES="$gpu" conda run --prefix "$robust_env" \
    python "$robust_src/train.py" -s "$robust_dataset" -m "$destination" \
      --iterations "$steps" --seed 42 --resolution 1 --eval --disable_viewer \
      --test_iterations "$steps" --save_iterations "$steps" --checkpoint_iterations "$steps" --quiet
}

run_spotless_train() {
  local dataset="$1"
  local destination="$2"
  local steps="$3"
  local gpu="$4"
  local sls_dataset
  local -a bounds=()
  if [[ "$dataset" == "$patio_data" ]]; then
    bounds=(--lower_bound 0.3 --upper_bound 0.8)
    sls_dataset="$work_root/dataset_views/spotless/patio_high"
  else
    bounds=(--lower_bound 0.5 --upper_bound 0.9)
    sls_dataset="$work_root/dataset_views/spotless/android"
  fi
  env CUDA_VISIBLE_DEVICES="$gpu" conda run --prefix "$spotless_env" \
    python "$spotless_src/examples/spotless_trainer.py" \
      --data_dir "$sls_dataset" --data_factor 1 --result_dir "$destination" \
      --loss_type robust --semantics --no-cluster \
      --train_keyword clutter --test_keyword extra --seed 42 \
      --max_steps "$steps" --eval_steps "$steps" --save_steps "$steps" \
      --disable_viewer "${bounds[@]}"
}

smoke_updates=0
smoke_failed=0
write_state "smoke" "RUNNING" "4 项各 100 更新，总预算 400/800；失败不自动扩展预算"
for run_id in "${run_ids[@]}"; do
  dataset="$(dataset_for "$run_id")"
  method="$(method_for "$run_id")"
  destination="$smoke_dir/$run_id"
  gpu="$(select_gpu)"
  smoke_updates=$((smoke_updates + 100))
  if [[ "$smoke_updates" -gt 800 ]]; then
    fail "smoke 累计更新超过 800 的硬上限"
  fi
  if [[ "$method" == "robustsplat" ]]; then
    if ! run_logged "smoke_$run_id" run_robust_train "$dataset" "$destination" 100 "$gpu"; then
      smoke_failed=1
      printf '%s,FAILED,100,%s\n' "$run_id" "$gpu" >> "$state_dir/smoke_ledger.csv"
      continue
    fi
  else
    if ! run_logged "smoke_$run_id" run_spotless_train "$dataset" "$destination" 100 "$gpu"; then
      smoke_failed=1
      printf '%s,FAILED,100,%s\n' "$run_id" "$gpu" >> "$state_dir/smoke_ledger.csv"
      continue
    fi
  fi
  printf '%s,PASS,100,%s\n' "$run_id" "$gpu" >> "$state_dir/smoke_ledger.csv"
done

if [[ "$smoke_failed" -ne 0 ]]; then
  write_state "smoke" "FAILED" "至少一项 smoke 失败；未启动任何正式训练"
  "$eval_env/bin/python" "$code_root/tools/p02_finalize_report.py" --work-root "$work_root" --status PARTIAL
  exit 1
fi

run_robust_render() {
  local destination="$1"
  local gpu="$2"
  env CUDA_VISIBLE_DEVICES="$gpu" P02_FLOAT_PREDICTIONS=1 P02_TIMING_WARMUP=10 P02_TIMING_REPEATS=3 \
    conda run --prefix "$robust_env" python "$robust_src/render.py" \
      -m "$destination" --iteration 30000 --skip_train --quiet
}

run_spotless_render() {
  local dataset="$1"
  local destination="$2"
  local gpu="$3"
  local sls_dataset
  local -a bounds=()
  if [[ "$dataset" == "$patio_data" ]]; then
    bounds=(--lower_bound 0.3 --upper_bound 0.8)
    sls_dataset="$work_root/dataset_views/spotless/patio_high"
  else
    bounds=(--lower_bound 0.5 --upper_bound 0.9)
    sls_dataset="$work_root/dataset_views/spotless/android"
  fi
  env CUDA_VISIBLE_DEVICES="$gpu" P02_FLOAT_PREDICTIONS=1 P02_TIMING_WARMUP=10 P02_TIMING_REPEATS=3 \
    conda run --prefix "$spotless_env" python "$spotless_src/examples/spotless_trainer.py" \
      --data_dir "$sls_dataset" --data_factor 1 --result_dir "$destination" \
      --loss_type robust --semantics --no-cluster \
      --train_keyword clutter --test_keyword extra --seed 42 \
      --max_steps 30000 --eval_steps 30000 --save_steps 30000 \
      --disable_viewer --ckpt "$destination/ckpts/ckpt_29999.pt" "${bounds[@]}"
}

evaluate_run() {
  local run_id="$1"
  local dataset="$2"
  local destination="$3"
  local method="$4"
  local prediction_dir
  local checkpoint
  local checkpoint_sha256
  local scene
  if [[ "$method" == "robustsplat" ]]; then
    prediction_dir="$destination/test/ours_30000/float_predictions"
    checkpoint="$destination/chkpnt30000.pth"
  else
    prediction_dir="$destination/renders/float_step29999"
    checkpoint="$destination/ckpts/ckpt_29999.pt"
  fi
  [[ -f "$checkpoint" ]] || return 1
  checkpoint_sha256="$(sha256sum "$checkpoint" | awk '{print $1}')"
  if [[ "$run_id" == P02-android-* ]]; then
    scene="android"
  else
    scene="patio_high"
  fi
  "$eval_env/bin/python" "$code_root/tools/p02_evaluate_predictions.py" \
    --run-id "$run_id" --method "$method" --scene "$scene" \
    --data-dir "$dataset" --prediction-dir "$prediction_dir" \
    --checkpoint-sha256 "$checkpoint_sha256" \
    --output-dir "$report_dir/runs/$run_id"
}

formal_failed=0
write_state "formal" "RUNNING" "4 项 seed42/30k 连续正式训练、原生渲染、独立评测和严格计时"
for run_id in "${run_ids[@]}"; do
  dataset="$(dataset_for "$run_id")"
  method="$(method_for "$run_id")"
  destination="$output_dir/$run_id"
  gpu="$(select_gpu)"
  start_epoch="$(date +%s)"
  if [[ "$method" == "robustsplat" ]]; then
    if ! run_logged "train_$run_id" run_robust_train "$dataset" "$destination" 30000 "$gpu"; then
      formal_failed=1
      printf '%s,FAILED_TRAIN,42,30000,%s,%s,\n' "$run_id" "$gpu" "$start_epoch" >> "$state_dir/formal_ledger.csv"
      continue
    fi
    if ! run_logged "render_$run_id" run_robust_render "$destination" "$gpu"; then
      formal_failed=1
      printf '%s,FAILED_RENDER,42,30000,%s,%s,%s\n' "$run_id" "$gpu" "$start_epoch" "$(date +%s)" >> "$state_dir/formal_ledger.csv"
      continue
    fi
  else
    if ! run_logged "train_$run_id" run_spotless_train "$dataset" "$destination" 30000 "$gpu"; then
      formal_failed=1
      printf '%s,FAILED_TRAIN,42,30000,%s,%s,\n' "$run_id" "$gpu" "$start_epoch" >> "$state_dir/formal_ledger.csv"
      continue
    fi
    if ! run_logged "render_$run_id" run_spotless_render "$dataset" "$destination" "$gpu"; then
      formal_failed=1
      printf '%s,FAILED_RENDER,42,30000,%s,%s,%s\n' "$run_id" "$gpu" "$start_epoch" "$(date +%s)" >> "$state_dir/formal_ledger.csv"
      continue
    fi
  fi
  if ! run_logged "evaluate_$run_id" evaluate_run "$run_id" "$dataset" "$destination" "$method"; then
    formal_failed=1
    printf '%s,FAILED_EVAL,42,30000,%s,%s,%s\n' "$run_id" "$gpu" "$start_epoch" "$(date +%s)" >> "$state_dir/formal_ledger.csv"
    continue
  fi
  printf '%s,COMPLETE,42,30000,%s,%s,%s\n' "$run_id" "$gpu" "$start_epoch" "$(date +%s)" >> "$state_dir/formal_ledger.csv"
done

if [[ "$formal_failed" -ne 0 ]]; then
  write_state "formal" "PARTIAL" "至少一项正式运行、渲染或评测失败；其余项目已继续执行；没有自动重训"
  "$eval_env/bin/python" "$code_root/tools/p02_finalize_report.py" --work-root "$work_root" --status PARTIAL
  exit 1
fi

"$eval_env/bin/python" "$code_root/tools/p02_finalize_report.py" --work-root "$work_root" --status COMPLETE
write_state "p02" "COMPLETE" "四项 seed42/30k 训练、独立评测和严格计时均完成；P02 到此停止"
