#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
用法:
  bash scripts/p03_server_pipeline.sh all REPO_ROOT P02_WORK_ROOT P03_WORK_ROOT PREPARED_DATA COMMON_DATA EXPECTED_COMMIT

严格顺序：冻结预检 -> 复用环境/权重 -> Corner RU/SD训练特征 -> 三身份100步smoke
-> 三身份seed42/30k从头训练 -> 同一空闲L20统一计时 -> 浮点质量渲染/独立评测
-> 显存/成本/代表图/报告/小型ZIP。脚本不自动重试，不启动P04/OAC/R。
EOF
}

if [[ $# -ne 7 || "$1" != "all" ]]; then
  usage
  exit 2
fi

repo_root="$(realpath "$2")"
p02_root="$(realpath "$3")"
p03_root="$(realpath -m "$4")"
prepared="$(realpath "$5")"
common="$(realpath "$6")"
expected_commit="$7"
code_root="$repo_root/uncertain-nerf"
eval_env="$code_root/.venv-gsplat153"
eval_python="$eval_env/bin/python"
gsplat_src="$code_root/external/gsplat-v1.5.3-ru"
robust_commit="a130281d6d0c004032a9a57e8d6a14962d9836d3"
spotless_commit="0caae3cc45bb1fddf86bd47e4a521888f5c49889"
robust_src="$p02_root/sources/RobustSplat-$robust_commit"
spotless_src="$p02_root/sources/SpotLessSplats-$spotless_commit"
robust_env="$p02_root/envs/robustsplat"
spotless_env="$p02_root/envs/spotless"
dino_src="$code_root/external/dinov2"
dino_weight="$code_root/data/PURI-GS-assets/dinov2/dinov2_vits14_reg4_pretrain.pth"
state="$p03_root/state"
logs="$p03_root/logs"
report="$p03_root/report"
outputs="$p03_root/outputs"
smoke="$p03_root/smoke"
features="$p03_root/features"
views="$p03_root/dataset_views"
gpu_cap_seconds=43200
LAST_STAGE_EXIT=0

fail_before_root() {
  echo "P03错误: $1" >&2
  exit 1
}

[[ -d "$repo_root/.git" ]] || fail_before_root "REPO_ROOT不是Git仓库"
[[ -d "$code_root" ]] || fail_before_root "缺少uncertain-nerf代码目录"
[[ "$p03_root" != "$repo_root" && "$p03_root" != "$repo_root/"* ]] || fail_before_root "P03_WORK_ROOT必须在仓库外"
[[ ! -e "$p03_root" ]] || fail_before_root "P03_WORK_ROOT已存在；拒绝覆盖或隐式续跑"
mkdir -p "$state" "$logs" "$report" "$outputs" "$smoke" "$features" "$views"

printf 'stage,category,run_id,gpu,started_epoch,ended_epoch,elapsed_seconds,exit_code,timeout_seconds\n' > "$state/gpu_stage_events.csv"
printf 'stage,category,run_id,gpu,epoch,memory_used_mib,utilization_percent\n' > "$state/gpu_memory_samples.csv"
printf 'stage,category,run_id,gpu,started_epoch,ended_epoch,elapsed_seconds,exit_code,timeout_seconds\n' > "$state/stage_events.csv"
printf 'run_id,attempt,status,budget_updates,actual_step,gpu,started_epoch,ended_epoch,exit_code,reason\n' > "$state/smoke_ledger.csv"
printf 'run_id,attempt,status,seed,configured_updates,actual_step,gpu,started_epoch,ended_epoch,exit_code,reason\n' > "$state/formal_ledger.csv"

json_string() {
  python3 -c 'import json,sys; print(json.dumps(sys.argv[1], ensure_ascii=False))' "$1"
}

write_state() {
  local stage_name="$1" status_value="$2" detail="$3"
  printf '{"stage":%s,"status":%s,"detail":%s,"updated_utc":%s,"next_work_package_started":false}\n' \
    "$(json_string "$stage_name")" "$(json_string "$status_value")" "$(json_string "$detail")" \
    "$(json_string "$(date -u +%Y-%m-%dT%H:%M:%SZ)")" > "$state/pipeline.json"
}

gpu_used_seconds() {
  awk -F, 'NR>1 {sum += $7} END {printf "%.0f", sum+0}' "$state/gpu_stage_events.csv"
}

select_gpu() {
  local label="$1"
  "$eval_python" "$code_root/tools/p02_select_gpu.py" --output "$state/gpu_selection_${label}.json"
}

sample_gpu() {
  local label="$1" category="$2" run_id="$3" gpu="$4" stop_file="$5"
  while [[ ! -e "$stop_file" ]]; do
    local row epoch
    row="$(nvidia-smi --id="$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | head -n1)"
    epoch="$(date +%s)"
    printf '%s,%s,%s,%s,%s,%s\n' "$label" "$category" "$run_id" "$gpu" "$epoch" "${row// /}" >> "$state/gpu_memory_samples.csv"
    sleep 2
  done
}

run_cpu_stage() {
  local label="$1" category="$2" run_id="$3" timeout_seconds="$4"
  shift 4
  local start_epoch end_epoch exit_code=0
  start_epoch="$(date +%s)"
  write_state "$label" RUNNING "$logs/$label.log"
  timeout --signal=TERM --kill-after=30s "${timeout_seconds}s" "$@" > "$logs/$label.log" 2>&1 || exit_code=$?
  end_epoch="$(date +%s)"
  printf '%s,%s,%s,,%s,%s,%s,%s,%s\n' "$label" "$category" "$run_id" "$start_epoch" "$end_epoch" "$((end_epoch-start_epoch))" "$exit_code" "$timeout_seconds" >> "$state/stage_events.csv"
  if [[ "$exit_code" -ne 0 ]]; then
    write_state "$label" FAILED "$logs/$label.log"
  fi
  LAST_STAGE_EXIT="$exit_code"
  return "$exit_code"
}

_run_gpu_stage() {
  local label="$1" category="$2" run_id="$3" requested_timeout="$4" gpu="$5" callback="$6"
  shift 6
  local used remaining allowed start_epoch end_epoch exit_code=0 stop_file sampler
  used="$(gpu_used_seconds)"
  remaining=$((gpu_cap_seconds-used))
  [[ "$remaining" -gt 0 ]] || { LAST_STAGE_EXIT=88; write_state budget EXHAUSTED "12 GPU-hour hard cap reached"; return 88; }
  allowed="$requested_timeout"
  if [[ "$allowed" -gt "$remaining" ]]; then allowed="$remaining"; fi
  "$eval_python" "$code_root/tools/p02a_gpu_guard.py" \
    --physical-gpu "$gpu" --label "$label" --append-jsonl "$state/gpu_guards.jsonl" \
    --settle-timeout-seconds 60 > "$logs/${label}_gpu_guard.log" 2>&1 || { LAST_STAGE_EXIT=89; return 89; }
  start_epoch="$(date +%s)"
  stop_file="$state/stop_sampler_${label}"
  sample_gpu "$label" "$category" "$run_id" "$gpu" "$stop_file" &
  sampler=$!
  write_state "$label" RUNNING "GPU $gpu; timeout ${allowed}s; $logs/$label.log"
  "$callback" "$gpu" "$allowed" "$@" > "$logs/$label.log" 2>&1 || exit_code=$?
  touch "$stop_file"
  wait "$sampler" || true
  end_epoch="$(date +%s)"
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' "$label" "$category" "$run_id" "$gpu" "$start_epoch" "$end_epoch" "$((end_epoch-start_epoch))" "$exit_code" "$allowed" >> "$state/gpu_stage_events.csv"
  if [[ "$exit_code" -ne 0 ]]; then
    write_state "$label" FAILED "$logs/$label.log"
  fi
  LAST_STAGE_EXIT="$exit_code"
  return "$exit_code"
}

run_gpu_stage() {
  local label="$1" category="$2" run_id="$3" timeout_seconds="$4"
  shift 4
  local gpu select_exit
  gpu="$(select_gpu "$label")" || {
    select_exit=$?
    LAST_STAGE_EXIT="$select_exit"
    printf 'NONE\n' > "$state/current_gpu.txt"
    return "$select_exit"
  }
  printf '%s\n' "$gpu" > "$state/current_gpu.txt"
  _run_gpu_stage "$label" "$category" "$run_id" "$timeout_seconds" "$gpu" "$@"
}

run_fixed_gpu_stage() {
  local label="$1" category="$2" run_id="$3" timeout_seconds="$4" gpu="$5"
  shift 5
  _run_gpu_stage "$label" "$category" "$run_id" "$timeout_seconds" "$gpu" "$@"
}

partial_stop() {
  local stage_name="$1" detail="$2"
  write_state "$stage_name" PARTIAL "$detail"
  "$eval_python" "$code_root/tools/p03_finalize_report.py" --work-root "$p03_root" --common-dir "$common" --status PARTIAL
  exit 1
}

apply_exact_patch() {
  local source_dir="$1" patch_file="$2"
  if git -C "$source_dir" apply --reverse --check --ignore-whitespace "$patch_file" >/dev/null 2>&1; then
    return 0
  fi
  git -C "$source_dir" apply --check --ignore-whitespace "$patch_file"
  git -C "$source_dir" apply --ignore-whitespace "$patch_file"
}

write_command() {
  local destination="$1"
  shift
  mkdir -p "$destination"
  printf '%q ' "$@" > "$destination/p03_wrapper_command.txt"
  printf '\n' >> "$destination/p03_wrapper_command.txt"
}

stage_cuda_precheck() {
  local gpu="$1" allowed="$2"
  local per_check=$((allowed/3))
  timeout --signal=TERM --kill-after=30s "${per_check}s" env CUDA_VISIBLE_DEVICES="$gpu" \
    "$eval_python" -c "import torch; assert torch.cuda.is_available(); x=torch.ones(1,device='cuda'); assert x.item()==1; print(torch.cuda.get_device_name(0)); print('P03_EVAL_CUDA_FORWARD=PASS')"
  timeout --signal=TERM --kill-after=30s "${per_check}s" env CUDA_VISIBLE_DEVICES="$gpu" \
    conda run --prefix "$robust_env" python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); print('P03_ROBUST_CUDA_FORWARD=PASS')"
  timeout --signal=TERM --kill-after=30s "${per_check}s" env CUDA_VISIBLE_DEVICES="$gpu" \
    conda run --prefix "$spotless_env" python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); print('P03_SPOTLESS_CUDA_FORWARD=PASS')"
}

stage_ru_features() {
  local gpu="$1" allowed="$2"
  timeout --signal=TERM --kill-after=30s "${allowed}s" env CUDA_VISIBLE_DEVICES="$gpu" \
    "$eval_python" "$code_root/tools/cache_puri_gs_features.py" \
      --scene corner --gsplat-dir "$gsplat_src" --data-dir "$prepared" --data-factor 4 \
      --dataset-format ontogo-corner --output-dir "$features/ru" \
      --dino-repo-dir "$dino_src" --dino-weight-path "$dino_weight" --device cuda:0 \
      --train-keyword clutter --test-keyword extra
}

stage_sls_features() {
  local gpu="$1" allowed="$2"
  timeout --signal=TERM --kill-after=30s "${allowed}s" env CUDA_VISIBLE_DEVICES="$gpu" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1 \
    conda run --prefix "$spotless_env" python "$code_root/tools/p02_extract_sls_features.py" \
      --spotless-source "$spotless_src" --data-dir "$views/spotless/corner" \
      --feature-dir "$views/spotless/corner/SD" --output-status "$features/sls_generation.json" --seed 42
}

train_internal() {
  local gpu="$1" allowed="$2" destination="$3" steps="$4" smoke_mode="$5"
  timeout --signal=TERM --kill-after=30s "${allowed}s" env P03_SMOKE_AUDIT="$smoke_mode" \
    "$eval_python" "$code_root/run_puri_gs.py" \
      --config "$code_root/configs/puri_gs_ru_full30k.yaml" --gsplat-dir "$gsplat_src" \
      --data-dir "$prepared" --result-dir "$destination" --gpu "$gpu" --max-steps "$steps" \
      --data-factor 4 --dataset-format ontogo-corner --train-keyword clutter --test-keyword extra \
      --dino-repo-dir "$dino_src" --dino-weight-path "$dino_weight" --feature-cache-dir "$features/ru"
}

train_robust() {
  local gpu="$1" allowed="$2" destination="$3" steps="$4" smoke_mode="$5"
  local -a command=(env CUDA_VISIBLE_DEVICES="$gpu" P03_SMOKE_AUDIT="$smoke_mode" \
    conda run --prefix "$robust_env" python "$robust_src/train.py" \
    -s "$views/robustsplat/corner" -m "$destination" --iterations "$steps" --seed 42 \
    --resolution 1 --eval --disable_viewer --test_iterations "$steps" --save_iterations "$steps" \
    --checkpoint_iterations "$steps" --quiet)
  write_command "$destination" "${command[@]}"
  timeout --signal=TERM --kill-after=30s "${allowed}s" "${command[@]}"
}

train_sls() {
  local gpu="$1" allowed="$2" destination="$3" steps="$4" smoke_mode="$5"
  local -a command=(env CUDA_VISIBLE_DEVICES="$gpu" P03_SMOKE_AUDIT="$smoke_mode" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1 P03_SKIP_TRAJECTORY=1 \
    conda run --prefix "$spotless_env" python "$spotless_src/examples/spotless_trainer.py" \
    --data_dir "$views/spotless/corner" --data_factor 1 --result_dir "$destination" \
    --loss_type robust --semantics --no-cluster --lower_bound 0.5 --upper_bound 0.9 \
    --train_keyword clutter --test_keyword extra --seed 42 --max_steps "$steps" \
    --eval_steps "$steps" --save_steps "$steps" --disable_viewer)
  write_command "$destination" "${command[@]}"
  timeout --signal=TERM --kill-after=30s "${allowed}s" "${command[@]}"
}

render_internal() {
  local gpu="$1" allowed="$2" checkpoint="$3" destination="$4" expected_step="$5" mode="$6"
  timeout --signal=TERM --kill-after=30s "${allowed}s" env CUDA_VISIBLE_DEVICES="$gpu" \
    "$eval_python" "$code_root/tools/p03_internal_eval.py" --gsplat-dir "$gsplat_src" \
      --data-dir "$prepared" --checkpoint "$checkpoint" --output-dir "$destination" \
      --mode "$mode" --warmup 10 --repeats 3 --expected-step "$expected_step"
}

render_robust() {
  local gpu="$1" allowed="$2" destination="$3" iteration="$4" timing="$5" memory="$6"
  timeout --signal=TERM --kill-after=30s "${allowed}s" env CUDA_VISIBLE_DEVICES="$gpu" \
    P02_FLOAT_PREDICTIONS="$([[ "$timing" == 0 ]] && echo 1 || echo 0)" \
    P02_TIMING_WARMUP=10 P02_TIMING_REPEATS="$timing" \
    P03_TIMING_ONLY="$([[ "$timing" == 3 ]] && echo 1 || echo 0)" P03_MEASURE_MEMORY="$memory" \
    conda run --prefix "$robust_env" python "$robust_src/render.py" \
      -m "$destination" --iteration "$iteration" --skip_train --quiet
}

render_sls() {
  local gpu="$1" allowed="$2" destination="$3" step="$4" timing="$5" memory="$6"
  local max_steps=$((step+1))
  timeout --signal=TERM --kill-after=30s "${allowed}s" env CUDA_VISIBLE_DEVICES="$gpu" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1 P03_SKIP_TRAJECTORY=1 \
    P02_FLOAT_PREDICTIONS="$([[ "$timing" == 0 ]] && echo 1 || echo 0)" \
    P02_TIMING_WARMUP=10 P02_TIMING_REPEATS="$timing" \
    P03_TIMING_ONLY="$([[ "$timing" == 3 ]] && echo 1 || echo 0)" P03_MEASURE_MEMORY="$memory" \
    conda run --prefix "$spotless_env" python "$spotless_src/examples/spotless_trainer.py" \
      --data_dir "$views/spotless/corner" --data_factor 1 --result_dir "$destination" \
      --loss_type robust --semantics --no-cluster --lower_bound 0.5 --upper_bound 0.9 \
      --train_keyword clutter --test_keyword extra --seed 42 --max_steps "$max_steps" \
      --eval_steps "$max_steps" --save_steps "$max_steps" --disable_viewer \
      --ckpt "$destination/ckpts/ckpt_${step}.pt"
}

independent_eval() {
  local gpu="$1" allowed="$2" run_id="$3" method="$4" prediction_dir="$5" checkpoint="$6"
  local checkpoint_hash
  checkpoint_hash="$(sha256sum "$checkpoint" | awk '{print $1}')"
  timeout --signal=TERM --kill-after=30s "${allowed}s" env CUDA_VISIBLE_DEVICES="$gpu" \
    "$eval_python" "$code_root/tools/p03_evaluate_predictions.py" \
      --data-dir "$common" --prediction-dir "$prediction_dir" --run-id "$run_id" \
      --method "$method" --checkpoint-sha256 "$checkpoint_hash" --output-dir "$report/runs/$run_id" --device cuda:0
}

[[ "$(git -C "$repo_root" branch --show-current)" == "ru-part" ]] || fail_before_root "服务器分支不是ru-part"
[[ "$(git -C "$repo_root" rev-parse HEAD)" == "$expected_commit" ]] || fail_before_root "服务器HEAD与P03运行提交不一致"
git -C "$repo_root" fetch origin ru-part > "$logs/git_fetch.log" 2>&1
[[ "$(git -C "$repo_root" rev-parse origin/ru-part)" == "$expected_commit" ]] || fail_before_root "origin/ru-part不是P03运行提交"
[[ -z "$(git -C "$repo_root" status --porcelain)" ]] || fail_before_root "服务器主仓库工作区不干净"
[[ -x "$eval_python" && -x "$robust_env/bin/python" && -x "$spotless_env/bin/python" ]] || fail_before_root "P02已验证环境不完整"
[[ "$(git -C "$robust_src" rev-parse HEAD)" == "$robust_commit" ]] || fail_before_root "RobustSplat固定提交不匹配"
[[ "$(git -C "$spotless_src" rev-parse HEAD)" == "$spotless_commit" ]] || fail_before_root "SpotLessSplats固定提交不匹配"
[[ "$(git -C "$dino_src" rev-parse HEAD)" == "7764ea0f912e53c92e82eb78a2a1631e92725fc8" ]] || fail_before_root "内部DINOv2源码提交不匹配"
[[ "$(sha256sum "$dino_weight" | awk '{print $1}')" == "f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb" ]] || fail_before_root "DINOv2权重哈希不匹配"

write_state preflight RUNNING "应用数据/记录适配并核验冻结输入"
bash "$code_root/scripts/prepare_puri_gs_ontogo.sh" "$gsplat_src" > "$logs/prepare_ontogo.log" 2>&1
apply_exact_patch "$gsplat_src" "$code_root/patches/gsplat_v1.5.3_p03_corner.patch"
apply_exact_patch "$gsplat_src" "$code_root/patches/p03_gsplat_smoke_finiteness.patch"
apply_exact_patch "$robust_src" "$code_root/patches/p03_robustsplat_timing_config.patch"
apply_exact_patch "$spotless_src" "$code_root/patches/p03_spotless_timing_memory.patch"
mkdir -p "$report/source_diffs" "$report/environment" "$report/server_preflight"
git -C "$robust_src" diff > "$report/source_diffs/robustsplat_actual.patch"
git -C "$spotless_src" diff > "$report/source_diffs/spotless_actual.patch"
git -C "$gsplat_src" diff > "$report/source_diffs/gsplat_actual.patch"
sha256sum "$code_root"/patches/gsplat_v1.5.3_p03_corner.patch \
  "$code_root"/patches/p03_gsplat_smoke_finiteness.patch \
  "$code_root"/patches/p03_robustsplat_timing_config.patch \
  "$code_root"/patches/p03_spotless_timing_memory.patch > "$report/patch_sha256.txt"

cp "$code_root/reports/p03/protocol_manifest.json" "$report/protocol_manifest.json"
cp "$code_root/reports/p03/input_validation.json" "$report/input_validation.json"
cp "$code_root/reports/p03/run_plan_P03.csv" "$report/run_plan_P03.csv"
cp "$code_root/reports/p03/PREFLIGHT_P03.md" "$report/PREFLIGHT_P03.md"
run_cpu_stage server_input_validation input_validation corner 900 \
  "$eval_python" "$code_root/tools/p03_validate_inputs.py" --prepared-dir "$prepared" --common-dir "$common" \
    --spotless-source "$spotless_src" --output-dir "$report/server_preflight" || partial_stop server_input_validation "转移后的Corner输入核验失败"
cmp "$report/server_preflight/protocol_manifest.json" "$code_root/reports/p03/protocol_manifest.json" || partial_stop server_input_validation "服务器协议与开训前冻结协议不一致"
cp "$common/common_input_protocol.json" "$report/common_input_protocol.json"
cp "$common/common_input_validation.json" "$report/common_input_validation.json"

nvidia-smi -L > "$report/environment/nvidia_smi_L.txt"
nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total --format=csv > "$report/environment/gpu_inventory.csv"
"$eval_python" -m pip freeze > "$report/environment/internal_pip_freeze.txt"
conda run --prefix "$robust_env" python -m pip freeze > "$report/environment/robustsplat_pip_freeze.txt"
conda run --prefix "$spotless_env" python -m pip freeze > "$report/environment/spotless_pip_freeze.txt"
printf 'repo_commit=%s\nrobust_commit=%s\nspotless_commit=%s\ndino_commit=%s\n' \
  "$expected_commit" "$robust_commit" "$spotless_commit" "$(git -C "$dino_src" rev-parse HEAD)" > "$report/environment/source_commits.txt"

if ! run_gpu_stage cuda_precheck preflight corner 600 stage_cuda_precheck; then
  write_state preflight FAILED "CUDA只读前向检查失败"
  exit 1
fi

make_view() {
  local method="$1" view="$views/$method/corner"
  mkdir -p "$view/sparse/0"
  ln -s "$common/images" "$view/images"
  for name in cameras.bin images.bin points3D.bin; do ln -s "$common/sparse/0/$name" "$view/sparse/0/$name"; done
  ln -s "$common/common_input_protocol.json" "$view/common_input_protocol.json"
  ln -s "$common/common_input_validation.json" "$view/common_input_validation.json"
  ln -s "$common/split.json" "$view/split.json"
  if [[ "$method" == "spotless" ]]; then mkdir -p "$view/SD"; fi
}
make_view robustsplat
make_view spotless

if ! run_gpu_stage features_ru features P03-corner-ru 1200 stage_ru_features; then
  write_state features_ru FAILED "RU特征生成失败；P03停止"
  "$eval_python" "$code_root/tools/p03_finalize_report.py" --work-root "$p03_root" --common-dir "$common" --status PARTIAL
  exit 1
fi
if ! run_cpu_stage validate_features_ru feature_validation P03-corner-ru 900 \
  "$eval_python" "$code_root/tools/p03_validate_ru_features.py" --feature-dir "$features/ru" --output "$features/ru_validation.json"
then
  write_state validate_features_ru FAILED "RU特征独立校验失败；P03停止"
  "$eval_python" "$code_root/tools/p03_finalize_report.py" --work-root "$p03_root" --common-dir "$common" --status PARTIAL
  exit 1
fi
if ! run_gpu_stage features_sls features P03-corner-sls-mlp 3600 stage_sls_features; then
  write_state features_sls FAILED "SLS特征生成失败；P03停止"
  "$eval_python" "$code_root/tools/p03_finalize_report.py" --work-root "$p03_root" --common-dir "$common" --status PARTIAL
  exit 1
fi
if ! run_cpu_stage validate_features_sls feature_validation P03-corner-sls-mlp 1200 env \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1 \
  conda run --prefix "$spotless_env" python "$code_root/tools/p02_extract_sls_features.py" \
    --spotless-source "$spotless_src" --data-dir "$views/spotless/corner" \
    --feature-dir "$views/spotless/corner/SD" --output-status "$features/sls_validation.json" --seed 42 --validate-only
then
  write_state validate_features_sls FAILED "SLS特征独立校验失败；P03停止"
  "$eval_python" "$code_root/tools/p03_finalize_report.py" --work-root "$p03_root" --common-dir "$common" --status PARTIAL
  exit 1
fi

run_cpu_stage loader_robust loader_audit P03-corner-robustsplat 900 \
  conda run --prefix "$robust_env" python "$code_root/tools/p02a_loader_audit.py" \
    --method robustsplat --source "$robust_src" --data-dir "$views/robustsplat/corner" --output "$report/loader_robust.json" || partial_stop loader_robust "RobustSplat实际loader核验失败"
run_cpu_stage loader_sls loader_audit P03-corner-sls-mlp 1200 \
  conda run --prefix "$spotless_env" python "$code_root/tools/p02a_loader_audit.py" \
    --method sls-mlp --source "$spotless_src" --data-dir "$views/spotless/corner" --output "$report/loader_sls.json" || partial_stop loader_sls "SLS实际loader或训练特征集合核验失败"
run_cpu_stage loader_combined loader_audit corner 900 \
  "$eval_python" "$code_root/tools/p03_loader_audit.py" --prepared-dir "$prepared" --common-dir "$common" \
    --ru-feature-validation "$features/ru_validation.json" --robust-audit "$report/loader_robust.json" \
    --sls-audit "$report/loader_sls.json" --output "$report/loader_audit.json" || partial_stop loader_combined "三方法loader逐名统一核验失败"
cp "$features/ru/manifest.json" "$report/ru_feature_generation_manifest.json"
cp "$features/ru_validation.json" "$report/ru_feature_validation.json"
cp "$features/sls_generation.json" "$report/sls_feature_generation_manifest.json"
cp "$features/sls_validation.json" "$report/sls_feature_validation.json"

declare -a run_ids=(P03-corner-ru P03-corner-robustsplat P03-corner-sls-mlp)
smoke_failed=0
for run_id in "${run_ids[@]}"; do
  destination="$smoke/$run_id/attempt_1"
  start_epoch="$(date +%s)"
  printf '%s\n' "$run_id" > "$state/current_run_id.txt"
  if [[ "$run_id" == "P03-corner-ru" ]]; then
    callback=train_internal
  elif [[ "$run_id" == "P03-corner-robustsplat" ]]; then
    callback=train_robust
  else
    callback=train_sls
  fi
  if ! run_gpu_stage "smoke_train_$run_id" smoke "$run_id" 600 "$callback" "$destination" 100 1; then
    exit_code="$LAST_STAGE_EXIT"; gpu="$(cat "$state/current_gpu.txt")"; end_epoch="$(date +%s)"
    printf '%s,1,FAILED,100,UNKNOWN,%s,%s,%s,%s,training failed; attempt consumed\n' "$run_id" "$gpu" "$start_epoch" "$end_epoch" "$exit_code" >> "$state/smoke_ledger.csv"
    smoke_failed=1
    continue
  fi
  gpu="$(cat "$state/current_gpu.txt")"
  if ! grep -q 'P03_SMOKE_FINITE_LOSS_GRADIENT=PASS' "$logs/smoke_train_$run_id.log"; then
    end_epoch="$(date +%s)"; printf '%s,1,FAILED,100,UNKNOWN,%s,%s,%s,90,finite loss/gradient evidence missing\n' "$run_id" "$gpu" "$start_epoch" "$end_epoch" >> "$state/smoke_ledger.csv"; smoke_failed=1; continue
  fi
  if [[ "$run_id" == "P03-corner-ru" ]]; then
    checkpoint="$destination/ckpts/ckpt_99_rank0.pt"; family=internal
    if ! run_gpu_stage "smoke_render_$run_id" smoke_render "$run_id" 300 render_internal "$checkpoint" "$report/smoke_native/$run_id" 99 quality; then end_epoch="$(date +%s)"; printf '%s,1,FAILED,100,99,%s,%s,%s,%s,smoke checkpoint render failed\n' "$run_id" "$gpu" "$start_epoch" "$end_epoch" "$LAST_STAGE_EXIT" >> "$state/smoke_ledger.csv"; smoke_failed=1; continue; fi
    prediction="$report/smoke_native/$run_id/float_predictions"; actual_step=99
    audit_python=("$eval_python")
  elif [[ "$run_id" == "P03-corner-robustsplat" ]]; then
    checkpoint="$destination/chkpnt100.pth"; family=robustsplat
    if ! run_gpu_stage "smoke_render_$run_id" smoke_render "$run_id" 300 render_robust "$destination" 100 0 0; then end_epoch="$(date +%s)"; printf '%s,1,FAILED,100,100,%s,%s,%s,%s,smoke checkpoint render failed\n' "$run_id" "$gpu" "$start_epoch" "$end_epoch" "$LAST_STAGE_EXIT" >> "$state/smoke_ledger.csv"; smoke_failed=1; continue; fi
    prediction="$destination/test/ours_100/float_predictions"; actual_step=100
    audit_python=(conda run --prefix "$robust_env" python)
  else
    checkpoint="$destination/ckpts/ckpt_99.pt"; family=sls-mlp
    if ! run_gpu_stage "smoke_render_$run_id" smoke_render "$run_id" 300 render_sls "$destination" 99 0 0; then end_epoch="$(date +%s)"; printf '%s,1,FAILED,100,99,%s,%s,%s,%s,smoke checkpoint render failed\n' "$run_id" "$gpu" "$start_epoch" "$end_epoch" "$LAST_STAGE_EXIT" >> "$state/smoke_ledger.csv"; smoke_failed=1; continue; fi
    prediction="$destination/renders/float_step0099"; actual_step=99
    audit_python=("$eval_python")
  fi
  if ! run_cpu_stage "smoke_checkpoint_$run_id" smoke_validation "$run_id" 600 \
    "${audit_python[@]}" "$code_root/tools/p02a_checkpoint_audit.py" --checkpoint "$checkpoint" --family "$family" \
      --run-id "$run_id-smoke1" --output "$report/smoke_checkpoints/$run_id.json"; then
    end_epoch="$(date +%s)"; printf '%s,1,FAILED,100,%s,%s,%s,%s,%s,smoke checkpoint reload audit failed\n' "$run_id" "$actual_step" "$gpu" "$start_epoch" "$end_epoch" "$LAST_STAGE_EXIT" >> "$state/smoke_ledger.csv"; smoke_failed=1; continue
  fi
  if ! run_cpu_stage "smoke_prediction_$run_id" smoke_validation "$run_id" 600 \
    "$eval_python" "$code_root/tools/p03_prediction_audit.py" --data-dir "$common" --prediction-dir "$prediction" \
      --run-id "$run_id-smoke1" --output "$report/smoke_predictions/$run_id.json"; then
    end_epoch="$(date +%s)"; printf '%s,1,FAILED,100,%s,%s,%s,%s,%s,frozen-view size or finiteness audit failed\n' "$run_id" "$actual_step" "$gpu" "$start_epoch" "$end_epoch" "$LAST_STAGE_EXIT" >> "$state/smoke_ledger.csv"; smoke_failed=1; continue
  fi
  end_epoch="$(date +%s)"
  printf '%s,1,PASS,100,%s,%s,%s,%s,0,finite loss/gradient; checkpoint reload; 20 frozen views correct\n' "$run_id" "$actual_step" "$gpu" "$start_epoch" "$end_epoch" >> "$state/smoke_ledger.csv"
done

if [[ "$smoke_failed" -ne 0 ]]; then
  write_state smoke FAILED "至少一个身份smoke失败；没有启动任何正式训练"
  "$eval_python" "$code_root/tools/p03_finalize_report.py" --work-root "$p03_root" --common-dir "$common" --status PARTIAL
  exit 1
fi

formal_failed=0
mkdir -p "$report/checkpoints"
for run_id in "${run_ids[@]}"; do
  destination="$outputs/$run_id/attempt_1"
  start_epoch="$(date +%s)"
  if [[ "$run_id" == "P03-corner-ru" ]]; then
    callback=train_internal; timeout_seconds=1800; family=internal
  elif [[ "$run_id" == "P03-corner-robustsplat" ]]; then
    callback=train_robust; timeout_seconds=1800; family=robustsplat
  else
    callback=train_sls; timeout_seconds=2700; family=sls-mlp
  fi
  if ! run_gpu_stage "train_$run_id" formal_training "$run_id" "$timeout_seconds" "$callback" "$destination" 30000 0; then
    exit_code="$LAST_STAGE_EXIT"; gpu="$(cat "$state/current_gpu.txt")"; end_epoch="$(date +%s)"
    printf '%s,1,FAILED,42,30000,UNKNOWN,%s,%s,%s,%s,training failed; no automatic restart\n' "$run_id" "$gpu" "$start_epoch" "$end_epoch" "$exit_code" >> "$state/formal_ledger.csv"
    formal_failed=1
    continue
  fi
  gpu="$(cat "$state/current_gpu.txt")"
  if [[ "$run_id" == "P03-corner-ru" ]]; then checkpoint="$destination/ckpts/ckpt_29999_rank0.pt"; actual_step=29999; audit_python=("$eval_python")
  elif [[ "$run_id" == "P03-corner-robustsplat" ]]; then checkpoint="$destination/chkpnt30000.pth"; actual_step=30000; audit_python=(conda run --prefix "$robust_env" python)
  else checkpoint="$destination/ckpts/ckpt_29999.pt"; actual_step=29999; audit_python=("$eval_python"); fi
  if ! run_cpu_stage "checkpoint_$run_id" checkpoint_audit "$run_id" 900 \
    "${audit_python[@]}" "$code_root/tools/p02a_checkpoint_audit.py" --checkpoint "$checkpoint" --family "$family" \
      --run-id "$run_id" --output "$report/checkpoints/$run_id.json"; then
    end_epoch="$(date +%s)"; printf '%s,1,FAILED_CHECKPOINT,42,30000,%s,%s,%s,%s,91,checkpoint audit failed\n' "$run_id" "$actual_step" "$gpu" "$start_epoch" "$end_epoch" >> "$state/formal_ledger.csv"; formal_failed=1; continue
  fi
  end_epoch="$(date +%s)"
  printf '%s,1,COMPLETE,42,30000,%s,%s,%s,%s,0,from-scratch native final checkpoint validated\n' "$run_id" "$actual_step" "$gpu" "$start_epoch" "$end_epoch" >> "$state/formal_ledger.csv"
done

if [[ "$formal_failed" -ne 0 ]]; then
  write_state formal PARTIAL "至少一项正式训练或checkpoint审计失败；没有自动重启"
  "$eval_python" "$code_root/tools/p03_finalize_report.py" --work-root "$p03_root" --common-dir "$common" --status PARTIAL
  exit 1
fi

timing_gpu="$(select_gpu timing_session)" || partial_stop timing "统一计时前没有空闲L20"
cp "$state/gpu_selection_timing_session.json" "$report/timing_gpu.json"
ru_out="$outputs/P03-corner-ru/attempt_1"
robust_out="$outputs/P03-corner-robustsplat/attempt_1"
sls_out="$outputs/P03-corner-sls-mlp/attempt_1"
run_fixed_gpu_stage timing_P03-corner-ru timing P03-corner-ru 300 "$timing_gpu" render_internal \
  "$ru_out/ckpts/ckpt_29999_rank0.pt" "$report/timing/P03-corner-ru" 29999 timing || partial_stop timing_P03-corner-ru "RU统一计时失败"
run_fixed_gpu_stage timing_P03-corner-robustsplat timing P03-corner-robustsplat 300 "$timing_gpu" render_robust "$robust_out" 30000 3 0 || partial_stop timing_P03-corner-robustsplat "RobustSplat统一计时失败"
run_fixed_gpu_stage timing_P03-corner-sls-mlp timing P03-corner-sls-mlp 300 "$timing_gpu" render_sls "$sls_out" 29999 3 0 || partial_stop timing_P03-corner-sls-mlp "SLS统一计时失败"

run_gpu_stage quality_render_P03-corner-ru quality_render P03-corner-ru 600 render_internal \
  "$ru_out/ckpts/ckpt_29999_rank0.pt" "$report/native/P03-corner-ru" 29999 quality || partial_stop quality_render_P03-corner-ru "RU浮点质量渲染失败"
run_gpu_stage memory_P03-corner-ru inference_memory P03-corner-ru 300 render_internal \
  "$ru_out/ckpts/ckpt_29999_rank0.pt" "$report/memory/P03-corner-ru" 29999 memory || partial_stop memory_P03-corner-ru "RU推理显存测量失败"
run_gpu_stage quality_render_P03-corner-robustsplat quality_render P03-corner-robustsplat 600 render_robust "$robust_out" 30000 0 1 || partial_stop quality_render_P03-corner-robustsplat "RobustSplat浮点渲染或显存测量失败"
run_gpu_stage quality_render_P03-corner-sls-mlp quality_render P03-corner-sls-mlp 600 render_sls "$sls_out" 29999 0 1 || partial_stop quality_render_P03-corner-sls-mlp "SLS浮点渲染或显存测量失败"

run_gpu_stage evaluate_P03-corner-ru independent_evaluation P03-corner-ru 600 independent_eval \
  P03-corner-ru puri-gs-ru "$report/native/P03-corner-ru/float_predictions" "$ru_out/ckpts/ckpt_29999_rank0.pt" || partial_stop evaluate_P03-corner-ru "RU独立质量评测失败"
run_gpu_stage evaluate_P03-corner-robustsplat independent_evaluation P03-corner-robustsplat 600 independent_eval \
  P03-corner-robustsplat robustsplat "$robust_out/test/ours_30000/float_predictions" "$robust_out/chkpnt30000.pth" || partial_stop evaluate_P03-corner-robustsplat "RobustSplat独立质量评测失败"
run_gpu_stage evaluate_P03-corner-sls-mlp independent_evaluation P03-corner-sls-mlp 600 independent_eval \
  P03-corner-sls-mlp sls-mlp "$sls_out/renders/float_step29999" "$sls_out/ckpts/ckpt_29999.pt" || partial_stop evaluate_P03-corner-sls-mlp "SLS独立质量评测失败"

cp "$features/ru/manifest.json" "$report/ru_feature_generation_manifest.json"
cp "$features/ru_validation.json" "$report/ru_feature_validation.json"
cp "$features/sls_generation.json" "$report/sls_feature_generation_manifest.json"
cp "$features/sls_validation.json" "$report/sls_feature_validation.json"
"$eval_python" "$code_root/tools/p03_finalize_report.py" --work-root "$p03_root" --common-dir "$common" --status COMPLETE
write_state p03 COMPLETE "P03训练、统一计时、独立质量、成本和ZIP均完成；到此停止"
echo "P03_COMPLETE_STOP"
