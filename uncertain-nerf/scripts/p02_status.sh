#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "用法: bash scripts/p02_status.sh /绝对路径/P02-work"
  exit 2
fi

work_root="$(realpath "$1")"
state_file="$work_root/state/pipeline.json"
current_log="$work_root/state/current_log.txt"

echo "P02 工作目录: $work_root"
if [[ -f "$state_file" ]]; then
  echo "流水线状态:"
  cat "$state_file"
else
  echo "流水线状态文件尚未生成。"
fi

if [[ -f "$current_log" ]]; then
  log_path="$(cat "$current_log")"
  echo "当前/最近日志: $log_path"
  if [[ -f "$log_path" ]]; then
    tail -n 40 "$log_path"
  fi
fi

echo "GPU 即时状态:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
