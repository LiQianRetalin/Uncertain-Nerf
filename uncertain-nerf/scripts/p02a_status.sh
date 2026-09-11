#!/usr/bin/env bash
set -Eeuo pipefail

work_root="${1:-/home/chenglong/P02-work}"
state="$work_root/p02a-state.json"
log="$work_root/p02a-launcher.log"
echo "P02-A 工作目录: $work_root/p02a"
if [[ -f "$state" ]]; then
  cat "$state"
else
  echo '{"status":"NOT_STARTED"}'
fi
echo
echo "最新日志:"
tail -n 30 "$log" 2>/dev/null || true
echo
echo "GPU:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || true
