# P02-A 验收补全运行说明

本包只审计既有 P01/P02 资产，不训练、不续训、不提取全量特征。新增训练身份、optimizer updates 和训练 smoke 均为 0。输出固定写入 `/home/chenglong/P02-work/p02a/`，不会覆盖 P01/P02。

## 1. 本地只做一次 Git 里程碑

保持 `ru-part` 分支。将本次新增的 `p02a_*` 工具、`p02a_server_audit.sh`、`p02a_status.sh` 和本说明纳入同一个提交，建议提交备注：

`P02-A：补齐运行身份、loader、特征、计时与验收证据`

推送到远端 `ru-part` 后，不需要为模型、日志或图片重复提交。

## 2. 服务器同步和语法检查

在服务器进入 `/home/chenglong/Uncertain-Nerf`，确认分支仍为 `ru-part` 后，用图形化或已有 Git 方式拉取。随后执行：

```bash
cd /home/chenglong/Uncertain-Nerf

git branch --show-current
git rev-parse HEAD
git rev-parse origin/ru-part
git status --short

bash -n uncertain-nerf/scripts/p02a_server_audit.sh
bash -n uncertain-nerf/scripts/p02a_status.sh
```

正确标志：分支显示 `ru-part`；两个哈希完全相同；`git status --short` 没有输出；两条 `bash -n` 没有任何输出且返回命令提示符。

## 3. 一次性后台执行

把下面的 `<服务器HEAD完整哈希>` 替换为刚才 `git rev-parse HEAD` 的完整输出。日期标签只用于归档名。

```bash
cd /home/chenglong/Uncertain-Nerf

nohup bash uncertain-nerf/scripts/p02a_server_audit.sh \
  /home/chenglong/Uncertain-Nerf \
  /home/chenglong/P02-work \
  /home/chenglong/P02-data/android_colmap_common_factor4_v1 \
  /home/chenglong/P02-data/patio_high_colmap_common_v1 \
  <服务器HEAD完整哈希> \
  20260912 \
  > /home/chenglong/P02-work/p02a-launcher.log 2>&1 < /dev/null &

echo $! > /home/chenglong/P02-work/p02a.pid
echo "P02A_PID=$(cat /home/chenglong/P02-work/p02a.pid)"
```

脚本只会在不存在的 `p02a/` 中创建新证据。如果该目录已存在，会拒绝覆盖并停止。

## 4. 连续进度显示

```bash
while true; do
  clear
  date
  echo
  bash /home/chenglong/Uncertain-Nerf/uncertain-nerf/scripts/p02a_status.sh \
    /home/chenglong/P02-work

  if [ -f /home/chenglong/P02-work/p02a-state.json ]; then
    P02A_PERCENT=$(python3 -c 'import json; print(json.load(open("/home/chenglong/P02-work/p02a-state.json"))["percent"])')
    P02A_FILLED=$((P02A_PERCENT / 2))
    printf '\n['
    printf '%*s' "$P02A_FILLED" '' | tr ' ' '#'
    printf '%*s' "$((50 - P02A_FILLED))" '' | tr ' ' '-'
    printf '] %3d%%\n' "$P02A_PERCENT"
  fi

  if ! kill -0 "$(cat /home/chenglong/P02-work/p02a.pid 2>/dev/null)" 2>/dev/null; then
    echo "后台进程已经退出，请检查最终状态。"
    break
  fi
  sleep 10
done
```

可以按 `Ctrl+C` 退出监视，不会终止后台审计。SSH 断开或本地电脑关机也不会终止 `nohup` 后台进程。

## 5. 完成判据

```bash
cat /home/chenglong/P02-work/p02a-state.json
cat /home/chenglong/P02-work/p02a/status.json
tail -n 60 /home/chenglong/P02-work/p02a-launcher.log
sha256sum -c /home/chenglong/P02-work/P02A-final-20260912.tar.gz.sha256
```

必须同时看到：

- live state 为 `"status":"COMPLETE"`；
- `new_training_identities`、`optimizer_updates`、`new_training_smoke` 均为 0；
- `quality_rows=8`、`per_image_metric_rows=256`、`timing_repeats=24`；
- `next_work_package_started=false`；
- 日志末尾有 `P02A_COMPLETE_STOP`；
- 校验输出以 `OK` 结束。

任何阶段失败时不要删除 `p02a/` 或重跑。先保留 `p02a-state.json`、`p02a-launcher.log` 和对应 `p02a/logs/`，只针对一个明确错误处理。
