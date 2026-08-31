# PURI-GS A1 Clean Responsibility Post-mortem 操作手册

## 目的与边界

本诊断只读取已有 garden/room A1 10k checkpoint，检查 room 质量下降是否伴随
更广泛的静态困难区域降权。它不恢复 A1 候选资格，不训练、不反向传播、不改参数，
也不进入 A2/A3。

固定分析内容：

- garden 161 张、room 272 张训练图各执行一次冻结 rasterization；
- 每图统计 q 均值/分位数、`q<0.8`、`q<0.5`、`q=0.2` 比例；
- 统计低 q 区域与图内最高 10% Sobel 梯度区域的重合率；
- 统计原始有效像素数和
  `Neff=(sum(q)^2)/(sum(q^2)+epsilon)`，同时保存 `Neff/valid_pixels`；
- 累积 garden/room 像素级固定区间 q 直方图，并比较逐图分布；
- 从 clean 逐图指标中选 room A1-B1 PSNR 最差的 5 张测试图；
- 每张最差测试图按固定相机位姿分数选最近 3 张训练图：
  `center_distance/scene_scale + 0.25*(view_angle/pi)`；
- 只为这些邻近训练图保存七联责任诊断图，重复邻居只保存一次。

## 输入

```text
logs-puri/garden_a1_short10k_seed42_128535b/ckpts/ckpt_9999_rank0.pt
logs-puri/room_a1_short10k_seed42_128535b/ckpts/ckpt_9999_rank0.pt
analysis/phase2_128535b/mipnerf360_clean_per_image.json
data/mipnerf360/360_v2/garden
data/mipnerf360/360_v2/room
```

## 本机：生成并上传 Git bundle

服务器访问 GitHub 曾发生 GnuTLS 中断，因此使用 bundle 保持精确 commit。先在本机
PowerShell 执行：

```powershell
cd 'E:\6-Project\1-UncertainNerf'

git branch --show-current
git status --short

$commitShort = git rev-parse --short=7 HEAD
$bundle = "E:\7-DataSet\puri_gs_a1_postmortem_${commitShort}.bundle"

if (Test-Path -LiteralPath $bundle) {
    throw "STOP_BUNDLE_EXISTS: $bundle"
}

git bundle create $bundle dev
if ($LASTEXITCODE -ne 0) { throw 'git bundle creation failed' }

git bundle verify $bundle
Get-FileHash -Algorithm SHA256 -LiteralPath $bundle

$remote = 'chenglong@172.16.55.2'
$remoteDir = '/home/chenglong/Uncertain-Nerf/uncertain-nerf/tmp/git-transfer'

ssh $remote "mkdir -p '$remoteDir'"
scp $bundle "${remote}:$remoteDir/"
```

## 服务器：fast-forward 到诊断工具 commit

将下列 `<bundle-name>` 替换为上一步实际 bundle 文件名：

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf

git branch --show-current
git status --short

BUNDLE="$PWD/tmp/git-transfer/<bundle-name>"
test -s "$BUNDLE"
git bundle verify "$BUNDLE"
git fetch "$BUNDLE" dev
git show --no-patch --oneline FETCH_HEAD
git merge --ff-only FETCH_HEAD

git branch --show-current
git rev-parse HEAD
git status --short
```

分支必须仍为 `dev`，`git status --short` 必须无输出。

## 服务器：输入和 GPU 自检

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf

PY="$PWD/.venv-gsplat153/bin/python"
GS="$PWD/external/gsplat-v1.5.3"
RUNS="$PWD/logs-puri"
DATA_ROOT="$PWD/data/mipnerf360/360_v2"
PHASE2_ANALYSIS="$PWD/analysis/phase2_128535b"
TOOL_COMMIT_SHORT="$(git rev-parse --short=7 HEAD)"
OUTPUT="$PWD/analysis/a1_postmortem_${TOOL_COMMIT_SHORT}"

for required in \
  "$RUNS/garden_a1_short10k_seed42_128535b/ckpts/ckpt_9999_rank0.pt" \
  "$RUNS/room_a1_short10k_seed42_128535b/ckpts/ckpt_9999_rank0.pt" \
  "$PHASE2_ANALYSIS/mipnerf360_clean_per_image.json" \
  "$DATA_ROOT/garden/sparse/0/cameras.bin" \
  "$DATA_ROOT/room/sparse/0/cameras.bin"
do
  test -s "$required" && echo "FOUND $required" || echo "MISSING $required"
done

if [[ -e "$OUTPUT" ]]; then
  echo "STOP_OUTPUT_EXISTS $OUTPUT"
else
  echo "OUTPUT_AVAILABLE $OUTPUT"
fi

nvidia-smi -i 6 \
  --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
```

所有输入必须为 `FOUND`，输出必须为 `OUTPUT_AVAILABLE`，GPU 必须是空闲 L20。

## 服务器：运行只读诊断

建议在现有 screen 中运行：

```bash
screen -S puri-a1-postmortem
```

screen 内执行：

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf

PY="$PWD/.venv-gsplat153/bin/python"
GS="$PWD/external/gsplat-v1.5.3"
TOOL_COMMIT_SHORT="$(git rev-parse --short=7 HEAD)"
OUTPUT="$PWD/analysis/a1_postmortem_${TOOL_COMMIT_SHORT}"
LOG="$PWD/tmp/a1_postmortem_${TOOL_COMMIT_SHORT}.log"

set -o pipefail

CUDA_VISIBLE_DEVICES=6 \
PYTHONPATH="$PWD" \
"$PY" tools/analyze_clean_responsibility_postmortem.py \
  --runs-root "$PWD/logs-puri" \
  --run-commit-short 128535b \
  --garden-data "$PWD/data/mipnerf360/360_v2/garden" \
  --room-data "$PWD/data/mipnerf360/360_v2/room" \
  --clean-per-image-json "$PWD/analysis/phase2_128535b/mipnerf360_clean_per_image.json" \
  --gsplat-dir "$GS" \
  --output-dir "$OUTPUT" \
  --device cuda \
  --worst-test-count 5 \
  --neighbors-per-test 3 \
  --pose-angular-weight 0.25 \
  2>&1 | tee "$LOG"

POSTMORTEM_STATUS="${PIPESTATUS[0]}"
echo "POSTMORTEM_STATUS=$POSTMORTEM_STATUS"
echo "POSTMORTEM_LOG=$LOG"
```

正确结束必须显示 garden 161/161、room 272/272、`PASS_EXECUTION` 对应输出，且：

```text
POSTMORTEM_STATUS=0
```

## 服务器：检查输出并打包

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf

PY="$PWD/.venv-gsplat153/bin/python"
TOOL_COMMIT_SHORT="$(git rev-parse --short=7 HEAD)"
OUTPUT="$PWD/analysis/a1_postmortem_${TOOL_COMMIT_SHORT}"

find "$OUTPUT" -maxdepth 2 -type f -printf '%P %s bytes\n' | sort

"$PY" - "$OUTPUT/clean_responsibility_scene_summary.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print("execution=", d["execution_decision"])
for scene in ("garden", "room"):
    s = d["scenes"][scene]
    print(scene, "train_images=", s["train_image_count"])
    for field in (
        "q_mean", "q_lt_0_8_ratio", "q_lt_0_5_ratio", "q_eq_0_2_ratio",
        "edge_overlap_q_0_8", "edge_overlap_q_0_5", "neff_ratio",
    ):
        print(" ", field, s["image_weighted_statistics"][field]["mean"])
print("room_minus_garden=", d["room_minus_garden"])
PY

ARCHIVE="$PWD/tmp/puri_gs_a1_postmortem_${TOOL_COMMIT_SHORT}.tar.gz"
if [[ -e "$ARCHIVE" ]]; then
  echo "STOP_ARCHIVE_EXISTS $ARCHIVE"
else
  tar -czf "$ARCHIVE" -C "$PWD/analysis" "a1_postmortem_${TOOL_COMMIT_SHORT}"
  ls -lh "$ARCHIVE"
fi
```

## 本机：下载证据

将 `<tool-commit-short>` 替换为服务器实际短哈希：

```powershell
$remote = 'chenglong@172.16.55.2'
$short = '<tool-commit-short>'
$remoteFile = "/home/chenglong/Uncertain-Nerf/uncertain-nerf/tmp/puri_gs_a1_postmortem_${short}.tar.gz"
$localFile = "C:\Users\liqian\Desktop\puri_gs_a1_postmortem_${short}.tar.gz"

scp "${remote}:$remoteFile" $localFile
Get-Item -LiteralPath $localFile | Select-Object FullName,Length,LastWriteTime
Get-FileHash -Algorithm SHA256 -LiteralPath $localFile
```

## 输出

```text
clean_responsibility_per_train_image.csv
clean_responsibility_per_train_image.json
clean_responsibility_scene_summary.json
clean_responsibility_distribution.png
room_worst_test_neighbors.json
room_worst_test_neighbors/*.png
```

出现输入缺失、split 不一致、checkpoint step 不为 9999、固定 A1 配置不同、
gsplat source checkout 遮蔽 wheel、OOM、NaN/Inf 或输出已存在时立即停止，不删除
结果、不重跑、不在服务器临时改代码。
