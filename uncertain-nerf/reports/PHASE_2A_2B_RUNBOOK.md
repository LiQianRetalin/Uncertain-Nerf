# PURI-GS Phase 2A–2B 操作手册

本手册只分析固定 A1 并运行 garden/room 10k clean 短筛；不修改 A1 参数，
不实现 A2/A3，不改变 PyTorch、CUDA、gsplat 或 rasterizer。

## 已审计路径

- 本地数据根：`E:\7-DataSet\nerf数据集\mipnerf360\360_v2`
- garden：185 张 `images`、185 张 `images_4`、1 个相机、185 个注册视图、138766 个点
- room：311 张 `images`、311 张 `images_4`、1 个相机、311 个注册视图、112627 个点
- Android 结果根：`logs-puri/android_short10k_seed42_32a7d78`
- Android A1 checkpoint：`logs-puri/android_short10k_seed42_32a7d78/a1/ckpts/ckpt_9999_rank0.pt`
- Android A1 训练图清单：`logs-puri/android_short10k_seed42_32a7d78/a1/dataset_split.json` 的 `train`
- Android 测试真值源文件：`data/nerf_robustnerf/robustnerf/android/images_4/` 下由上述 split 的 `test` 列表指定的 19 张图
- 实际评测像素：`*_eval/renders/test_step9999_*.png` 左半幅；这是 loader 完成相机畸变处理后的真值

## 本地 Windows：生成两个最小 tar

在本机 PowerShell 执行。命令拒绝覆盖已存在的同名 tar。

```powershell
$sourceRoot = 'E:\7-DataSet\nerf数据集\mipnerf360\360_v2'
$gardenTar = 'E:\7-DataSet\puri_gs_mipnerf360_garden_minimal.tar'
$roomTar = 'E:\7-DataSet\puri_gs_mipnerf360_room_minimal.tar'

foreach ($archive in @($gardenTar, $roomTar)) {
    if (Test-Path -LiteralPath $archive) {
        throw "STOP_ARCHIVE_EXISTS: $archive"
    }
}

tar.exe -cf $gardenTar -C $sourceRoot garden/images garden/images_4 garden/sparse/0/cameras.bin garden/sparse/0/images.bin garden/sparse/0/points3D.bin
if ($LASTEXITCODE -ne 0) { throw 'garden tar failed' }

tar.exe -cf $roomTar -C $sourceRoot room/images room/images_4 room/sparse/0/cameras.bin room/sparse/0/images.bin room/sparse/0/points3D.bin
if ($LASTEXITCODE -ne 0) { throw 'room tar failed' }

Get-Item -LiteralPath $gardenTar, $roomTar | Select-Object FullName, Length, LastWriteTime
tar.exe -tf $gardenTar | Select-Object -First 10
tar.exe -tf $roomTar | Select-Object -First 10
```

预期未压缩输入规模约为 garden 2.13 GiB、room 1.03 GiB。tar 位于
`E:\7-DataSet`，不在仓库中，不进入 Git。

## 本地 Windows：上传 tar

继续在本机 PowerShell 执行；不要使用 `BatchMode=yes`，SSH/SCP 会提示输入密码。

```powershell
$remote = 'chenglong@172.16.55.2'
$remoteTemp = '/home/chenglong/Uncertain-Nerf/uncertain-nerf/tmp/mipnerf360_phase2b_transfer'
$gardenTar = 'E:\7-DataSet\puri_gs_mipnerf360_garden_minimal.tar'
$roomTar = 'E:\7-DataSet\puri_gs_mipnerf360_room_minimal.tar'

ssh $remote "mkdir -p '$remoteTemp'"
scp $gardenTar $roomTar "${remote}:$remoteTemp/"
```

## Linux 服务器：同步代码与解包

先在服务器项目目录确认 `git status --short` 无输出，再 fast-forward pull `dev`。
pull 后记录的 `COMMIT_SHORT` 会用于全部新结果目录。

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf

git status --short
git pull --ff-only origin dev
git branch --show-current
git rev-parse HEAD

COMMIT_SHORT="$(git rev-parse --short=7 HEAD)"
echo "COMMIT_SHORT=${COMMIT_SHORT}"
```

解包前拒绝覆盖服务器上已有的 garden/room：

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf

TRANSFER="$PWD/tmp/mipnerf360_phase2b_transfer"
DATA_ROOT="$PWD/data/mipnerf360/360_v2"

test -s "$TRANSFER/puri_gs_mipnerf360_garden_minimal.tar"
test -s "$TRANSFER/puri_gs_mipnerf360_room_minimal.tar"

if [[ -e "$DATA_ROOT/garden" || -e "$DATA_ROOT/room" ]]; then
  echo "STOP_DATA_TARGET_EXISTS: inspect existing garden/room before extraction"
  exit 3
fi

mkdir -p "$DATA_ROOT"
tar -xf "$TRANSFER/puri_gs_mipnerf360_garden_minimal.tar" -C "$DATA_ROOT"
tar -xf "$TRANSFER/puri_gs_mipnerf360_room_minimal.tar" -C "$DATA_ROOT"
```

只读完整性检查：

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf

DATA_ROOT="$PWD/data/mipnerf360/360_v2"

for scene in garden room; do
  for required in images images_4 sparse/0/cameras.bin sparse/0/images.bin sparse/0/points3D.bin; do
    test -e "$DATA_ROOT/$scene/$required" || { echo "MISSING $scene/$required"; exit 3; }
  done
done

GARDEN_IMAGES="$(find "$DATA_ROOT/garden/images" -maxdepth 1 -type f | wc -l)"
GARDEN_IMAGES4="$(find "$DATA_ROOT/garden/images_4" -maxdepth 1 -type f | wc -l)"
ROOM_IMAGES="$(find "$DATA_ROOT/room/images" -maxdepth 1 -type f | wc -l)"
ROOM_IMAGES4="$(find "$DATA_ROOT/room/images_4" -maxdepth 1 -type f | wc -l)"

echo "garden images=${GARDEN_IMAGES} images_4=${GARDEN_IMAGES4}"
echo "room images=${ROOM_IMAGES} images_4=${ROOM_IMAGES4}"

[[ "$GARDEN_IMAGES" -eq 185 && "$GARDEN_IMAGES4" -eq 185 ]] || exit 3
[[ "$ROOM_IMAGES" -eq 311 && "$ROOM_IMAGES4" -eq 311 ]] || exit 3
```

服务器 `data/`、`tmp/`、`logs-puri/` 和 `analysis/` 均被 Git 忽略。

## Linux 服务器：Phase 2A Android 已有结果分析

这两条命令不训练。第一条只读分析已有 19 个测试结果，第二条冻结 A1 checkpoint，
对 8 个确定性抽取的训练视图各执行一次 rasterization，无反向传播。

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf

PY="$PWD/.venv-gsplat153/bin/python"
GS="$PWD/external/gsplat-v1.5.3"
ANDROID_DATA="$PWD/data/nerf_robustnerf/robustnerf/android"
ANDROID_RESULT="$PWD/logs-puri/android_short10k_seed42_32a7d78"
COMMIT_SHORT="$(git rev-parse --short=7 HEAD)"
ANALYSIS="$PWD/analysis/phase2_${COMMIT_SHORT}"

CUDA_VISIBLE_DEVICES=6 PYTHONPATH="$PWD" "$PY" tools/analyze_android_pairwise.py --result-root "$ANDROID_RESULT" --output-dir "$ANALYSIS" --device cuda

CUDA_VISIBLE_DEVICES=6 PYTHONPATH="$PWD" "$PY" tools/analyze_responsibility_selectivity.py --a1-result-dir "$ANDROID_RESULT/a1" --data-dir "$ANDROID_DATA" --gsplat-dir "$GS" --output-dir "$ANALYSIS" --device cuda
```

第一条必须生成 `PAIRWISE_STABLE`、`PAIRWISE_BORDERLINE` 或
`PAIRWISE_UNSTABLE`。第二条应生成 `PENDING_EIGHT_FRAME_VISUAL_REVIEW`，这是正常状态；
边缘敏感性必须结合 8 张图人工分类。

## Linux 服务器：六个 clean 10k run 与独立评测

以下整段在一个服务器 Bash 终端中执行。它先验证空闲 L20 和固定环境，再严格顺序
运行 garden B0/B1/A1、room B0/B1/A1。任何错误会因 `set -euo pipefail` 停止。

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
set -euo pipefail

ROOT="$PWD"
PY="$ROOT/.venv-gsplat153/bin/python"
GS="$ROOT/external/gsplat-v1.5.3"
RUNS="$ROOT/logs-puri"
DATA_ROOT="$ROOT/data/mipnerf360/360_v2"
GPU=6
COMMIT_SHORT="$(git rev-parse --short=7 HEAD)"

[[ "$(git branch --show-current)" == "dev" ]] || { echo STOP_BRANCH; exit 3; }

GPU_NAME="$(nvidia-smi -i "$GPU" --query-gpu=name --format=csv,noheader | xargs)"
GPU_STATE="$(nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
IFS=',' read -r GPU_MEMORY GPU_UTIL <<<"$GPU_STATE"
echo "gpu=$GPU name=$GPU_NAME memory_mib=$GPU_MEMORY utilization=$GPU_UTIL"
[[ "$GPU_NAME" == *"NVIDIA L20"* ]] || { echo STOP_GPU_MODEL; exit 3; }
(( GPU_MEMORY <= 1024 && GPU_UTIL <= 5 )) || { echo STOP_GPU_BUSY; exit 3; }

bash scripts/prepare_gsplat_robot_baseline.sh "$GS"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT" "$PY" scripts/verify_gsplat_robot_install.py --expected-gpu "NVIDIA L20"

GS_RUNTIME="$(CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT" "$PY" -c 'import pathlib, gsplat; print(pathlib.Path(gsplat.__file__).resolve())')"
echo "gsplat_runtime_path=$GS_RUNTIME"
[[ "$GS_RUNTIME" != "$GS"/* ]] || { echo STOP_GSPLAT_SHADOWED; exit 3; }

run_one() {
  local scene="$1"
  local profile="$2"
  local config="$3"
  local expected_train="$4"
  local expected_test="$5"
  local data="$DATA_ROOT/$scene"
  local name="${scene}_${profile}_short10k_seed42_${COMMIT_SHORT}"
  local result="$RUNS/$name"
  local evaluate="${result}_eval"

  for target in "$result" "$evaluate" "${result}.train.log" "${result}.eval.log"; do
    [[ ! -e "$target" ]] || { echo "STOP_RESULT_EXISTS $target"; return 3; }
  done

  PURI_GSPLAT_PYTHON="$PY" bash scripts/train_puri_gs.sh "$config" "$GS" "$data" "$result" "$GPU" 10000 4 2>&1 | tee "${result}.train.log"

  test -s "$result/ckpts/ckpt_9999_rank0.pt"
  test -s "$result/stats/train_step9999_rank0.json"
  "$PY" - "$result/dataset_split.json" "$expected_train" "$expected_test" <<'PY'
import json, sys
split = json.load(open(sys.argv[1]))
actual = (len(split["train"]), len(split["test"]))
expected = (int(sys.argv[2]), int(sys.argv[3]))
if actual != expected:
    raise SystemExit(f"STOP_SPLIT expected={expected} actual={actual}")
print(f"split=PASS train={actual[0]} test={actual[1]}")
PY

  PYTHONUNBUFFERED=1 "$PY" run_puri_gs.py --config "$config" --gsplat-dir "$GS" --data-dir "$data" --result-dir "$evaluate" --gpu "$GPU" --data-factor 4 --checkpoint "$result/ckpts/ckpt_9999_rank0.pt" 2>&1 | tee "${result}.eval.log"

  test -s "$evaluate/stats/test_step9999.json"
  test "$(find "$evaluate/renders" -maxdepth 1 -type f -name 'test_step9999_*.png' | wc -l)" -eq "$expected_test"
}

run_one garden b0 configs/puri_gs_b0_default.yaml 161 24
run_one garden b1 configs/puri_gs_b1_absgrad.yaml 161 24
run_one garden a1 configs/puri_gs_a1_responsibility.yaml 161 24
run_one room b0 configs/puri_gs_b0_default.yaml 272 39
run_one room b1 configs/puri_gs_b1_absgrad.yaml 272 39
run_one room a1 configs/puri_gs_a1_responsibility.yaml 272 39
```

训练日志位于对应 `logs-puri/*.<train|eval>.log`。另开终端可执行：

```bash
tail -f /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/garden_b0_short10k_seed42_*.train.log
```

## Linux 服务器：clean 逐图指标、p50/p95 与门禁汇总

该命令再次独立加载六份 checkpoint，用固定 10 帧预热，逐测试视图计时和计算同一
PSNR/SSIM/Alex-LPIPS；同时核对重算均值与官方 eval JSON。

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf

PY="$PWD/.venv-gsplat153/bin/python"
GS="$PWD/external/gsplat-v1.5.3"
COMMIT_SHORT="$(git rev-parse --short=7 HEAD)"
ANALYSIS="$PWD/analysis/phase2_${COMMIT_SHORT}"

CUDA_VISIBLE_DEVICES=6 PYTHONPATH="$PWD" "$PY" tools/summarize_clean_screen.py --runs-root "$PWD/logs-puri" --commit-short "$COMMIT_SHORT" --garden-data "$PWD/data/mipnerf360/360_v2/garden" --room-data "$PWD/data/mipnerf360/360_v2/room" --gsplat-dir "$GS" --output-dir "$ANALYSIS" --device cuda --warmup 10
```

## 收集人工复核材料

复制每个 clean 场景的首、中、末三个同序测试视图；不修改原结果。

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf

COMMIT_SHORT="$(git rev-parse --short=7 HEAD)"
ANALYSIS="$PWD/analysis/phase2_${COMMIT_SHORT}"
REVIEW="$ANALYSIS/clean_visual_review"
mkdir -p "$REVIEW"

for profile in b0 b1 a1; do
  for index in 0000 0012 0023; do
    cp "$PWD/logs-puri/garden_${profile}_short10k_seed42_${COMMIT_SHORT}_eval/renders/test_step9999_${index}.png" "$REVIEW/garden_${profile}_${index}.png"
  done
  for index in 0000 0019 0038; do
    cp "$PWD/logs-puri/room_${profile}_short10k_seed42_${COMMIT_SHORT}_eval/renders/test_step9999_${index}.png" "$REVIEW/room_${profile}_${index}.png"
  done
done

EVIDENCE="$PWD/tmp/puri_gs_phase2_evidence_${COMMIT_SHORT}.tar.gz"
[[ ! -e "$EVIDENCE" ]] || { echo "STOP_EVIDENCE_EXISTS $EVIDENCE"; exit 3; }
tar -czf "$EVIDENCE" -C "$PWD/analysis" "phase2_${COMMIT_SHORT}"
echo "$EVIDENCE"
```

将证据包下载回本机并提供给 Codex。最终边缘敏感性和 clean 纹理复核完成后，才生成：

- `reports/PHASE_2A_ANDROID_PAIRED_ANALYSIS.md`
- `reports/PHASE_2A_RESPONSIBILITY_SELECTIVITY.md`
- `reports/PHASE_2B_CLEAN_SCREEN.md`
- `reports/PHASE_2_DECISION.md`
- `reports/phase_2_decision.json`

在最终决策前不进入 A2/A3，不调参数，不运行 30k 或第二随机种子。

## 失败停止条件

- 当前分支不是 `dev` 或 pull 不是 fast-forward；
- Android 19 张输出集合、顺序或嵌入真值不一致；
- garden/room 数量不是 185/311，或 COLMAP 三文件缺失；
- split 不是 garden 161/24、room 272/39；
- GPU 不是空闲 L20；
- `gsplat_runtime_path` 指向 external checkout；
- import/CUDA、NaN/Inf、OOM、checkpoint 或评测渲染缺失；
- 分析目录或训练目录已存在；
- 重算指标与官方指标超过工具内固定容差。
