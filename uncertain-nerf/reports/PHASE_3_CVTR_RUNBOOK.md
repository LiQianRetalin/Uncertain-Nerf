# PURI-GS Phase 3：CVTR 实施与服务器运行手册

## 1. 已确认仓库映射

工作分支保持 `dev`，未创建分支或 worktree。现有三份 B1 10k checkpoint 不在
Windows 工作区，实际服务器路径为：

```text
/home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/android_short10k_seed42_32a7d78/b1/ckpts/ckpt_9999_rank0.pt
/home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/garden_b1_short10k_seed42_128535b/ckpts/ckpt_9999_rank0.pt
/home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/room_b1_short10k_seed42_128535b/ckpts/ckpt_9999_rank0.pt
```

官方 trainer 保存语句只写：

```python
{"step": step, "splats": self.splats.state_dict()}
```

所以三份 checkpoint 均不含 optimizer、strategy、scheduler 或 RNG state。Phase 3
launcher 会重新读取实际文件并把 key、SHA256、字节数及三个 state 布尔值写入每个
run 的 `continuation_source.json`。B1-C 和 CVTR-C 均对称使用 fresh optimizer、
strategy、scheduler 和 seed-42 RNG 初始化。

相机结构为 gsplat COLMAP `Parser` 的：

```text
parser.Ks_dict[camera_id]   -> 3x3 K
parser.camtoworlds[index]   -> 4x4 camera-to-world
Dataset[item]["K"]          -> 当前视图 K
Dataset[item]["camtoworld"] -> 当前视图 camera-to-world
```

深度固定为 `RGB+ED` 的 expected camera-z；完整审计见
`reports/PHASE_3_DEPTH_SEMANTICS.md`。

## 2. 本地派生数据

已生成且不进入 Git：

```text
E:\7-DataSet\PURI-GS-derived\room_cvtr_synthetic_v1
```

内容为 16 张 Room 训练图派生图、16 张真值 mask 和 manifest。前 8 张 clean，后
8 张为不透明单矩形纹理 patch，后 8 张面积均为 `0.0799899085`。原始 Room 数据
没有修改。

## 3. 本地上传小型派生集

本步骤目的：把约 8.35 MB 的 Room 合成派生集上传服务器；不上传完整数据集。

执行位置：Windows 本地。

需要打开的目录：`E:\6-Project\1-UncertainNerf\uncertain-nerf`

需要检查的文件：

```text
E:\7-DataSet\PURI-GS-derived\room_cvtr_synthetic_v1\manifest.json
```

需要执行的命令：

```powershell
cd 'E:\6-Project\1-UncertainNerf\uncertain-nerf'

$remote = 'chenglong@172.16.55.2'
$remotePath = '/home/chenglong/Uncertain-Nerf/uncertain-nerf/tmp/room_cvtr_synthetic_v1'
$localPath = 'E:\7-DataSet\PURI-GS-derived\room_cvtr_synthetic_v1'

ssh $remote "if test -e '$remotePath'; then echo STOP_DERIVED_EXISTS; else echo DERIVED_TARGET_AVAILABLE; fi"
scp -r $localPath "${remote}:$remotePath"
ssh $remote "find '$remotePath' -maxdepth 2 -type f | wc -l"
```

命令执行完成后应看到：`DERIVED_TARGET_AVAILABLE`，上传后文件数为 `33`。

日志位置：本步骤无独立日志，PowerShell 输出即记录。

如何判断成功：远端共 33 个文件，且 `manifest.json` 存在。

出现什么情况应停止：显示 `STOP_DERIVED_EXISTS`、上传失败或文件数不是 33。

是否需要 Git 提交：否，派生数据禁止提交。

是否需要服务器 pull：代码 push 完成后才进行一次服务器 pull。

## 4. 服务器 pull 后环境与补丁检查

本步骤目的：检查固定 L20 环境，并在已存在的 Phase 2 Python trainer patch 上增加
CVTR/continuation 增量；不修改或重编译 CUDA rasterizer。

执行位置：Linux 服务器。

需要打开的目录：`/home/chenglong/Uncertain-Nerf/uncertain-nerf`

需要检查的文件：

```text
patches/gsplat_v1.5.3_puri_gs_cvtr.patch
configs/puri_gs_b1_continuation.yaml
configs/puri_gs_cvtr.yaml
```

完成既有 pull 流程后执行：

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
set -euo pipefail

ROOT="$PWD"
PY="$ROOT/.venv-gsplat153/bin/python"
GS="$ROOT/external/gsplat-v1.5.3"
GPU=6
LOG="$ROOT/tmp/phase3_environment.log"

test -x "$PY"
test -s "$GS/examples/simple_trainer.py"
test -s "$ROOT/patches/gsplat_v1.5.3_puri_gs_cvtr.patch"

patch --dry-run -p1 -d "$GS" < "$ROOT/patches/gsplat_v1.5.3_puri_gs_cvtr.patch"
patch -p1 -d "$GS" < "$ROOT/patches/gsplat_v1.5.3_puri_gs_cvtr.patch"
patch --dry-run -R -p1 -d "$GS" < "$ROOT/patches/gsplat_v1.5.3_puri_gs_cvtr.patch"

nvidia-smi -i "$GPU" --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader | tee "$LOG"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT" "$PY" - <<'PY' | tee -a "$LOG"
import pathlib, torch, gsplat
print("torch=", torch.__version__, "cuda=", torch.version.cuda)
print("gpu=", torch.cuda.get_device_name(0))
print("capability=", torch.cuda.get_device_capability(0))
print("gsplat=", gsplat.__version__)
print("gsplat_runtime=", pathlib.Path(gsplat.__file__).resolve())
PY

PYTHONPATH="$ROOT" "$PY" -m compileall -q puri_gs tools tests run_puri_gs.py
```

命令执行完成后应看到：L20、PyTorch 2.4.0+cu121、gsplat 1.5.3，且
`gsplat_runtime` 位于固定 wheel，不在 `$GS` 源码目录内。

日志位置：`tmp/phase3_environment.log`

如何判断成功：两次 patch 检查分别为“可正向应用”和“已可反向应用”，compileall
无输出且退出码为 0。

出现什么情况应停止：patch 已经应用、patch 冲突、GPU 不是空闲 L20、import 失败、
runtime 被源码目录遮蔽。不要升级环境或重编译 rasterizer。

是否需要 Git 提交：否，这是服务器固定 checkout 的 Python 示例补丁。

是否需要服务器 pull：否，本阶段只 pull 一次。

## 5. 服务器单元测试与深度硬门

本步骤目的：先验证 CPU 逻辑，再执行真实 gsplat 单 Gaussian expected-z 数值测试。

执行位置：Linux 服务器。

需要打开的目录：`/home/chenglong/Uncertain-Nerf/uncertain-nerf`

需要检查的文件：`tests/test_cvtr*.py`

需要执行的命令：

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
set -euo pipefail

ROOT="$PWD"
PY="$ROOT/.venv-gsplat153/bin/python"
GPU=6
LOG="$ROOT/tmp/phase3_tests.log"

PYTHONPATH="$ROOT" "$PY" -m pytest -q \
  tests/test_cvtr.py \
  tests/test_cvtr_synthetic.py \
  tests/test_cvtr_continuation.py \
  tests/test_puri_gs_configs.py \
  2>&1 | tee "$LOG"

CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT" "$PY" -m pytest -q \
  tests/test_cvtr_geometry.py::test_single_gaussian_rgb_ed_is_expected_camera_z \
  2>&1 | tee -a "$LOG"
```

命令执行完成后应看到：第一组全部 passed；第二组严格显示 `1 passed`。

日志位置：`tmp/phase3_tests.log`

如何判断成功：第二组不得是 skipped。

出现什么情况应停止：pytest 缺失、任何 failed/error、单 Gaussian 测试 skipped。

是否需要 Git 提交：否。

是否需要服务器 pull：否。

## 6. Checkpoint 真实性与 state 审计

本步骤目的：确认三份 B1 源真实存在、step 为 9999，且 state 边界符合 Phase 3。

执行位置：Linux 服务器。

需要执行的命令：

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
set -euo pipefail

ROOT="$PWD"
PY="$ROOT/.venv-gsplat153/bin/python"
ANDROID_CKPT="$ROOT/logs-puri/android_short10k_seed42_32a7d78/b1/ckpts/ckpt_9999_rank0.pt"
GARDEN_CKPT="$ROOT/logs-puri/garden_b1_short10k_seed42_128535b/ckpts/ckpt_9999_rank0.pt"
ROOM_CKPT="$ROOT/logs-puri/room_b1_short10k_seed42_128535b/ckpts/ckpt_9999_rank0.pt"

"$PY" - "$ANDROID_CKPT" "$GARDEN_CKPT" "$ROOM_CKPT" <<'PY'
import hashlib, pathlib, sys, torch
for raw in sys.argv[1:]:
    path = pathlib.Path(raw)
    if not path.is_file():
        raise SystemExit(f"STOP_MISSING_CHECKPOINT {path}")
    value = torch.load(path, map_location="cpu", weights_only=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    states = {
        "optimizer": bool(set(value) & {"optimizer", "optimizers"}),
        "strategy": bool(set(value) & {"strategy", "strategy_state"}),
        "rng": bool(set(value) & {"rng", "rng_state", "random_state"}),
    }
    print(path, "step=", value.get("step"), "keys=", sorted(value), "state=", states, "sha256=", digest)
    if value.get("step") != 9999 or "splats" not in value or any(states.values()):
        raise SystemExit(f"STOP_CHECKPOINT_CONTRACT {path}")
PY
```

命令执行完成后应看到：三份均 `step=9999`、keys 为 `splats, step`，三个 state
均为 false，并输出各自 SHA256。

日志位置：需要留存时重定向到 `tmp/phase3_checkpoint_audit.log`。

如何判断成功：三个 checkpoint 全部通过，没有 A1 路径。

出现什么情况应停止：缺失、step 不符、包含额外 state、非有限 splat。

是否需要 Git 提交：否。

是否需要服务器 pull：否。

## 7. 合成 mask 门

本步骤目的：用 Room B1 与已上传的 16 帧派生集计算真实 CVTR mask 指标。

执行位置：Linux 服务器。

需要打开的目录：`/home/chenglong/Uncertain-Nerf/uncertain-nerf`

需要执行的命令：

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
set -euo pipefail

ROOT="$PWD"
PY="$ROOT/.venv-gsplat153/bin/python"
GS="$ROOT/external/gsplat-v1.5.3"
GPU=6
ROOM_DATA="$ROOT/data/mipnerf360/360_v2/room"
ROOM_CKPT="$ROOT/logs-puri/room_b1_short10k_seed42_128535b/ckpts/ckpt_9999_rank0.pt"
DERIVED="$ROOT/tmp/room_cvtr_synthetic_v1"
OUTPUT="$ROOT/analysis/cvtr_synthetic"
LOG="$ROOT/tmp/phase3_synthetic.log"

test ! -e "$OUTPUT" || { echo "STOP_OUTPUT_EXISTS $OUTPUT"; exit 3; }

set -o pipefail
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT" "$PY" \
  tools/validate_cvtr_synthetic.py \
  --data-dir "$ROOM_DATA" \
  --gsplat-dir "$GS" \
  --derived-dir "$DERIVED" \
  --checkpoint "$ROOM_CKPT" \
  --source-commit 128535bbc69940af53bc574a55e81fe6a8be060d \
  --config configs/puri_gs_cvtr.yaml \
  --output-dir "$OUTPUT" \
  --device cuda \
  2>&1 | tee "$LOG"

"$PY" - "$OUTPUT/metrics.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(json.dumps(d, indent=2))
if d["gate"] != "PASS":
    raise SystemExit("STOP_MASK_VALIDATION_FAIL")
PY
```

命令执行完成后应看到：Precision >= 0.80、Recall >= 0.50、F1 >= 0.60、
Clean FPR <= 0.02、`gate=PASS`。

日志位置：`tmp/phase3_synthetic.log`

如何判断成功：进程退出 0 且 `analysis/cvtr_synthetic/metrics.json` 为 PASS。

出现什么情况应停止：`MASK_VALIDATION_FAIL`、OOM、NaN、输入/输出缺失。失败后不
生成真实场景 mask，不调参。

是否需要 Git 提交：否，服务器结果暂不提交。

是否需要服务器 pull：否。

## 8. Android、Garden、Room mask 预检查

本步骤目的：从各自 B1 checkpoint 一次性生成固定 mask 并执行 clean guard。

执行位置：Linux 服务器。

需要执行的命令：

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
set -euo pipefail

ROOT="$PWD"
PY="$ROOT/.venv-gsplat153/bin/python"
GS="$ROOT/external/gsplat-v1.5.3"
GPU=6
MASK_ROOT="$ROOT/analysis/cvtr"
LOG_ROOT="$ROOT/tmp/phase3_masks"
mkdir -p "$LOG_ROOT"
test ! -e "$MASK_ROOT" || { echo "STOP_MASK_ROOT_EXISTS $MASK_ROOT"; exit 3; }

CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT" "$PY" tools/build_cvtr_masks.py \
  --scene android \
  --data-dir "$ROOT/data/nerf_robustnerf/robustnerf/android" \
  --checkpoint "$ROOT/logs-puri/android_short10k_seed42_32a7d78/b1/ckpts/ckpt_9999_rank0.pt" \
  --source-commit 32a7d787d6886d847061a853c66f7de90f2eebe4 \
  --gsplat-dir "$GS" --config configs/puri_gs_cvtr.yaml \
  --output-dir "$MASK_ROOT/android" --device cuda \
  --train-keyword clutter --test-keyword extra \
  2>&1 | tee "$LOG_ROOT/android.log"

CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT" "$PY" tools/build_cvtr_masks.py \
  --scene garden \
  --data-dir "$ROOT/data/mipnerf360/360_v2/garden" \
  --checkpoint "$ROOT/logs-puri/garden_b1_short10k_seed42_128535b/ckpts/ckpt_9999_rank0.pt" \
  --source-commit 128535bbc69940af53bc574a55e81fe6a8be060d \
  --gsplat-dir "$GS" --config configs/puri_gs_cvtr.yaml \
  --output-dir "$MASK_ROOT/garden" --device cuda \
  2>&1 | tee "$LOG_ROOT/garden.log"

CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT" "$PY" tools/build_cvtr_masks.py \
  --scene room \
  --data-dir "$ROOT/data/mipnerf360/360_v2/room" \
  --checkpoint "$ROOT/logs-puri/room_b1_short10k_seed42_128535b/ckpts/ckpt_9999_rank0.pt" \
  --source-commit 128535bbc69940af53bc574a55e81fe6a8be060d \
  --gsplat-dir "$GS" --config configs/puri_gs_cvtr.yaml \
  --output-dir "$MASK_ROOT/room" --device cuda \
  2>&1 | tee "$LOG_ROOT/room.log"

"$PY" - "$MASK_ROOT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
for scene in ("android", "garden", "room"):
    d = json.load(open(root / scene / "scene_statistics.json"))
    print(scene)
    for key in (
        "mean_candidate_ratio", "mean_transient_mask_ratio",
        "p95_transient_mask_ratio", "mean_valid_neighbor_ratio",
        "mean_neff_ratio", "minimum_neff_ratio", "clean_mask_guard",
    ):
        print(" ", key, d[key])
    if d["clean_mask_guard"] != "PASS":
        raise SystemExit(f"STOP_CLEAN_MASK_GUARD_FAIL {scene}")
PY
```

命令执行完成后应看到：三场景统计齐全；Garden/Room mean mask <= 0.05、p95 <=
0.12、minimum Neff >= 0.90；Android 面积 <= 0.15 且 minimum Neff >= 0.90。

日志位置：`tmp/phase3_masks/{android,garden,room}.log`

如何判断成功：三份 `scene_statistics.json` 的 guard 均为 PASS。

出现什么情况应停止：任一命令非 0、`CLEAN_MASK_GUARD_FAIL`、Neff < 0.90、
mask > 15%。不得启动 continuation，不调参数。

是否需要 Git 提交：否。

是否需要服务器 pull：否。

## 9. 六个 continuation（必须等待用户明确确认）

本步骤目的：从同一场景的同一 B1 10k source 分叉 B1-C/CVTR-C，各增加 5000 step。

执行位置：Linux 服务器。

在用户明确确认前：不要执行本节。

确认后需要执行的命令：

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
set -euo pipefail

ROOT="$PWD"
PY="$ROOT/.venv-gsplat153/bin/python"
GS="$ROOT/external/gsplat-v1.5.3"
RUNS="$ROOT/logs-puri"
GPU=6
COMMIT_SHORT="$(git rev-parse --short=7 HEAD)"

run_one() {
  local scene="$1"
  local method="$2"
  local config="$3"
  local data="$4"
  local source="$5"
  local mask_dir="$6"
  local result="$RUNS/${scene}_${method}_10kto15k_seed42_${COMMIT_SHORT}"
  local log="${result}.train.log"
  local split_args=()
  local mask_args=()

  [[ ! -e "$result" && ! -e "$log" ]] || { echo "STOP_RESULT_EXISTS $result"; return 3; }
  if [[ "$scene" == "android" ]]; then
    split_args=(--train-keyword clutter --test-keyword extra)
  fi
  if [[ "$method" == "cvtr" ]]; then
    mask_args=(--cvtr-mask-dir "$mask_dir")
  fi

  CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT" PYTHONUNBUFFERED=1 "$PY" \
    run_puri_gs.py --config "$config" --gsplat-dir "$GS" --data-dir "$data" \
    --result-dir "$result" --gpu "$GPU" --data-factor 4 \
    --resume-checkpoint "$source" "${split_args[@]}" "${mask_args[@]}" \
    2>&1 | tee "$log"
  test -s "$result/ckpts/ckpt_14999_rank0.pt"
  test -s "$result/stats/train_step14999_rank0.json"
  test -s "$result/continuation_source.json"
}

run_one android b1c configs/puri_gs_b1_continuation.yaml \
  "$ROOT/data/nerf_robustnerf/robustnerf/android" \
  "$ROOT/logs-puri/android_short10k_seed42_32a7d78/b1/ckpts/ckpt_9999_rank0.pt" \
  "$ROOT/analysis/cvtr/android"

run_one android cvtr configs/puri_gs_cvtr.yaml \
  "$ROOT/data/nerf_robustnerf/robustnerf/android" \
  "$ROOT/logs-puri/android_short10k_seed42_32a7d78/b1/ckpts/ckpt_9999_rank0.pt" \
  "$ROOT/analysis/cvtr/android"

run_one garden b1c configs/puri_gs_b1_continuation.yaml \
  "$ROOT/data/mipnerf360/360_v2/garden" \
  "$ROOT/logs-puri/garden_b1_short10k_seed42_128535b/ckpts/ckpt_9999_rank0.pt" \
  "$ROOT/analysis/cvtr/garden"

run_one garden cvtr configs/puri_gs_cvtr.yaml \
  "$ROOT/data/mipnerf360/360_v2/garden" \
  "$ROOT/logs-puri/garden_b1_short10k_seed42_128535b/ckpts/ckpt_9999_rank0.pt" \
  "$ROOT/analysis/cvtr/garden"

run_one room b1c configs/puri_gs_b1_continuation.yaml \
  "$ROOT/data/mipnerf360/360_v2/room" \
  "$ROOT/logs-puri/room_b1_short10k_seed42_128535b/ckpts/ckpt_9999_rank0.pt" \
  "$ROOT/analysis/cvtr/room"

run_one room cvtr configs/puri_gs_cvtr.yaml \
  "$ROOT/data/mipnerf360/360_v2/room" \
  "$ROOT/logs-puri/room_b1_short10k_seed42_128535b/ckpts/ckpt_9999_rank0.pt" \
  "$ROOT/analysis/cvtr/room"
```

命令执行完成后应看到：六份 `ckpt_14999_rank0.pt`，每份 source inventory 的
step/SHA 与同场景另一分支完全一致。

日志位置示例：`logs-puri/android_cvtr_10kto15k_seed42_${COMMIT_SHORT}.train.log`；
其余五个场景/方法使用同一命名规则。

查看日志命令：

```bash
tail -f /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/android_cvtr_10kto15k_seed42_*.train.log
```

如何判断成功：六个 run 均从 step 10000 到 14999，没有运行 0–10k 或 30k。

出现什么情况应停止：用户未确认、mask gate 未通过、输出已存在、source SHA 不同、
OOM/NaN/Inf、任何 run 非 0。

是否需要 Git 提交：训练后暂不提交大文件。

是否需要服务器 pull：否。

## 10. 六次独立评测与最终汇总

六个训练都成功后，按各自相同 config、split 和 step-14999 checkpoint 建立 `_eval`
目录。评测不加载 CVTR mask，证明推理路径与 B1-C 一致。

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
set -euo pipefail

ROOT="$PWD"
PY="$ROOT/.venv-gsplat153/bin/python"
GS="$ROOT/external/gsplat-v1.5.3"
RUNS="$ROOT/logs-puri"
GPU=6
COMMIT_SHORT="$(git rev-parse --short=7 HEAD)"

eval_one() {
  local scene="$1"
  local method="$2"
  local config="$3"
  local data="$4"
  local expected="$5"
  local train="$RUNS/${scene}_${method}_10kto15k_seed42_${COMMIT_SHORT}"
  local output="${train}_eval"
  local log="${train}.eval.log"
  local split_args=()
  [[ ! -e "$output" && ! -e "$log" ]] || { echo "STOP_EVAL_EXISTS $output"; return 3; }
  if [[ "$scene" == "android" ]]; then split_args=(--train-keyword clutter --test-keyword extra); fi
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT" "$PY" run_puri_gs.py \
    --config "$config" --gsplat-dir "$GS" --data-dir "$data" --result-dir "$output" \
    --gpu "$GPU" --data-factor 4 --checkpoint "$train/ckpts/ckpt_14999_rank0.pt" \
    "${split_args[@]}" 2>&1 | tee "$log"
  test -s "$output/stats/test_step14999.json"
  test "$(find "$output/renders" -maxdepth 1 -name 'test_step14999_*.png' | wc -l)" -eq "$expected"
}

eval_one android b1c configs/puri_gs_b1_continuation.yaml "$ROOT/data/nerf_robustnerf/robustnerf/android" 19
eval_one android cvtr configs/puri_gs_cvtr.yaml "$ROOT/data/nerf_robustnerf/robustnerf/android" 19
eval_one garden b1c configs/puri_gs_b1_continuation.yaml "$ROOT/data/mipnerf360/360_v2/garden" 24
eval_one garden cvtr configs/puri_gs_cvtr.yaml "$ROOT/data/mipnerf360/360_v2/garden" 24
eval_one room b1c configs/puri_gs_b1_continuation.yaml "$ROOT/data/mipnerf360/360_v2/room" 39
eval_one room cvtr configs/puri_gs_cvtr.yaml "$ROOT/data/mipnerf360/360_v2/room" 39

OUTPUT="$ROOT/analysis/cvtr_continuation"
test ! -e "$OUTPUT" || { echo "STOP_SUMMARY_EXISTS $OUTPUT"; exit 3; }

CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT" "$PY" tools/summarize_cvtr_phase.py \
  --runs-root "$RUNS" --commit-short "$COMMIT_SHORT" \
  --android-data "$ROOT/data/nerf_robustnerf/robustnerf/android" \
  --garden-data "$ROOT/data/mipnerf360/360_v2/garden" \
  --room-data "$ROOT/data/mipnerf360/360_v2/room" \
  --gsplat-dir "$GS" --mask-root "$ROOT/analysis/cvtr" \
  --synthetic-metrics "$ROOT/analysis/cvtr_synthetic/metrics.json" \
  --output-dir "$OUTPUT" --device cuda --warmup 10 --bootstrap-samples 10000 \
  2>&1 | tee "$ROOT/tmp/phase3_summary.log"
```

日志位置：六个 `*.eval.log` 和 `tmp/phase3_summary.log`。

成功标志：生成 `per_image_metrics.csv`、`summary.json`、`bootstrap.json`、
`PHASE_3B_CVTR_CONTINUATION.md`、`PHASE_3_CVTR_DECISION.md` 和 decision JSON。

停止标志：source SHA 不同、官方/复算指标不一致、输出已存在、任何评测缺图。汇总
得到 `CVTR_REJECT` 是研究结论，不是程序错误；不得自动调参。

## 11. Git 里程碑状态

本地代码实现、CPU 单元测试、合成派生工具 smoke、launcher dry-run 和补丁正反检查
已完成，可以形成用于服务器验证的代码提交。真实单 Gaussian CUDA、synthetic mask
数值门、真实 mask guard 和 continuation 尚待服务器，不能宣称 Phase 3 实验完成。

建议提交说明：

```text
implement cross-view transient responsibility mask pipeline
```

提交应包含：CVTR 源代码、测试、两个配置、三个工具、trainer 增量 patch、深度报告、
本运行手册。

提交不应包含：`E:\7-DataSet` 派生数据、`analysis/`、checkpoint、mask PNG、render
cache、训练日志和服务器结果。
