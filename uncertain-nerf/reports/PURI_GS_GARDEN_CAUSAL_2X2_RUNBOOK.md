# PURI-GS-RU Garden 2×2 因果试验实施与运行手册

本手册只补齐 `Y01=DG-only` 与 `Y10=Mask-only`，复用已经审计通过的
`Y00=Garden B1` 和 `Y11=Garden RU`。不得据此自动实现 RU 新版本、进入阶段 U、
执行 3×3、改变阈值或启动额外训练。各步骤必须严格串行；上一停点未经 Codex
复核，不执行下一步。

## 已锁定的四象限

| 象限 | M | T | 配置 | 拓扑事实 |
| --- | ---: | ---: | --- | --- |
| Y00 B1 | 0 | 0 | `configs/puri_gs_b1_full30k.yaml` | refine 600..14900/100，共144次；实际reset=[] |
| Y01 DG-only | 0 | 1 | `configs/puri_gs_garden_dg_only_full30k.yaml` | refine 10000..19900/100，共100次；reset=[15000,18000] |
| Y10 Mask-only | 1 | 0 | `configs/puri_gs_garden_mask_only_full30k.yaml` | 与实际B1完全一致；reset=[]，head更新30000次、暂停0次 |
| Y11 RU | 1 | 1 | `configs/puri_gs_ru_full30k.yaml` | 与Y01一致；head更新29400次、暂停600次 |

T=0 中 `reset=[]` 不是新改动：固定 gsplat 1.5.3 源码的 reset 表达式在 Python
运算符优先级下不产生事件。因果试验必须复现已有 B1 的实际行为，不能趁本试验
修复它。

所有象限保持：30,000 steps、seed 42、SH 3、factor 4、test_every 8、SSIM 0.2、
AbsGrad、`grow_grad2d=0.0006`、gsplat 1.5.3。新配置中不活动子系统的字段必须缺席。

## 本地代码里程碑的通过标志

Codex 已执行以下自检；最终结果以工作区当前测试输出为准：

```text
Garden专项（配置/事件/命令行/补丁/汇总/拒绝覆盖）：13 passed
全仓回归：212 passed, 2 skipped
Python compileall：PYTHON-COMPILEALL-PASS
Shell语法：SHELL-BASH-N-PASS
运行手册6个bash代码块：RUNBOOK-BASH-BLOCKS-PASS
两个30k命令 dry-run：PASS
补丁在 gsplat v1.5.3 + 基础补丁 + RU补丁上顺序应用：PASS
补丁在历史 efficiency/On-the-go RU superset 上叠加：SUPERSET-PREPARE-PASS
```

新增配置的逐字段因果差异：

| 对比 | 唯一能力变化 | 保持不变 |
| --- | --- | --- |
| DG-only vs B1 | T: 0→1 | M=0及全部公共训练字段 |
| DG-only vs RU | M: 1→0 | T=1及全部公共训练字段 |
| Mask-only vs B1 | M: 0→1 | T=0及全部公共训练字段 |
| Mask-only vs RU | T: 1→0 | M=1及全部公共训练字段 |

现有 B1/RU 的配置文件没有修改；旧 `--puri_gs_ru_enabled` 仍同时解析为 M=1、T=1。

## 步骤 1：Git 里程碑 1（当前停点）

本步骤目的：提交并推送已经自检通过的因果试验代码，不包含任何实验大文件。

执行位置：Windows 本地。

需要打开的目录：`E:\6-Project\1-UncertainNerf\uncertain-nerf`

需要检查的文件：图形化 Git 的 Changes/变更列表。

需要执行的命令：不执行 Git 命令；按用户规则只使用图形化 Git。

图形化 Git 操作：

1. 确认当前分支仍是 `dev`。
2. 提交说明填写：`add Garden 2x2 causal ablation for PURI-GS-RU`。
3. 提交本手册“里程碑 1 应包含文件”列出的代码、小配置、测试和说明。
4. 确认没有大文件后提交并 push `dev`。

命令执行完成后应看到：图形化 Git 显示 push 成功，工作区没有本次任务留下的未提交文件。

日志位置：图形化 Git 的提交详情和 push 记录。

如何判断成功：`dev` 上出现一个包含本任务全部小文件的提交；远端也显示相同提交。

出现什么情况应立即停止：分支不是 `dev`；出现 checkpoint、日志、数据、DINO、缓存、
PNG、视频、`tmp/` 或其他不认识的大文件；push 冲突或失败。

是否需要 Git 提交：是，本步骤只提交一次。

是否需要服务器 pull：本步骤完成后先不要自行 pull；把结果发给 Codex，复核后只 pull 一次。

完成后需要发给 Codex 的内容：提交哈希、图形化 Git 的文件列表、push 成功状态和工作区状态。

### 里程碑 1 应包含文件

```text
configs/puri_gs_garden_dg_only_full30k.yaml
configs/puri_gs_garden_mask_only_full30k.yaml
patches/gsplat_v1.5.3_puri_gs_garden_causal.patch
puri_gs/config.py
puri_gs/delayed_absgrad.py
puri_gs/ru_training.py
run_puri_gs.py
scripts/prepare_puri_gs_ru.sh
tests/test_garden_causal_2x2.py
tests/test_garden_causal_summary.py
tools/summarize_puri_gs_garden_causal_2x2.py
reports/PURI_GS_GARDEN_CAUSAL_2X2_RUNBOOK.md
```

不得包含：`E:\7-DataSet` 下任何内容、DINO 源码/权重、feature cache、checkpoint、
训练日志、独立评测输出、PNG、视频、`tmp/` 和大型证据包。

---

下面是后续步骤的已审核代码记录，不代表可以越过当前停点执行。

## 步骤 2：服务器 pull 后环境与 CUDA smoke

本步骤目的：一次 pull 后验证服务器代码、GPU 6、固定环境、数据、DINO/cache、两个能力开关和标准 checkpoint 评测路径。

执行位置：Linux 服务器。

需要打开的目录：`/home/chenglong/Uncertain-Nerf/uncertain-nerf`

需要检查的文件：两个新配置、因果补丁、Garden 数据、DINO 权重、feature cache manifest。

需要执行的命令：先通过既有图形化 Git/服务器同步界面把 `dev` pull 一次；不要在 shell 中创建分支、rebase 或 reset。随后整段执行：

```bash
set -euo pipefail
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf

echo "branch=$(git branch --show-current)"
echo "commit=$(git rev-parse HEAD)"
git status --porcelain=v1 --untracked-files=all
test -z "$(git status --porcelain=v1 --untracked-files=all)"

ROOT=/home/chenglong/Uncertain-Nerf/uncertain-nerf
PY="$ROOT/.venv-gsplat153/bin/python"
GS="$ROOT/external/gsplat-v1.5.3-ru"
DATA="$ROOT/data/mipnerf360/360_v2/garden"
DINO_REPO="$ROOT/external/dinov2"
DINO_WEIGHT="$ROOT/data/PURI-GS-assets/dinov2/dinov2_vits14_reg4_pretrain.pth"
CACHE="$ROOT/data/PURI-GS-derived/semantic_features/garden"
CONSOLE="$ROOT/logs-puri/phase_r_causal-console"
SMOKE="$ROOT/logs-puri/phase_r_causal-smoke"

test "$(git branch --show-current)" = dev
test -x "$PY"
test -f "$ROOT/configs/puri_gs_garden_dg_only_full30k.yaml"
test -f "$ROOT/configs/puri_gs_garden_mask_only_full30k.yaml"
test -f "$DATA/sparse/0/cameras.bin"
test -f "$DATA/sparse/0/images.bin"
test -f "$DATA/sparse/0/points3D.bin"
test -d "$DATA/images_4"
test -f "$DINO_REPO/hubconf.py"
test -f "$DINO_WEIGHT"
test -f "$CACHE/manifest.json"
test ! -e "$SMOKE/garden_dg_only_10"
test ! -e "$SMOKE/garden_mask_only_10"
test ! -e "$SMOKE/garden_dg_only_10_eval"
test ! -e "$SMOKE/garden_mask_only_10_eval"
mkdir -p "$CONSOLE" "$SMOKE"

nvidia-smi --query-gpu=index,name --format=csv,noheader | grep -Fx "6, NVIDIA L20"
"$PY" - <<'PY'
import torch
import gsplat
print("python-imports=PASS")
print("torch=", torch.__version__)
print("cuda_runtime=", torch.version.cuda)
print("gsplat=", gsplat.__version__)
assert torch.__version__ == "2.4.0+cu121"
assert gsplat.__version__.split("+")[0] == "1.5.3"
print("PINNED-RUNTIME-PASS")
PY

echo "garden_images=$(find "$DATA/images" -maxdepth 1 -type f | wc -l)"
echo "garden_images_4=$(find "$DATA/images_4" -maxdepth 1 -type f | wc -l)"
test "$(find "$DATA/images" -maxdepth 1 -type f | wc -l)" -eq 185
test "$(find "$DATA/images_4" -maxdepth 1 -type f | wc -l)" -eq 185
sha256sum "$DINO_WEIGHT"
test "$(sha256sum "$DINO_WEIGHT" | awk '{print $1}')" = \
  f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb
git -C "$DINO_REPO" rev-parse HEAD
test "$(git -C "$DINO_REPO" rev-parse HEAD)" = \
  7764ea0f912e53c92e82eb78a2a1631e92725fc8

bash scripts/prepare_puri_gs_ru.sh "$GS"

"$PY" - <<'PY'
from pathlib import Path
from puri_gs.config import causal_factors, load_experiment_config
root = Path("/home/chenglong/Uncertain-Nerf/uncertain-nerf")
expected = {
    "puri_gs_b1_full30k.yaml": (False, False),
    "puri_gs_garden_dg_only_full30k.yaml": (False, True),
    "puri_gs_garden_mask_only_full30k.yaml": (True, False),
    "puri_gs_ru_full30k.yaml": (True, True),
}
for name, factors in expected.items():
    actual = causal_factors(load_experiment_config(root / "configs" / name))
    assert actual == factors, (name, actual, factors)
    print(name, "M=", int(actual[0]), "T=", int(actual[1]))
print("GARDEN-FOUR-QUADRANT-CONFIG-PASS")
PY

CUDA_VISIBLE_DEVICES=6 "$PY" tools/check_puri_gs_ru_environment.py \
  --gsplat-dir "$GS" \
  --dino-repo-dir "$DINO_REPO" \
  --dino-weight-path "$DINO_WEIGHT" \
  --feature-cache-dir "$CACHE" \
  --device cuda:0 \
  --output "$CONSOLE/garden_causal_environment.json" \
  2>&1 | tee "$CONSOLE/garden_causal_environment.log"

"$PY" run_puri_gs.py \
  --config "$ROOT/configs/puri_gs_garden_dg_only_full30k.yaml" \
  --gsplat-dir "$GS" --data-dir "$DATA" \
  --result-dir "$SMOKE/garden_dg_only_10" --gpu 6 --max-steps 10 \
  2>&1 | tee "$CONSOLE/garden_dg_only_10.log"

"$PY" run_puri_gs.py \
  --config "$ROOT/configs/puri_gs_garden_mask_only_full30k.yaml" \
  --gsplat-dir "$GS" --data-dir "$DATA" \
  --result-dir "$SMOKE/garden_mask_only_10" --gpu 6 --max-steps 10 \
  --dino-repo-dir "$DINO_REPO" \
  --dino-weight-path "$DINO_WEIGHT" \
  --feature-cache-dir "$CACHE" \
  2>&1 | tee "$CONSOLE/garden_mask_only_10.log"

"$PY" run_puri_gs.py \
  --config "$ROOT/configs/puri_gs_garden_dg_only_full30k.yaml" \
  --gsplat-dir "$GS" --data-dir "$DATA" \
  --result-dir "$SMOKE/garden_dg_only_10_eval" --gpu 6 \
  --checkpoint "$SMOKE/garden_dg_only_10/ckpts/ckpt_9_rank0.pt" \
  2>&1 | tee "$CONSOLE/garden_dg_only_10_eval.log"

"$PY" run_puri_gs.py \
  --config "$ROOT/configs/puri_gs_garden_mask_only_full30k.yaml" \
  --gsplat-dir "$GS" --data-dir "$DATA" \
  --result-dir "$SMOKE/garden_mask_only_10_eval" --gpu 6 \
  --checkpoint "$SMOKE/garden_mask_only_10/ckpts/ckpt_9_rank0.pt" \
  2>&1 | tee "$CONSOLE/garden_mask_only_10_eval.log"

"$PY" - <<'PY'
import json
from pathlib import Path
root = Path("/home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/phase_r_causal-smoke")
for name, factors in (("garden_dg_only_10", (0, 1)), ("garden_mask_only_10", (1, 0))):
    contract = json.loads((root / name / "causal_contract.json").read_text())
    assert (contract["M"], contract["T"]) == factors
    checkpoint = root / name / "ckpts" / "ckpt_9_rank0.pt"
    assert checkpoint.is_file()
for name in ("garden_dg_only_10_eval", "garden_mask_only_10_eval"):
    validation = json.loads((root / name / "ru_validation.json").read_text())
    assert validation["standard_checkpoint_load_pass"] is True
    assert validation["evaluation_imported_dino"] is False
    assert validation["evaluation_loaded_mask_head"] is False
    assert validation["evaluation_rasterization_count_ratio"] == 1.0
print("GARDEN-CAUSAL-CUDA-SMOKE-PASS")
PY
```

命令执行完成后应看到：GPU 行为 `6, NVIDIA L20`；`PINNED-RUNTIME-PASS`；
`PURI-GS-GARDEN-CAUSAL-PATCH-READY`；`GARDEN-FOUR-QUADRANT-CONFIG-PASS`；
环境 JSON 状态为 `PURI_GS_RU_ENVIRONMENT_READY`；DG 日志含 `M=0 T=1`，Mask 日志含
`M=1 T=0`；最后为 `GARDEN-CAUSAL-CUDA-SMOKE-PASS`。

日志位置：`/home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/phase_r_causal-console/`

如何判断成功：命令退出码为 0 且上述标志全部出现；两个评测都走 standard splats-only 单次 rasterization。

出现什么情况应立即停止：工作区在 pull 前不干净；GPU 6 不是 L20；版本、DINO SHA
`f4331770...641fb`、DINO commit `7764ea0f...5fc8`、数据计数185/185、cache manifest、
补丁、任一 smoke 或独立加载不符。不得自动安装、升级、重建 cache 或改参数。

是否需要 Git 提交：否。

是否需要服务器 pull：是，只在本步骤开始前 pull 一次。

完成后需要发给 Codex 的内容：从 branch/commit 开始到最终 PASS 的完整输出，以及四个 smoke 日志末尾各 40 行。

## 步骤 3：DG-only 30k 训练

本步骤目的：只运行 Y01，无 DINO、无 cache、无 mask head，使用 RU 完整延迟拓扑。

执行位置：Linux 服务器。

需要打开的目录：`/home/chenglong/Uncertain-Nerf/uncertain-nerf`

需要检查的文件：`configs/puri_gs_garden_dg_only_full30k.yaml`

需要执行的命令：

```bash
set -euo pipefail
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
ROOT=/home/chenglong/Uncertain-Nerf/uncertain-nerf
PY="$ROOT/.venv-gsplat153/bin/python"
RESULT="$ROOT/logs-puri/phase_r_causal/garden_dg_only_30k"
LOG="$ROOT/logs-puri/phase_r_causal-console/garden_dg_only_30k.log"
test ! -e "$RESULT"
mkdir -p "$ROOT/logs-puri/phase_r_causal" "$ROOT/logs-puri/phase_r_causal-console"
PURI_GSPLAT_PYTHON="$PY" bash scripts/train_puri_gs.sh \
  "$ROOT/configs/puri_gs_garden_dg_only_full30k.yaml" \
  "$ROOT/external/gsplat-v1.5.3-ru" \
  "$ROOT/data/mipnerf360/360_v2/garden" \
  "$RESULT" 6 30000 4 2>&1 | tee "$LOG"

"$PY" - <<'PY'
import json
from pathlib import Path
root = Path("/home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/phase_r_causal/garden_dg_only_30k")
contract = json.loads((root / "causal_contract.json").read_text())
metrics = json.loads((root / "train_metrics.json").read_text())
assert (contract["M"], contract["T"]) == (0, 1)
assert contract["topology_events"]["reset_steps"] == [15000, 18000]
assert metrics["step"] == 29999
assert (root / "ckpts/ckpt_29999_rank0.pt").is_file()
assert not (root / "DINO_time.json").exists()
assert not (root / "aux/dino_environment.json").exists()
print("GARDEN-DG-ONLY-30K-TRAIN-PASS")
PY
```

命令执行完成后应看到：开头 `PURI-GS-FACTORS M=0 T=1 resets=[15000, 18000]`，
训练到 step 29999，最后 `GARDEN-DG-ONLY-30K-TRAIN-PASS`。

日志位置：`logs-puri/phase_r_causal-console/garden_dg_only_30k.log`

如何判断成功：checkpoint、train metrics、合同均存在，且没有 DINO/Mask 辅助产物。

出现什么情况应立即停止：目录预先存在；日志出现 DINO、feature cache、mask head；OOM、NaN、
非零退出；reset 不为 `[15000,18000]`。不得开始 Mask-only。

是否需要 Git 提交：否。

是否需要服务器 pull：否。

完成后需要发给 Codex 的内容：训练日志开头40行、末尾80行和 PASS 检查输出。

## 步骤 4：DG-only 独立评测（只运行一次）

本步骤目的：对 Y01 的固定 checkpoint 做一次 standard splats-only 独立评测。

执行位置：Linux 服务器。

需要打开的目录：`/home/chenglong/Uncertain-Nerf/uncertain-nerf`

需要检查的文件：`logs-puri/phase_r_causal/garden_dg_only_30k/ckpts/ckpt_29999_rank0.pt`

需要执行的命令：

```bash
set -euo pipefail
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
ROOT=/home/chenglong/Uncertain-Nerf/uncertain-nerf
PY="$ROOT/.venv-gsplat153/bin/python"
RUN="$ROOT/logs-puri/phase_r_causal/garden_dg_only_30k"
EVAL="$RUN/independent_eval"
LOG="$ROOT/logs-puri/phase_r_causal-console/garden_dg_only_eval.log"
test ! -e "$EVAL"
"$PY" run_puri_gs.py \
  --config "$ROOT/configs/puri_gs_garden_dg_only_full30k.yaml" \
  --gsplat-dir "$ROOT/external/gsplat-v1.5.3-ru" \
  --data-dir "$ROOT/data/mipnerf360/360_v2/garden" \
  --result-dir "$EVAL" --gpu 6 --data-factor 4 \
  --checkpoint "$RUN/ckpts/ckpt_29999_rank0.pt" \
  2>&1 | tee "$LOG"

"$PY" - <<'PY'
import csv, json
from pathlib import Path
root = Path("/home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/phase_r_causal/garden_dg_only_30k/independent_eval")
rows = list(csv.DictReader((root / "per_image_metrics.csv").open()))
validation = json.loads((root / "ru_validation.json").read_text())
assert len(rows) == 24 and len({row["image_name"] for row in rows}) == 24
assert validation["standard_checkpoint_load_pass"] is True
assert validation["evaluation_imported_dino"] is False
assert validation["evaluation_loaded_mask_head"] is False
assert validation["evaluation_rasterization_count_ratio"] == 1.0
print(json.loads((root / "test_metrics.json").read_text()))
print("GARDEN-DG-ONLY-EVAL-PASS")
PY
```

命令执行完成后应看到：24/24 评测完成、PSNR/SSIM/LPIPS、`GARDEN-DG-ONLY-EVAL-PASS`。

日志位置：`logs-puri/phase_r_causal-console/garden_dg_only_eval.log`

如何判断成功：CSV 恰好24个唯一名称，四项 standard-path 验证严格满足。

出现什么情况应立即停止：评测目录已存在；图名/数量不符；加载 DINO/head；raster ratio不为1；
任何指标非有限。不得开始 Mask-only。

是否需要 Git 提交：否。

是否需要服务器 pull：否。

完成后需要发给 Codex 的内容：完整检查输出和评测日志末尾80行。

## 步骤 5：Mask-only 30k 训练

本步骤目的：只运行 Y10，保持正式 RU Mask 的全部公式与 DINO/cache，但使用实际 B1 拓扑。

执行位置：Linux 服务器。

需要打开的目录：`/home/chenglong/Uncertain-Nerf/uncertain-nerf`

需要检查的文件：Mask-only配置、DINO权重、Garden feature cache。

需要执行的命令：

```bash
set -euo pipefail
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
ROOT=/home/chenglong/Uncertain-Nerf/uncertain-nerf
PY="$ROOT/.venv-gsplat153/bin/python"
RESULT="$ROOT/logs-puri/phase_r_causal/garden_mask_only_30k"
LOG="$ROOT/logs-puri/phase_r_causal-console/garden_mask_only_30k.log"
test ! -e "$RESULT"
"$PY" run_puri_gs.py \
  --config "$ROOT/configs/puri_gs_garden_mask_only_full30k.yaml" \
  --gsplat-dir "$ROOT/external/gsplat-v1.5.3-ru" \
  --data-dir "$ROOT/data/mipnerf360/360_v2/garden" \
  --result-dir "$RESULT" --gpu 6 --max-steps 30000 --data-factor 4 \
  --dino-repo-dir "$ROOT/external/dinov2" \
  --dino-weight-path "$ROOT/data/PURI-GS-assets/dinov2/dinov2_vits14_reg4_pretrain.pth" \
  --feature-cache-dir "$ROOT/data/PURI-GS-derived/semantic_features/garden" \
  2>&1 | tee "$LOG"

"$PY" - <<'PY'
import json
from pathlib import Path
root = Path("/home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/phase_r_causal/garden_mask_only_30k")
contract = json.loads((root / "causal_contract.json").read_text())
timing = json.loads((root / "DINO_time.json").read_text())
schedule = json.loads((root / "aux/training_schedule.json").read_text())
validation = json.loads((root / "aux/ru_training_validation.json").read_text())
assert (contract["M"], contract["T"]) == (1, 0)
assert contract["topology_events"]["reset_steps"] == []
assert contract["topology_events"]["mask_pause_segments"] == []
assert timing["mask_update_count"] == 30000
assert timing["mask_pause_count"] == 0
assert schedule["topology_events"] == contract["topology_events"]
assert validation["gradient_isolation_pass"] is True
assert validation["dino_trainable_parameter_count"] == 0
assert (root / "ckpts/ckpt_29999_rank0.pt").is_file()
print("GARDEN-MASK-ONLY-30K-TRAIN-PASS")
PY
```

命令执行完成后应看到：开头 `PURI-GS-FACTORS M=1 T=0 resets=[]`；DINO trainable=0；
训练到29999；最后 `GARDEN-MASK-ONLY-30K-TRAIN-PASS`。

日志位置：`logs-puri/phase_r_causal-console/garden_mask_only_30k.log`

如何判断成功：标准 checkpoint、DINO_time、aux、曲线均存在；梯度隔离通过；实际reset=[]；
head更新30000次、暂停0次。

出现什么情况应立即停止：目录已存在；M/T或reset不符；DINO SHA/commit变化；DINO可训练；
gradient isolation失败；pause非0；OOM、NaN、非零退出。不得改参数或自动重跑。

是否需要 Git 提交：否。

是否需要服务器 pull：否。

完成后需要发给 Codex 的内容：训练日志开头50行、末尾100行、PASS检查输出和
`DINO_time.json`、`training_schedule.json` 内容。

## 步骤 6：Mask-only 独立评测（只运行一次）

本步骤目的：对 Y10 checkpoint 做一次不加载DINO/head/cache的 standard splats-only评测。

执行位置：Linux 服务器。

需要打开的目录：`/home/chenglong/Uncertain-Nerf/uncertain-nerf`

需要检查的文件：`logs-puri/phase_r_causal/garden_mask_only_30k/ckpts/ckpt_29999_rank0.pt`

需要执行的命令：

```bash
set -euo pipefail
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
ROOT=/home/chenglong/Uncertain-Nerf/uncertain-nerf
PY="$ROOT/.venv-gsplat153/bin/python"
RUN="$ROOT/logs-puri/phase_r_causal/garden_mask_only_30k"
EVAL="$RUN/independent_eval"
LOG="$ROOT/logs-puri/phase_r_causal-console/garden_mask_only_eval.log"
test ! -e "$EVAL"
"$PY" run_puri_gs.py \
  --config "$ROOT/configs/puri_gs_garden_mask_only_full30k.yaml" \
  --gsplat-dir "$ROOT/external/gsplat-v1.5.3-ru" \
  --data-dir "$ROOT/data/mipnerf360/360_v2/garden" \
  --result-dir "$EVAL" --gpu 6 --data-factor 4 \
  --checkpoint "$RUN/ckpts/ckpt_29999_rank0.pt" \
  2>&1 | tee "$LOG"

"$PY" - <<'PY'
import csv, json
from pathlib import Path
root = Path("/home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/phase_r_causal/garden_mask_only_30k/independent_eval")
rows = list(csv.DictReader((root / "per_image_metrics.csv").open()))
validation = json.loads((root / "ru_validation.json").read_text())
assert len(rows) == 24 and len({row["image_name"] for row in rows}) == 24
assert validation["standard_checkpoint_load_pass"] is True
assert validation["evaluation_imported_dino"] is False
assert validation["evaluation_loaded_mask_head"] is False
assert validation["evaluation_rasterization_count_ratio"] == 1.0
print(json.loads((root / "test_metrics.json").read_text()))
print("GARDEN-MASK-ONLY-EVAL-PASS")
PY
```

命令执行完成后应看到：24/24、三项质量指标、`GARDEN-MASK-ONLY-EVAL-PASS`。

日志位置：`logs-puri/phase_r_causal-console/garden_mask_only_eval.log`

如何判断成功：24图名称与split一致；评测未加载DINO/head/cache；单次rasterization。

出现什么情况应立即停止：评测目录已存在；数量/图名/standard path/指标任一不符。

是否需要 Git 提交：否。

是否需要服务器 pull：否。

完成后需要发给 Codex 的内容：完整检查输出和日志末尾80行。

## 步骤 7：四象限汇总与强制停止

本步骤目的：验证四组证据、计算固定条件效应/交互项/paired bootstrap，并输出唯一归因标签。

执行位置：Linux 服务器。

需要打开的目录：`/home/chenglong/Uncertain-Nerf/uncertain-nerf`

需要检查的文件：四个run及各自 `independent_eval`；两个目标报告必须不存在。

需要执行的命令：

```bash
set -euo pipefail
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
ROOT=/home/chenglong/Uncertain-Nerf/uncertain-nerf
PY="$ROOT/.venv-gsplat153/bin/python"
test ! -e "$ROOT/reports/PHASE_R_GARDEN_CAUSAL_2X2.md"
test ! -e "$ROOT/reports/phase_r_garden_causal_2x2.json"
"$PY" tools/summarize_puri_gs_garden_causal_2x2.py \
  --b1 "$ROOT/logs-puri/ru-generalization-rerun-9e292309/garden_b1_30k" \
  --dg-only "$ROOT/logs-puri/phase_r_causal/garden_dg_only_30k" \
  --mask-only "$ROOT/logs-puri/phase_r_causal/garden_mask_only_30k" \
  --ru "$ROOT/logs-puri/ru-generalization-rerun-9e292309/garden_ru_30k" \
  --output-dir "$ROOT/reports" \
  2>&1 | tee "$ROOT/logs-puri/phase_r_causal-console/garden_causal_summary.log"
```

命令执行完成后应看到：`GARDEN-CAUSAL-2X2-SUMMARY-PASS`，后面恰有一个允许的归因标签，
并打印 Markdown 与 JSON 的绝对路径。

日志位置：`logs-puri/phase_r_causal-console/garden_causal_summary.log`；最终小报告为
`reports/PHASE_R_GARDEN_CAUSAL_2X2.md` 与 `reports/phase_r_garden_causal_2x2.json`。

如何判断成功：四组为同一161/24 split和同序图名；checkpoint仅`step/splats`；标准评测四项
通过；Mask update/pause与实际reset一致；质量、规模、时间、训练/推理显存、效率、五类
效应、10000次bootstrap、最差图和6张代表图索引齐全；唯一标签只能是：

```text
GARDEN_MASK_DOMINANT
GARDEN_TOPOLOGY_DOMINANT
GARDEN_NEGATIVE_INTERACTION
GARDEN_BOTH_CONTRIBUTE
GARDEN_CAUSAL_INCONCLUSIVE
```

出现什么情况应立即停止：报告已存在；任一证据缺失/不可比；工具报错；出现多个归因标签。
不得补跑大量实验、改门禁或实现新算法。

是否需要 Git 提交：报告经 Codex复核后才进入Git里程碑2，用图形化Git单独提交一次，说明为
`document Garden 2x2 causal ablation results`。

是否需要服务器 pull：否。

完成后需要发给 Codex 的内容：完整summary日志以及两个小报告。随后强制停止，等待用户与
ChatGPT给出下一份RU新版本指令。
