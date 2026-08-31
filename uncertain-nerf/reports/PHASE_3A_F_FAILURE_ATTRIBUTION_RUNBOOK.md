# PURI-GS Phase 3A-F：CVTR 零训练失败归因运行手册

本手册只执行 frozen Room 合成失败归因。禁止真实场景 mask、训练、backward、optimizer、
continuation 和参数扫描。原 `analysis/cvtr_synthetic`、parser-exact v2、B1 checkpoint
均只读。

## 1. 本地代码自检

本步骤目的：验证 trace、统计、契约审计和 CPU 人工 dry-run。

执行位置：Windows 本地的 WSL。

需要打开的目录：`E:\6-Project\1-UncertainNerf\uncertain-nerf`

需要检查的文件：

```text
puri_gs/cvtr.py
tools/analyze_cvtr_failure_attribution.py
tests/test_cvtr_stage_trace.py
tests/test_cvtr_failure_attribution.py
tests/test_cvtr_contract_audit.py
```

需要执行的命令：

```powershell
cd 'E:\6-Project\1-UncertainNerf\uncertain-nerf'

wsl bash -lc 'cd /mnt/e/6-Project/1-UncertainNerf/uncertain-nerf && .venv-v6/bin/python -m compileall -q puri_gs tools tests && .venv-v6/bin/python -m pytest -q'

wsl bash -lc 'cd /mnt/e/6-Project/1-UncertainNerf/uncertain-nerf && PYTHONPATH=. .venv-v6/bin/python tools/analyze_cvtr_failure_attribution.py --self-test-cpu'
```

预期输出：全量测试全部通过，本机不兼容的单 Gaussian 项最多有 1 个 skip；CPU self-test
显示 `status=PASS`、`optimizer_step_count=0`、`backward_call_count=0`。

日志位置：本步骤使用终端输出。

如何判断成功：pytest 无 failed/error，self-test 为 PASS。

出现什么情况应停止：语法/import/测试失败，或 self-test 出现非零训练计数。

是否需要 Git 提交：全部通过后达到本阶段唯一代码里程碑。

是否需要服务器 pull：本地 push 前不需要。

## 2. 图形化 Git 里程碑

本步骤目的：只传输失败归因源代码、测试和必要文档。

执行位置：Windows 本地 Codex Git 图形界面。

需要打开的目录：`E:\6-Project\1-UncertainNerf\uncertain-nerf`

需要检查的文件：本手册步骤 1 列出的五个代码/测试文件以及本运行手册。

需要执行的命令：无；使用图形化 Git 提交和 push。

建议提交说明：

```text
add zero-training CVTR failure-attribution diagnostics
```

预期输出：提交和 push 成功，分支仍为 `dev`。

日志位置：Git 图形界面的提交历史。

如何判断成功：提交不包含数据、analysis、checkpoint、日志、PNG 或证据包。

出现什么情况应停止：发现派生数据、原数据、checkpoint 或服务器结果进入 staged files。

是否需要 Git 提交：是，本阶段只进行这一次明确代码提交。

是否需要服务器 pull：push 成功后需要一次。

## 3. 服务器同步与环境检查

本步骤目的：同步唯一代码里程碑并验证固定环境；不修改环境。

执行位置：Linux 服务器。

需要打开的目录：`/home/chenglong/Uncertain-Nerf/uncertain-nerf`

需要检查的文件：新增工具和三组测试。

需要执行的命令：

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
set +e

git pull --ff-only

ROOT="$PWD"
PY="$ROOT/.venv-gsplat153/bin/python"

echo "branch=$(git branch --show-current)"
echo "commit=$(git rev-parse HEAD)"
git status --short

test -s "$ROOT/tools/analyze_cvtr_failure_attribution.py" \
  && echo FOUND_ATTRIBUTION_TOOL \
  || echo STOP_MISSING_ATTRIBUTION_TOOL

"$PY" - <<'PY'
import numpy
import PIL
import torch
import gsplat

print("torch=", torch.__version__)
print("cuda=", torch.version.cuda)
print("PIL=", PIL.__version__)
print("gsplat=", gsplat.__file__)
PY
```

预期输出：`branch=dev`、工具存在，固定 Python 可导入 NumPy、PIL、PyTorch 和 gsplat。

日志位置：终端输出。

如何判断成功：pull 为 fast-forward 或 already up to date，环境导入全部成功。

出现什么情况应停止：分支不是 dev、pull 冲突、工具缺失或依赖 import 失败。不得安装或升级包。

是否需要 Git 提交：否。

是否需要服务器 pull：本步骤执行一次，后续不再 pull。

## 4. 服务器新增测试与 CUDA 深度硬门

本步骤目的：先验证新增 CPU 契约，再执行 L20 single-Gaussian expected camera-z 数值门。

执行位置：Linux 服务器。

需要打开的目录：`/home/chenglong/Uncertain-Nerf/uncertain-nerf`

需要检查的文件：三组新增测试和 `tests/test_cvtr_geometry.py`。

需要执行的命令：

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
set +e

ROOT="$PWD"
PY="$ROOT/.venv-gsplat153/bin/python"
GPU=6
LOG="$ROOT/tmp/phase3a_f_tests.log"

PYTHONPATH="$ROOT" "$PY" -m pytest -q \
  tests/test_cvtr_stage_trace.py \
  tests/test_cvtr_failure_attribution.py \
  tests/test_cvtr_contract_audit.py \
  2>&1 | tee "$LOG"
CPU_TEST_EXIT_CODE=${PIPESTATUS[0]}

CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT" "$PY" -m pytest -q \
  tests/test_cvtr_geometry.py::test_single_gaussian_rgb_ed_is_expected_camera_z \
  2>&1 | tee -a "$LOG"
CUDA_TEST_EXIT_CODE=${PIPESTATUS[0]}

echo "CPU_TEST_EXIT_CODE=$CPU_TEST_EXIT_CODE"
echo "CUDA_TEST_EXIT_CODE=$CUDA_TEST_EXIT_CODE"
```

预期输出：新增测试 `36 passed`，CUDA 项严格 `1 passed`，两个退出码均为 0。

日志位置：`tmp/phase3a_f_tests.log`

如何判断成功：没有 skip/failed/error，CUDA 项为 1 passed。

出现什么情况应停止：任何测试失败、CUDA 项 skipped、GPU 非 L20 或架构不兼容。

是否需要 Git 提交：否。

是否需要服务器 pull：否。

## 5. 冻结输入与唯一输出预检查

本步骤目的：确认原失败证据完整、GPU 空闲、输出和报告不存在。

执行位置：Linux 服务器。

需要打开的目录：`/home/chenglong/Uncertain-Nerf/uncertain-nerf`

需要检查的文件：固定 Room 数据、B1 checkpoint、v2 manifest、原 metrics/cache。

需要执行的命令：

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
set +e

ROOT="$PWD"
GPU=6
COMMIT_SHORT="$(git rev-parse --short=7 HEAD)"
OUTPUT="$ROOT/analysis/cvtr_failure_attribution_$COMMIT_SHORT"
ARCHIVE="$ROOT/analysis/puri_gs_cvtr_failure_attribution_$COMMIT_SHORT.tar.gz"
CHECKSUM="$ROOT/analysis/puri_gs_cvtr_failure_attribution_$COMMIT_SHORT.sha256"

for required in \
  "$ROOT/data/mipnerf360/360_v2/room/sparse/0/cameras.bin" \
  "$ROOT/tmp/room_cvtr_synthetic_v2/manifest.json" \
  "$ROOT/analysis/cvtr_synthetic/metrics.json" \
  "$ROOT/analysis/cvtr_synthetic/manifest.json" \
  "$ROOT/analysis/cvtr_synthetic/render_cache_manifest.json" \
  "$ROOT/analysis/cvtr_synthetic/neighbors.json" \
  "$ROOT/logs-puri/room_b1_short10k_seed42_128535b/ckpts/ckpt_9999_rank0.pt" \
  "$ROOT/configs/puri_gs_cvtr.yaml"
do
  test -s "$required" && echo "FOUND $required" || echo "STOP_MISSING $required"
done

SOURCE_IMAGE_COUNT=$(find "$ROOT/data/mipnerf360/360_v2/room/images" -type f | wc -l)
FACTOR_IMAGE_COUNT=$(find "$ROOT/data/mipnerf360/360_v2/room/images_4_png" -type f | wc -l)
echo "SOURCE_IMAGE_COUNT=$SOURCE_IMAGE_COUNT"
echo "FACTOR_IMAGE_COUNT=$FACTOR_IMAGE_COUNT"

for forbidden in \
  "$OUTPUT" \
  "$ARCHIVE" \
  "$CHECKSUM" \
  "$ROOT/reports/PHASE_3A_CVTR_FAILURE_ATTRIBUTION.md" \
  "$ROOT/reports/phase_3a_cvtr_failure_attribution.json"
do
  test -e "$forbidden" && echo "STOP_OUTPUT_EXISTS $forbidden" || echo "OUTPUT_AVAILABLE $forbidden"
done

echo "=== GPU 6 processes ==="
nvidia-smi -i "$GPU" \
  --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader

echo "=== existing failure metrics ==="
"$ROOT/.venv-gsplat153/bin/python" -m json.tool \
  "$ROOT/analysis/cvtr_synthetic/metrics.json"
```

预期输出：所有 required 均 FOUND；`SOURCE_IMAGE_COUNT` 和 `FACTOR_IMAGE_COUNT` 均为
311；所有新输出均 OUTPUT_AVAILABLE；GPU 6 没有计算进程；原 metrics 为
`MASK_VALIDATION_FAIL`。

日志位置：终端输出。

如何判断成功：输入完整且任何新输出都不存在。

出现什么情况应停止：缺输入、输出已存在、GPU 忙、原 gate 不是失败结果。不得删除旧输出重试。

是否需要 Git 提交：否。

是否需要服务器 pull：否。

## 6. 零训练失败归因正式运行

本步骤目的：只读复现 final mask，执行 F0–F6 分阶段归因并生成证据包。

执行位置：Linux 服务器 GPU 6。

需要打开的目录：`/home/chenglong/Uncertain-Nerf/uncertain-nerf`

需要检查的文件：步骤 5 的全部冻结输入。

需要执行的命令：

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
set +e

ROOT="$PWD"
PY="$ROOT/.venv-gsplat153/bin/python"
GS="$ROOT/external/gsplat-v1.5.3"
GPU=6
ROOM_DATA="$ROOT/data/mipnerf360/360_v2/room"
ROOM_CKPT="$ROOT/logs-puri/room_b1_short10k_seed42_128535b/ckpts/ckpt_9999_rank0.pt"
DERIVED="$ROOT/tmp/room_cvtr_synthetic_v2"
EXISTING="$ROOT/analysis/cvtr_synthetic"
COMMIT_SHORT="$(git rev-parse --short=7 HEAD)"
OUTPUT="$ROOT/analysis/cvtr_failure_attribution_$COMMIT_SHORT"
LOG="$ROOT/tmp/phase3a_f_attribution_$COMMIT_SHORT.log"

CUDA_VISIBLE_DEVICES="$GPU" \
PYTHONPATH="$ROOT" \
PYTHONUNBUFFERED=1 \
"$PY" tools/analyze_cvtr_failure_attribution.py \
  --data-dir "$ROOM_DATA" \
  --gsplat-dir "$GS" \
  --derived-dir "$DERIVED" \
  --checkpoint "$ROOM_CKPT" \
  --config "$ROOT/configs/puri_gs_cvtr.yaml" \
  --existing-output "$EXISTING" \
  --output-dir "$OUTPUT" \
  --device cuda \
  --save-stage-arrays \
  --save-visualizations \
  2>&1 | tee "$LOG"

ATTRIBUTION_EXIT_CODE=${PIPESTATUS[0]}

echo "ATTRIBUTION_EXIT_CODE=$ATTRIBUTION_EXIT_CODE"
echo "OUTPUT=$OUTPUT"
echo "LOG=$LOG"
```

预期输出：逐项审计与分析完成，最终打印 `PASS_EXECUTION`、
`IMPLEMENTATION_CONTRACT_PASS` 和证据包 SHA；退出码为 0。

日志位置：`tmp/phase3a_f_attribution_<commit>.log`

如何判断成功：工具退出 0，contract 和 provenance 均 PASS，训练计数均为 0。

出现什么情况应停止：任何附件规定的 contract fail、复现 fail、非零训练计数或
`ATTRIBUTION_EXIT_CODE` 非 0。不得修参数或启动 continuation。

是否需要 Git 提交：否，服务器证据输出不提交。

是否需要服务器 pull：否。

## 7. 结果与证据包核验

本步骤目的：读取事实结果、核对零训练计数和压缩包 SHA。

执行位置：Linux 服务器。

需要打开的目录：`/home/chenglong/Uncertain-Nerf/uncertain-nerf`

需要检查的文件：provenance、contract、两份报告和 SHA sidecar。

需要执行的命令：

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
set +e

ROOT="$PWD"
PY="$ROOT/.venv-gsplat153/bin/python"
COMMIT_SHORT="$(git rev-parse --short=7 HEAD)"
OUTPUT="$ROOT/analysis/cvtr_failure_attribution_$COMMIT_SHORT"
CHECKSUM="$ROOT/analysis/puri_gs_cvtr_failure_attribution_$COMMIT_SHORT.sha256"

echo "=== provenance ==="
"$PY" -m json.tool "$OUTPUT/provenance.json"

echo "=== contract ==="
"$PY" -m json.tool "$OUTPUT/contract/contract_audit.json"

echo "=== final factual report ==="
"$PY" -m json.tool "$ROOT/reports/phase_3a_cvtr_failure_attribution.json"

echo "=== evidence SHA ==="
cd "$ROOT/analysis"
sha256sum -c "$(basename "$CHECKSUM")"
```

预期输出：`PASS_EXECUTION`、`IMPLEMENTATION_CONTRACT_PASS`，四个运行计数均为 0，
`checkpoint_unchanged=true`，SHA 检查为 OK。

日志位置：沿用步骤 6 日志；JSON/CSV/PNG/NPZ 位于唯一 output 目录。

如何判断成功：所有契约 PASS、final metrics 与原 metrics 差值不超过 `1e-5`、SHA OK。

出现什么情况应停止：任何 POTENTIAL_IMPLEMENTATION_ERROR、复现失败、计数非零或 SHA 失败。

是否需要 Git 提交：否。

是否需要服务器 pull：否。

## 8. 下载证据包

本步骤目的：把报告和证据包下载到 Windows，交给 ChatGPT 做后续研究决策。

执行位置：Windows PowerShell。

需要打开的目录：`E:\6-Project\1-UncertainNerf\uncertain-nerf`

需要检查的文件：服务器 tar.gz、sha256 和两份报告。

需要执行的命令：

```powershell
cd 'E:\6-Project\1-UncertainNerf\uncertain-nerf'

$remote = 'chenglong@172.16.55.2'
$remoteRoot = '/home/chenglong/Uncertain-Nerf/uncertain-nerf'
$commitShort = ssh $remote "cd '$remoteRoot' && git rev-parse --short=7 HEAD"
$destination = 'E:\7-DataSet\PURI-GS-derived\cvtr_failure_attribution'

New-Item -ItemType Directory -Force -Path $destination | Out-Null

scp "${remote}:${remoteRoot}/analysis/puri_gs_cvtr_failure_attribution_${commitShort}.tar.gz" $destination
scp "${remote}:${remoteRoot}/analysis/puri_gs_cvtr_failure_attribution_${commitShort}.sha256" $destination
scp "${remote}:${remoteRoot}/reports/PHASE_3A_CVTR_FAILURE_ATTRIBUTION.md" $destination
scp "${remote}:${remoteRoot}/reports/phase_3a_cvtr_failure_attribution.json" $destination

Get-ChildItem $destination
```

预期输出：四个文件下载成功。

日志位置：PowerShell 输出。

如何判断成功：本地存在 tar.gz、sha256、Markdown 和 JSON 报告。

出现什么情况应停止：远端文件缺失或 scp 失败；不得用旧证据包替代。

是否需要 Git 提交：否，证据包禁止提交。

是否需要服务器 pull：否。

完成后停止。不得运行真实场景 mask、continuation 或任何训练。
