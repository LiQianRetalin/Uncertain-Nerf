# PURI-GS-ST Phase 4A 服务器运行手册

本手册只运行冻结 B1 静态地图与每视图独立瞬态 Gaussian 层的 16 帧合成门。
不得修改固定配置，不得覆盖已有输出，不得在本阶段启动联合训练。

## 1. 环境与固定资产检查

本步骤目的：确认服务器仍是固定 `dev`、L20、Python、gsplat、Room、B1 checkpoint
和 parser-exact v2 资产。

执行位置：Linux 服务器。

需要打开的目录：`/home/chenglong/Uncertain-Nerf/uncertain-nerf`

需要检查的文件：

```text
configs/puri_gs_static_transient_gate.yaml
logs-puri/room_b1_short10k_seed42_128535b/ckpts/ckpt_9999_rank0.pt
tmp/room_cvtr_synthetic_v2/manifest.json
external/gsplat-v1.5.3/gsplat/rendering.py
```

需要执行的命令：

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf

test "$(git branch --show-current)" = "dev"
git rev-parse HEAD
git status --short
nvidia-smi -i 6 --query-gpu=index,name,memory.total --format=csv,noheader
/home/chenglong/Uncertain-Nerf/uncertain-nerf/.venv-gsplat153/bin/python -c "import sys, torch, gsplat; print(sys.executable); print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(gsplat.__version__, gsplat.__file__)"
sha256sum /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/room_b1_short10k_seed42_128535b/ckpts/ckpt_9999_rank0.pt
sha256sum /home/chenglong/Uncertain-Nerf/uncertain-nerf/tmp/room_cvtr_synthetic_v2/manifest.json
```

命令完成后应看到：分支为 `dev`；GPU 为 `NVIDIA L20`；checkpoint SHA 为
`a1edd558de7434381a719b3b0a9629b53fb268fa2f1eb72235609448e3589d7d`；manifest
SHA 为 `14461238a297fcadad6395c2486a4563e334f0ae8f64a649e07be3a70912c7a9`。

日志位置：本步骤只打印到终端。

如何判断成功：上述固定值全部一致，`git status --short` 没有未知代码改动。

出现什么情况应停止：分支、GPU、SHA、Python、PyTorch、CUDA 或 gsplat 版本不一致。

是否需要 Git 提交：否。

是否需要服务器 pull：本手册只应在已完成图形化 push 并在服务器 pull 后执行。

## 2. 单元测试

本步骤目的：运行 Phase 4A 定向测试与仓库回归测试。

执行位置：Linux 服务器。

需要打开的目录：`/home/chenglong/Uncertain-Nerf/uncertain-nerf`

需要检查的文件：`tests/test_transient_*.py`、`tests/test_static_transient_gate.py`

需要执行的命令：

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
set -o pipefail

PY=/home/chenglong/Uncertain-Nerf/uncertain-nerf/.venv-gsplat153/bin/python
ROOT=/home/chenglong/Uncertain-Nerf/uncertain-nerf
LOG=/home/chenglong/Uncertain-Nerf/uncertain-nerf/tmp/phase4a_unit_tests.log

PYTHONPATH="$ROOT" "$PY" -m pytest -q \
  tests/test_transient_layer_geometry.py \
  tests/test_transient_compositing.py \
  tests/test_transient_freeze.py \
  tests/test_static_transient_gate.py \
  2>&1 | tee "$LOG"
echo "PHASE4A_UNIT_EXIT_CODE=${PIPESTATUS[0]}"
```

命令完成后应看到：全部定向测试通过且 `PHASE4A_UNIT_EXIT_CODE=0`。

日志位置：`/home/chenglong/Uncertain-Nerf/uncertain-nerf/tmp/phase4a_unit_tests.log`

如何判断成功：无失败、无错误。

出现什么情况应停止：任一测试失败或 import 错误。

是否需要 Git 提交：否。

是否需要服务器 pull：否。

## 3. CUDA 单帧 smoke

本步骤目的：在 GPU 6 上验证真实 gsplat rasterization、梯度边界、网格覆盖、clean-null、
patch-on 和一张真实 patched 派生帧的两步优化。

执行位置：Linux 服务器。

需要打开的目录：`/home/chenglong/Uncertain-Nerf/uncertain-nerf`

需要检查的文件：`tools/fit_static_transient_gate.py`

需要执行的命令：

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
set -o pipefail

CUDA_VISIBLE_DEVICES=6 PYTHONPATH=/home/chenglong/Uncertain-Nerf/uncertain-nerf \
  /home/chenglong/Uncertain-Nerf/uncertain-nerf/.venv-gsplat153/bin/python \
  tools/fit_static_transient_gate.py \
  --config /home/chenglong/Uncertain-Nerf/uncertain-nerf/configs/puri_gs_static_transient_gate.yaml \
  --gsplat-dir /home/chenglong/Uncertain-Nerf/uncertain-nerf/external/gsplat-v1.5.3 \
  --data-dir /home/chenglong/Uncertain-Nerf/uncertain-nerf/data/mipnerf360/360_v2/room \
  --derived-dir /home/chenglong/Uncertain-Nerf/uncertain-nerf/tmp/room_cvtr_synthetic_v2 \
  --checkpoint /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/room_b1_short10k_seed42_128535b/ckpts/ckpt_9999_rank0.pt \
  --device cuda:0 --smoke \
  2>&1 | tee /home/chenglong/Uncertain-Nerf/uncertain-nerf/tmp/phase4a_cuda_smoke.log
echo "PHASE4A_SMOKE_EXIT_CODE=${PIPESTATUS[0]}"
```

命令完成后应看到：顶层 `status` 为 `PASS`，并且
`PHASE4A_SMOKE_EXIT_CODE=0`。

日志位置：`/home/chenglong/Uncertain-Nerf/uncertain-nerf/tmp/phase4a_cuda_smoke.log`

如何判断成功：投影、覆盖、合成、静态冻结、瞬态梯度、clean-null、patch-on 和实际帧
smoke 全部通过；checkpoint unchanged 为 true。

出现什么情况应停止：任一 contract 为 false、loss 非有限、CUDA 错误、静态状态或
checkpoint 改变。

是否需要 Git 提交：否。

是否需要服务器 pull：否。

## 4. Input contract 只读审计

本步骤目的：再次只读核验 Phase 3 固定资产、33 文件派生树、16 帧 parser 映射、相机矩阵、
网格投影和 rasterizer 语义；不创建正式输出。

执行位置：Linux 服务器。

需要打开的目录：`/home/chenglong/Uncertain-Nerf/uncertain-nerf`

需要检查的文件：同第 1 步。

需要执行的命令：

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
set -o pipefail

CUDA_VISIBLE_DEVICES=6 PYTHONPATH=/home/chenglong/Uncertain-Nerf/uncertain-nerf \
  /home/chenglong/Uncertain-Nerf/uncertain-nerf/.venv-gsplat153/bin/python \
  tools/fit_static_transient_gate.py \
  --config /home/chenglong/Uncertain-Nerf/uncertain-nerf/configs/puri_gs_static_transient_gate.yaml \
  --gsplat-dir /home/chenglong/Uncertain-Nerf/uncertain-nerf/external/gsplat-v1.5.3 \
  --data-dir /home/chenglong/Uncertain-Nerf/uncertain-nerf/data/mipnerf360/360_v2/room \
  --derived-dir /home/chenglong/Uncertain-Nerf/uncertain-nerf/tmp/room_cvtr_synthetic_v2 \
  --checkpoint /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/room_b1_short10k_seed42_128535b/ckpts/ckpt_9999_rank0.pt \
  --device cuda:0 --preflight \
  2>&1 | tee /home/chenglong/Uncertain-Nerf/uncertain-nerf/tmp/phase4a_input_contract.log
echo "PHASE4A_PREFLIGHT_EXIT_CODE=${PIPESTATUS[0]}"
```

命令完成后应看到：顶层 `status` 为 `PASS`，16 个 frame mapping 全部存在，
`PHASE4A_PREFLIGHT_EXIT_CODE=0`。

日志位置：`/home/chenglong/Uncertain-Nerf/uncertain-nerf/tmp/phase4a_input_contract.log`

如何判断成功：checkpoint、manifest、派生树 SHA 和相机/投影审计全部通过。

出现什么情况应停止：任一 SHA、文件数、帧顺序、图像内容、相机矩阵或投影不一致。

是否需要 Git 提交：否。

是否需要服务器 pull：否。

## 5. 正式 16 帧 gate

本步骤目的：每帧独立执行固定 500 步瞬态拟合并生成唯一 commit-scoped 输出。

执行位置：Linux 服务器。

需要打开的目录：`/home/chenglong/Uncertain-Nerf/uncertain-nerf`

需要检查的文件：运行前确认不存在
`analysis/static_transient_gate_$(git rev-parse --short HEAD)`。

需要执行的命令：

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
set -o pipefail

CUDA_VISIBLE_DEVICES=6 PYTHONPATH=/home/chenglong/Uncertain-Nerf/uncertain-nerf \
  /home/chenglong/Uncertain-Nerf/uncertain-nerf/.venv-gsplat153/bin/python \
  tools/fit_static_transient_gate.py \
  --config /home/chenglong/Uncertain-Nerf/uncertain-nerf/configs/puri_gs_static_transient_gate.yaml \
  --gsplat-dir /home/chenglong/Uncertain-Nerf/uncertain-nerf/external/gsplat-v1.5.3 \
  --data-dir /home/chenglong/Uncertain-Nerf/uncertain-nerf/data/mipnerf360/360_v2/room \
  --derived-dir /home/chenglong/Uncertain-Nerf/uncertain-nerf/tmp/room_cvtr_synthetic_v2 \
  --checkpoint /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/room_b1_short10k_seed42_128535b/ckpts/ckpt_9999_rank0.pt \
  --device cuda:0 \
  2>&1 | tee /home/chenglong/Uncertain-Nerf/uncertain-nerf/tmp/phase4a_formal_gate.log
echo "PHASE4A_FORMAL_EXIT_CODE=${PIPESTATUS[0]}"
```

命令完成后应看到：16 行 `PHASE4A_FRAME`、`PASS_EXECUTION`、一个 gate decision、
`PHASE4A_FORMAL_EXIT_CODE=0`。

日志位置：`/home/chenglong/Uncertain-Nerf/uncertain-nerf/tmp/phase4a_formal_gate.log`

如何判断成功：完成 16 帧并生成 `metrics/gate.json`。`ST_GATE_FAIL` 是有效研究结论，
但执行硬门失败或非零退出码不是有效结果。

出现什么情况应停止：输出目录已存在、任一执行硬门失败、非有限 loss、CUDA 错误、
checkpoint/静态状态变化。不得删掉已有正式输出后静默重跑。

是否需要 Git 提交：正式结果汇总前否。

是否需要服务器 pull：否。

## 6. 汇总、报告与证据包

本步骤目的：根据固定指标生成两份报告和不含原始数据/checkpoint/完整静态缓存的证据包。

执行位置：Linux 服务器。

需要打开的目录：`/home/chenglong/Uncertain-Nerf/uncertain-nerf`

需要检查的文件：正式输出中的 `metrics/gate.json`。

需要执行的命令：

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
set -o pipefail

PYTHONPATH=/home/chenglong/Uncertain-Nerf/uncertain-nerf \
  /home/chenglong/Uncertain-Nerf/uncertain-nerf/.venv-gsplat153/bin/python \
  tools/summarize_static_transient_gate.py \
  2>&1 | tee /home/chenglong/Uncertain-Nerf/uncertain-nerf/tmp/phase4a_summary.log
echo "PHASE4A_SUMMARY_EXIT_CODE=${PIPESTATUS[0]}"
```

命令完成后应看到：decision、两份报告路径、tar.gz、sha256 路径，以及
`PHASE4A_SUMMARY_EXIT_CODE=0`。

日志位置：`/home/chenglong/Uncertain-Nerf/uncertain-nerf/tmp/phase4a_summary.log`

如何判断成功：存在：

```text
reports/PHASE_4A_STATIC_TRANSIENT_GATE.md
reports/phase_4a_static_transient_gate.json
analysis/puri_gs_phase4a_static_transient_<commit_short>.tar.gz
analysis/puri_gs_phase4a_static_transient_<commit_short>.sha256
```

出现什么情况应停止：报告或证据包已存在、run provenance 与当前 commit 不一致、缺失
任一指标或关键可视化。

是否需要 Git 提交：先将事实结果交回 Codex/ChatGPT 审核，再决定是否形成里程碑 2。

是否需要服务器 pull：否。

## 7. 日志查看

本步骤目的：无需猜测文件位置即可查看关键状态。

执行位置：Linux 服务器。

需要打开的目录：`/home/chenglong/Uncertain-Nerf/uncertain-nerf`

需要检查的文件：Phase 4A 四个日志和最终 gate。

需要执行的命令：

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf

tail -n 80 /home/chenglong/Uncertain-Nerf/uncertain-nerf/tmp/phase4a_unit_tests.log
tail -n 120 /home/chenglong/Uncertain-Nerf/uncertain-nerf/tmp/phase4a_cuda_smoke.log
tail -n 80 /home/chenglong/Uncertain-Nerf/uncertain-nerf/tmp/phase4a_input_contract.log
tail -n 160 /home/chenglong/Uncertain-Nerf/uncertain-nerf/tmp/phase4a_formal_gate.log
tail -n 80 /home/chenglong/Uncertain-Nerf/uncertain-nerf/tmp/phase4a_summary.log
find /home/chenglong/Uncertain-Nerf/uncertain-nerf/analysis -maxdepth 3 -path '*/static_transient_gate_*/metrics/gate.json' -print -exec cat {} \;
```

命令完成后应看到：所有执行退出码为 0，正式 gate JSON 包含唯一 decision。

日志位置：上述五个 `tmp/phase4a_*.log`。

如何判断成功：执行硬门全部通过，报告与 gate decision 一致。

出现什么情况应停止：日志显示 hard gate false、checkpoint changed、NaN/Inf、traceback
或输出覆盖拒绝。

是否需要 Git 提交：仅在结果审核后决定。

是否需要服务器 pull：否。
