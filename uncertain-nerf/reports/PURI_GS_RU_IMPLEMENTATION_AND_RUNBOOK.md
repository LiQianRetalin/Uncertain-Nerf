# PURI-GS-RU 实施审计与逐步操作手册

更新时间：2026-09-01（Asia/Shanghai）

本文件只覆盖阶段 R：DINOv2 语义特征共识、静态责任头和延迟 AbsGrad 增密。阶段 U 未实现，也不得在没有用户明确确认时开始。

## 一、当前结论

代码实现、CPU 单元测试、Python 编译检查、Shell 语法检查、配置校验和 gsplat 补丁可重复应用检查已经完成。官方 DINOv2 权重、固定源码和 NVIDIA L20 环境检查均已通过；base 与 RU-superset 的 B1 双 100-step 训练及独立指标回归也已通过。Android 122 张训练图的 DINO 特征缓存、RU 100-step CUDA 训练和标准 checkpoint 独立 19 图评测均已通过。步骤 1–9 已完成，达到 Git 里程碑 1；正式 30k 仍需用户在 Git 提交、push、服务器主项目 pull 与环境复核后另行明确确认。

步骤 1–9 已完成。服务器 GPU 6 为 NVIDIA L20，显存 46068 MiB，驱动 580.65.06；PyTorch 为 2.4.0+cu121，运行时 gsplat 为 1.5.3+pt24cu121。官方 DINOv2 源码 commit 为 `7764ea0f912e53c92e82eb78a2a1631e92725fc8`。缓存的 122 个 coarse/fine payload 已逐个加载并验证形状和有限性。RU 100-step 产生 112790 个 Gaussian，DINO trainable 为 0，完成 100 次 mask 更新与 100 次梯度隔离运行时检查，标准 checkpoint 只含 `step` 和 `splats`。独立评测完成 19 张图：PSNR 17.742794、SSIM 0.663716、LPIPS 0.560165、36.1501 FPS；评测未导入 DINO、未加载 mask head，rasterization ratio 为 1.0。

## 二、首次仓库审计的 25 项事实

1. 当前分支为 `dev`，当前项目 commit 为 `0ae62b10ef4d69c35cfcc1aafdfab37f37bd98fe`。没有切换分支、没有创建 worktree、没有提交。
2. 修改中的已跟踪文件为 `puri_gs/config.py`、`run_puri_gs.py`、`scripts/train_puri_gs.sh`、`tests/test_puri_gs_configs.py`。新增文件见下一节。原始工作区在实现前是干净的；当前改动均属于本次 PURI-GS-RU 实施。
3. 本地 gsplat 源码路径为 `E:\6-Project\1-UncertainNerf\uncertain-nerf\external\gsplat-v1.5.3`，固定 commit 为 `937e29912570c372bed6747a5c9bf85fed877bae`。本地 WSL 实际运行包为 `/mnt/e/6-Project/1-UncertainNerf/uncertain-nerf/.venv-gsplat153/lib/python3.10/site-packages/gsplat/__init__.py`，版本 `1.5.3+pt24cu121`。历史服务器运行包为 `/home/chenglong/Uncertain-Nerf/uncertain-nerf/.venv-gsplat153/lib/python3.10/site-packages/gsplat/__init__.py`。
4. 原 B1 配置为 `configs/puri_gs_b1_absgrad.yaml`；本次新增正式 30k 配置 `configs/puri_gs_b1_full30k.yaml`。
5. 统一启动入口为 `run_puri_gs.py`，实际 trainer 为 gsplat 源码中的 `examples/simple_trainer.py`。没有新建第二套完整 trainer。
6. `DefaultStrategy` 在 `external/gsplat-v1.5.3/examples/simple_trainer.py` 的配置类中声明，并在 Runner 初始化 Gaussian 后构造/替换；RU 只将它包装成一个本地 Python 子类。
7. rasterization 的 `absgrad` 仍从 `self.cfg.strategy.absgrad` 传入，位置在 `external/gsplat-v1.5.3/examples/simple_trainer.py` 的 `rasterize_splats` 调用处。
8. `strategy.step_pre_backward` 在 Gaussian loss backward 前调用；RU 的 `step_post_backward` 按附件要求位于 Gaussian optimizer 前。B1 继续保持原 gsplat 顺序。
9. Android 本地原始数据为 `E:\7-DataSet\nerf数据集\nerf_robustnerf\robustnerf\android`；Room 为 `E:\7-DataSet\nerf数据集\mipnerf360\360_v2\room`。两者均未移动、重命名或写入 Git。
10. Android `images_4` 中固定为 122 张 `clutter` 训练图和 19 张 `extra` 测试图。Room `images_4` 共 311 张，沿用每 8 张取 1 张测试的 gsplat split，即 272 张训练、39 张测试。
11. 本地 `external/dinov2` 不存在；服务器已在 `/home/chenglong/Uncertain-Nerf/uncertain-nerf/external/dinov2` 准备官方源码，commit 为 `7764ea0f912e53c92e82eb78a2a1631e92725fc8`。源码作为外部依赖不进入项目 Git。
12. 首次审计时权重不存在；现已存在于 `E:\7-DataSet\PURI-GS-assets\dinov2\dinov2_vits14_reg4_pretrain.pth`。大小 88,291,785 字节，SHA-256 为 `F433177089A681826F849F194ECE3BB48F4D63FB38D32FC837E3DC7A4E5641FB`。本地只读加载确认包含 176 个 tensor，register tokens 形状为 `[1,4,384]`，patch embedding 为 `[384,3,14,14]`，全部 tensor 有限。
13. 当前没有调整 PyTorch、CUDA、gsplat 或虚拟环境，也没有新增 pip 依赖。本机 RTX 5070 Ti Laptop 是 `sm_120`，现有 PyTorch 2.4.0+cu121 只支持到 `sm_90`，因此不在本机强行升级环境或执行伪 CUDA smoke；真正 smoke 固定在 L20 上完成。
14. 新增算法与工具文件：`puri_gs/dino_features.py`、`puri_gs/semantic_mask.py`、`puri_gs/delayed_absgrad.py`、`puri_gs/ru_training.py`、`tools/cache_puri_gs_features.py`、`tools/check_puri_gs_ru_environment.py`、`tools/compare_puri_gs_b1_checkpoints.py`、`tools/summarize_puri_gs_ru.py`、两个正式配置、三个脚本、gsplat RU patch、本手册及对应测试。
15. 修改文件只有第 2 项列出的四个项目文件；没有修改 A1、CVTR、Phase 4A 或 CUDA rasterizer。
16. 完整固定配置集中在 `configs/puri_gs_ru_full30k.yaml`：30k、AbsGrad、`grow_grad2d=0.0006`、mask 500 步启用、0.25 阈值、7×7 收缩、384→16→1、DINO 16/36 两级、固定直方图、20000 步尺度切换、10000–19999 增密、15000/18000 reset、reset 后 300 步暂停 head、SSIM 0.2、seed 42、SH 3。配置校验拒绝混入 A1/CVTR 字段，并只允许 `<=100` 的 smoke 或完整 30000 步。
17. 单元测试覆盖 DINO 预处理/形状/冻结/缓存、mask 输出/阈值/7×7/参考 `.data` 正则、双向梯度隔离、增密与 reset 边界、checkpoint 兼容、独立推理不加载 DINO/head、B1 checkpoint 回归比较、配置字段、结果汇总与不覆盖保护。
18. 最终本地复测为 175 passed、1 skipped；定向 RU 测试为 24/24 passed。skip 是现有测试套件的预期跳过，不是 PURI-GS-RU 失败。
19. L20 smoke 固定流程：核对 GPU 6 确实是 NVIDIA L20；官方 DINO 单张 coarse/fine 前向；Android 特征缓存；base-patch B1 100 步；RU-superset B1 100 步；checkpoint 数值回归；RU 100 步；标准 checkpoint 独立加载和 19 张测试图评测。任何一项失败立即停。
20. 当前没有达到 Git 里程碑 1，不提交、不 push、服务器主项目不 pull。
21. 里程碑 1 通过后的建议提交说明固定为：`implement PURI-GS-RU semantic mask and delayed AbsGrad training`。
22. 只有里程碑 1 完成且用户用图形化 Git 提交并 push 后，服务器主项目才 pull；pull 后先核对 commit、准备固定 gsplat 源码并复查环境，仍不能自动开始 30k。
23. 首条服务器只读命令固定为 `nvidia-smi -i 6 --query-gpu=index,name,memory.total,driver_version --format=csv,noheader`。
24. 预期首条输出的 GPU 名称必须包含 `NVIDIA L20`，索引必须是 `6`。
25. 如果索引 6 不是 L20、CUDA 不可用、gsplat 不是 1.5.3、commit 不符、DINO SHA 不同、输出目录已存在、split 数量不符、出现 NaN/Inf 或任一验证状态不是 PASS，立即停止，不安装替代 backbone、不改阈值、不换 baseline、不删除旧结果。

## 三、实现内容与固定行为

### 1. 冻结 DINOv2

只加载官方 `dinov2_vits14_reg`，通过本地 `torch.hub` 源码和用户明确提供的本地权重离线加载。全部参数执行 `requires_grad_(False)` 和 `eval()`；训练记录参数量、可训练参数量 0、权重 SHA-256 和 DINO 源码 commit。预处理只有一套：RGB `[0,1]`、bicubic square resize、antialias、ImageNet mean/std、`x_norm_patchtokens`，输出固定为 `[384,16,16]` 与 `[384,36,36]`。

官方来源：[DINOv2 官方仓库](https://github.com/facebookresearch/dinov2)；[官方 ViT-S/14 registers 权重](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth)。

### 2. 唯一静态责任头

结构固定为 `Linear(384,16) → ReLU → Linear(16,1) → Sigmoid`，Adam 固定为学习率 `1e-3`、betas `(0.9,0.999)`、epsilon `1e-8`，无 scheduler。光度损失使用 `static_mask.detach()`；mask 监督使用 `render.detach()`；DINO 只在 `torch.no_grad()` 中运行。

参考实现的 `L_wreg` 使用 `.data`，因此它只改变 loss 数值、不向 head 产生梯度。本实现有意保留该行为，没有静默修正，也没有创造新的正则。

### 3. Mask 证据与光度损失

历史 residual histogram 固定为 10000 bins、范围 `[0,1]`、`0.95 × history + current`、分位数 0.60/0.80。3×3 规则保持为“中心是 inlier，或邻域 inlier 平均值严格大于 0.5”。特征目标为 `clamp(2*cos-1,0,1)`。500 步后用 0.25 hard threshold 和 7×7 min-pool。Masked L1 使用整体 mean，绝不除以 mask sum；DSSIM 权重固定为 0.2。

### 4. 延迟 AbsGrad

统计从第 0 步累计；10000 ≤ step < 20000 时，每 100 步允许 split/clone/prune；20000 起完全停止 topology change。opacity reset 仅在 15000 和 18000 执行。reset 当步仍允许 head 更新，之后 1–300 步暂停，即 15001–15300 和 18001–18300 暂停。Gaussian 训练持续。

### 5. 训练和推理隔离

训练每步只有一次完整 Gaussian rasterization；20000 步前额外一次 224×224 coarse rasterization。正式 checkpoint 仍只有 `step` 和 `splats`，辅助 head/optimizer/histogram 分别写入 `aux/`。独立评测命令不接收 DINO、head 或 feature cache 参数，并输出 `evaluation_imported_dino=false`、`evaluation_loaded_mask_head=false`、rasterization count ratio 1.0。

### 6. 输出与门禁

训练输出包括 `config.yaml`、`dataset_split.json`、`environment.json`、`run_command.txt`、`git_commit.txt`、`train_metrics.json`、RU 曲线、DINO 用时、辅助状态和五张代表性图。独立评测输出 `test_metrics.json`、`per_image_metrics.csv`、`efficiency_metrics.json`、`ru_validation.json`。汇总工具只接受四个固定 30k run，以 seed 42、10000 次 paired bootstrap 执行附件中的唯一门禁；报告已存在时拒绝覆盖。

## 四、代码文件清单

### 修改文件

- `puri_gs/config.py`
- `run_puri_gs.py`
- `scripts/train_puri_gs.sh`
- `tests/test_puri_gs_configs.py`

### 新增且应进入里程碑 1 Git 的文件

- `configs/puri_gs_b1_full30k.yaml`
- `configs/puri_gs_ru_full30k.yaml`
- `patches/gsplat_v1.5.3_puri_gs_ru.patch`
- `puri_gs/delayed_absgrad.py`
- `puri_gs/dino_features.py`
- `puri_gs/ru_training.py`
- `puri_gs/semantic_mask.py`
- `scripts/prepare_puri_gs_ru.sh`
- `scripts/train_puri_gs_ru.sh`
- `tests/test_b1_checkpoint_comparison.py`
- `tests/test_delayed_absgrad.py`
- `tests/test_dino_features.py`
- `tests/test_mask_gradient_isolation.py`
- `tests/test_ru_checkpoint_compatibility.py`
- `tests/test_ru_inference_path.py`
- `tests/test_ru_summary.py`
- `tests/test_semantic_mask.py`
- `tools/cache_puri_gs_features.py`
- `tools/check_puri_gs_ru_environment.py`
- `tools/compare_puri_gs_b1_checkpoints.py`
- `tools/summarize_puri_gs_ru.py`
- `reports/PURI_GS_RU_IMPLEMENTATION_AND_RUNBOOK.md`

### 不得进入 Git

DINO 权重、`external/dinov2`、原始数据、`.pt` 特征缓存、checkpoint、训练日志、PNG、`analysis/`、`tmp/` 内 smoke bundle 和大型证据包。

## 五、已完成的本地权重步骤

### 步骤 1：Windows 下载官方 DINOv2 权重并计算 SHA-256

本步骤目的：取得附件固定的 `dinov2_vits14_reg` 官方权重，并让后续本地与服务器使用同一个经过校验的文件。

执行位置：Windows 本地。

需要打开的目录：在文件资源管理器中创建并打开 `E:\7-DataSet\PURI-GS-assets\dinov2`。

需要检查的文件：下载完成后必须存在 `E:\7-DataSet\PURI-GS-assets\dinov2\dinov2_vits14_reg4_pretrain.pth`。文件名必须完全一致，不能是 `.crdownload`、`.tmp`、`.html` 或带 `(1)` 的副本。

需要执行的命令：先用浏览器打开官方权重链接：

```text
https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth
```

保存完成后打开 PowerShell，只执行：

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath 'E:\7-DataSet\PURI-GS-assets\dinov2\dinov2_vits14_reg4_pretrain.pth' | Format-List
```

命令执行完成后应看到：`Algorithm : SHA256`、一串 64 位十六进制 `Hash`，以及上述完整文件路径。

日志位置：本步骤不生成训练日志；请把 PowerShell 的三行输出完整复制给 Codex。

如何判断成功：文件存在，扩展名为 `.pth`，PowerShell 无红色错误，Hash 恰好为 64 位。

出现什么情况应立即停止：浏览器下载的是网页；文件名不同；文件大小为 0；`Get-FileHash` 报路径不存在；下载中断。不要自己找第三方权重，不要改用无 registers 模型。

是否需要 Git 提交：否。

是否需要服务器 pull：否。

完成状态：已于 2026-09-01 通过。SHA-256、文件大小和权重内部结构均已由 Codex复核，无需重新下载或再次计算。

## 六、后续已锁定步骤（现在不要执行）

以下命令已经为后续验收写清楚，但必须逐步获得 Codex 许可；不能因为命令已写出就连续执行。

### 步骤 2：服务器核对 L20、上传权重并准备官方 DINOv2 源码

本步骤目的：让服务器具备唯一固定的 DINO 源码和与 Windows 完全相同的权重。

执行位置：Windows PowerShell 使用项目既有的 `scp` 方式上传；Linux 服务器负责校验。

需要打开的目录：服务器项目 `/home/chenglong/Uncertain-Nerf/uncertain-nerf`；服务器权重目录 `/home/chenglong/Uncertain-Nerf/uncertain-nerf/data/PURI-GS-assets/dinov2`。

需要检查的文件：本地步骤 1 的 `.pth`；服务器 GPU 6；服务器 `external/dinov2/hubconf.py`。

需要执行的命令：首先只执行 GPU 检查：

```bash
set -euo pipefail
nvidia-smi -i 6 --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
```

GPU 检查完成状态：已通过，输出为 `6, NVIDIA L20, 46068 MiB, 580.65.06`。

权重目录创建状态：已通过，目录所有者为 `chenglong:chenglong`。

上传完成状态：已通过。远端权重 SHA-256 为 `f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb`；远端 smoke 包 SHA-256 为 `f1839120f45c0c23b72b67b007164edce3818d2006bd8d0c43da45392771f827`。该已上传 smoke 包从此冻结，不再重新生成。

DINOv2 源码状态：已通过，服务器 commit 为 `7764ea0f912e53c92e82eb78a2a1631e92725fc8`。

当前停点：步骤 3–9 已通过并达到 Git 里程碑 1。下一步由用户在本地图形化 Git 客户端提交并 push；随后服务器主项目才允许 pull 和复核。正式 30k 仍不允许执行。

GPU 正确后，执行：

```bash
set -euo pipefail
mkdir -p /home/chenglong/Uncertain-Nerf/uncertain-nerf/data/PURI-GS-assets/dinov2
```

在 Windows PowerShell 上传权重：

```powershell
$remote = 'chenglong@172.16.55.2'
$localWeight = 'E:\7-DataSet\PURI-GS-assets\dinov2\dinov2_vits14_reg4_pretrain.pth'
$remoteWeightDir = '/home/chenglong/Uncertain-Nerf/uncertain-nerf/data/PURI-GS-assets/dinov2'

if (-not (Test-Path -LiteralPath $localWeight -PathType Leaf)) {
    throw "LOCAL_WEIGHT_MISSING: $localWeight"
}

scp $localWeight "${remote}:$remoteWeightDir/"
if ($LASTEXITCODE -ne 0) { throw 'DINO weight upload failed' }
```

然后执行：

```bash
set -euo pipefail
sha256sum /home/chenglong/Uncertain-Nerf/uncertain-nerf/data/PURI-GS-assets/dinov2/dinov2_vits14_reg4_pretrain.pth
```

仅当 SHA 与步骤 1 完全一致时，再执行下面的官方依赖源码克隆；这不是项目分支或项目 commit 操作：

```bash
set -euo pipefail
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf/external
test ! -e /home/chenglong/Uncertain-Nerf/uncertain-nerf/external/dinov2
git clone https://github.com/facebookresearch/dinov2.git /home/chenglong/Uncertain-Nerf/uncertain-nerf/external/dinov2
git -C /home/chenglong/Uncertain-Nerf/uncertain-nerf/external/dinov2 rev-parse HEAD
test -f /home/chenglong/Uncertain-Nerf/uncertain-nerf/external/dinov2/hubconf.py
```

命令执行完成后应看到：第一条输出索引 6 和 `NVIDIA L20`；两个 SHA 完全一致；DINO 命令输出一个 40 位 commit；最后一个 `test` 无输出且退出码为 0。

日志位置：暂不写文件；复制 GPU 行、服务器 SHA 和 DINO commit 给 Codex。

如何判断成功：三项事实完全满足：GPU 6=L20、SHA 一致、`hubconf.py` 存在。

出现什么情况应立即停止：GPU 不是 L20；SHA 不同；服务器联网克隆失败；目标 `external/dinov2` 已存在。不要覆盖已有目录，不要执行 pip install。

是否需要 Git 提交：否。

是否需要服务器 pull：否。

### 步骤 3：上传 smoke-only 代码包并建立隔离副本

本步骤目的：在 Git 里程碑 1 之前，从服务器本地 `dev` commit 创建独立 clone，再把候选代码放入该副本 smoke。主项目中已有的 3 个 Phase 3A 暂存文件保持原样，不取消暂存、不提交、不复制到 smoke 副本。

执行位置：Windows PowerShell `scp` + Linux 服务器。

需要打开的目录：本地 `E:\6-Project\1-UncertainNerf\uncertain-nerf\tmp`；服务器 `/home/chenglong`。

需要检查的文件：`puri-gs-ru-smoke-candidate.zip` 及 Codex 给出的 SHA-256；服务器目标 `/home/chenglong/Uncertain-Nerf-RU-Smoke` 必须不存在。

需要执行的命令：在 Windows PowerShell 上传 zip：

```powershell
$remote = 'chenglong@172.16.55.2'
$localBundle = 'E:\6-Project\1-UncertainNerf\uncertain-nerf\tmp\puri-gs-ru-smoke-candidate.zip'

if (-not (Test-Path -LiteralPath $localBundle -PathType Leaf)) {
    throw "LOCAL_SMOKE_BUNDLE_MISSING: $localBundle"
}

scp $localBundle "${remote}:/home/chenglong/puri-gs-ru-smoke-candidate.zip"
if ($LASTEXITCODE -ne 0) { throw 'smoke bundle upload failed' }
```

服务器主项目存在以下用户暂存文件，这是允许保留的既有状态：`PHASE_3A_CVTR_FAILURE_ATTRIBUTION.md`、`PHASE_3A_CVTR_MASK_VALIDATION.md`、`phase_3a_cvtr_failure_attribution.json`。不得为了 smoke 清理它们。使用下面的本地 clone 命令；失败只退出子 Bash，不关闭 SSH 终端：

```bash
set +e
bash -euo pipefail <<'BASH'
cd /home/chenglong
test ! -e /home/chenglong/Uncertain-Nerf-RU-Smoke
test "$(git -C /home/chenglong/Uncertain-Nerf branch --show-current)" = "dev"
test "$(git -C /home/chenglong/Uncertain-Nerf rev-parse HEAD)" = "0ae62b10ef4d69c35cfcc1aafdfab37f37bd98fe"
test "$(sha256sum /home/chenglong/puri-gs-ru-smoke-candidate.zip | cut -d ' ' -f 1)" = "f1839120f45c0c23b72b67b007164edce3818d2006bd8d0c43da45392771f827"
git clone --no-hardlinks --branch dev --single-branch /home/chenglong/Uncertain-Nerf /home/chenglong/Uncertain-Nerf-RU-Smoke
unzip -q -o /home/chenglong/puri-gs-ru-smoke-candidate.zip -d /home/chenglong/Uncertain-Nerf-RU-Smoke/uncertain-nerf
cd /home/chenglong/Uncertain-Nerf-RU-Smoke/uncertain-nerf
test "$(git branch --show-current)" = "dev"
test "$(git rev-parse HEAD)" = "0ae62b10ef4d69c35cfcc1aafdfab37f37bd98fe"
test "$(git status --porcelain | wc -l)" -eq 26
test -f tools/check_puri_gs_ru_environment.py
test -f patches/gsplat_v1.5.3_puri_gs_ru.patch
echo "SMOKE-STAGING-PASS"
BASH
stage_status=$?
echo "SMOKE-STAGING-EXIT-CODE=$stage_status"
test "$stage_status" -eq 0 && echo "SMOKE-STAGING-COMMAND-PASS" || echo "SMOKE-STAGING-COMMAND-FAIL"
```

命令执行完成后应看到：`SMOKE-STAGING-PASS`、`SMOKE-STAGING-EXIT-CODE=0`、`SMOKE-STAGING-COMMAND-PASS`。分支仍为 `dev`，commit 仍为 `0ae62b10ef4d69c35cfcc1aafdfab37f37bd98fe`，候选改动数量为 26。

日志位置：本步骤没有训练日志。

如何判断成功：主项目 `/home/chenglong/Uncertain-Nerf` 未改，候选副本存在且 commit/文件清单正确。

出现什么情况应立即停止：输出 `SMOKE-STAGING-COMMAND-FAIL`、exit code 非 0、目标副本已存在、zip SHA 不同、commit 不同或候选改动数量不是 26。失败时不得删除可能已产生的隔离目录，先交给 Codex检查。

是否需要 Git 提交：否。

是否需要服务器 pull：否。

### 步骤 4：建立 base trainer 并执行 B1 100-step 参考 smoke

本步骤目的：先得到 RU patch 应保持不变的 B1 数值参考。

执行位置：Linux 服务器隔离副本。

需要打开的目录：`/home/chenglong/Uncertain-Nerf-RU-Smoke/uncertain-nerf`。

需要检查的文件：Android 服务器数据 `/home/chenglong/Uncertain-Nerf/uncertain-nerf/data/nerf_robustnerf/robustnerf/android`；固定 Python `/home/chenglong/Uncertain-Nerf/uncertain-nerf/.venv-gsplat153/bin/python`。

需要执行的命令：

```bash
set -euo pipefail
cd /home/chenglong/Uncertain-Nerf-RU-Smoke/uncertain-nerf
test ! -e /home/chenglong/Uncertain-Nerf-RU-Smoke/uncertain-nerf/external/gsplat-v1.5.3-ru-smoke
bash scripts/prepare_gsplat_robot_baseline.sh /home/chenglong/Uncertain-Nerf-RU-Smoke/uncertain-nerf/external/gsplat-v1.5.3-ru-smoke
mkdir -p /home/chenglong/Uncertain-Nerf-RU-Smoke/console
test ! -e /home/chenglong/Uncertain-Nerf-RU-Smoke/uncertain-nerf/logs-puri/b1_base_100
PURI_GSPLAT_PYTHON=/home/chenglong/Uncertain-Nerf/uncertain-nerf/.venv-gsplat153/bin/python bash scripts/train_puri_gs.sh configs/puri_gs_b1_full30k.yaml /home/chenglong/Uncertain-Nerf-RU-Smoke/uncertain-nerf/external/gsplat-v1.5.3-ru-smoke /home/chenglong/Uncertain-Nerf/uncertain-nerf/data/nerf_robustnerf/robustnerf/android /home/chenglong/Uncertain-Nerf-RU-Smoke/uncertain-nerf/logs-puri/b1_base_100 6 100 4 clutter extra 2>&1 | tee /home/chenglong/Uncertain-Nerf-RU-Smoke/console/b1_base_100.log
```

命令执行完成后应看到：trainer patch prepared；进度到 100/100；存在 `logs-puri/b1_base_100/ckpts/ckpt_99_rank0.pt` 和 `stats/train_step0099_rank0.json`；无 traceback、NaN、CUDA OOM。

日志位置：`/home/chenglong/Uncertain-Nerf-RU-Smoke/console/b1_base_100.log`。

如何判断成功：命令退出码 0，checkpoint 和 stats 均存在。

出现什么情况应立即停止：gsplat commit 不符；安装包不是 1.5.3；数据 split 失败；任何 CUDA/NaN/OOM 错误。

是否需要 Git 提交：否。

是否需要服务器 pull：否。

### 步骤 5：应用 RU patch，执行 B1 100-step 回归并比较独立评测指标

本步骤目的：证明 RU trainer 在功能关闭时保持 B1 独立评测质量和精确 Gaussian 数。两个独立 CUDA 训练进程即使 seed 相同，也不要求所有浮点 tensor 逐元素完全一致；tensor 差异只作诊断，不作为失败条件。

执行位置：Linux 服务器隔离副本。

需要打开的目录：同步骤 4。

需要检查的文件：步骤 4 checkpoint。

需要执行的命令：

```bash
set -euo pipefail
cd /home/chenglong/Uncertain-Nerf-RU-Smoke/uncertain-nerf
bash scripts/prepare_puri_gs_ru.sh /home/chenglong/Uncertain-Nerf-RU-Smoke/uncertain-nerf/external/gsplat-v1.5.3-ru-smoke
test ! -e /home/chenglong/Uncertain-Nerf-RU-Smoke/uncertain-nerf/logs-puri/b1_ru_superset_100
PURI_GSPLAT_PYTHON=/home/chenglong/Uncertain-Nerf/uncertain-nerf/.venv-gsplat153/bin/python bash scripts/train_puri_gs.sh configs/puri_gs_b1_full30k.yaml /home/chenglong/Uncertain-Nerf-RU-Smoke/uncertain-nerf/external/gsplat-v1.5.3-ru-smoke /home/chenglong/Uncertain-Nerf/uncertain-nerf/data/nerf_robustnerf/robustnerf/android /home/chenglong/Uncertain-Nerf-RU-Smoke/uncertain-nerf/logs-puri/b1_ru_superset_100 6 100 4 clutter extra 2>&1 | tee /home/chenglong/Uncertain-Nerf-RU-Smoke/console/b1_ru_superset_100.log
```

完成后分别将两个 checkpoint 独立评测到各自的 `independent_eval` 目录，再用已上传且 SHA-256 为 `381c7e74ac3ab7051e16079b15827501a3ebdd7152da3a61ac297ac0be59ed3f` 的 `tools/compare_puri_gs_b1_metrics.py` 比较：

```bash
set -euo pipefail
/home/chenglong/Uncertain-Nerf/uncertain-nerf/.venv-gsplat153/bin/python tools/compare_puri_gs_b1_metrics.py \
  --reference logs-puri/b1_base_100/ckpts/ckpt_99_rank0.pt \
  --candidate logs-puri/b1_ru_superset_100/ckpts/ckpt_99_rank0.pt \
  --reference-metrics logs-puri/b1_base_100/independent_eval/test_metrics.json \
  --candidate-metrics logs-puri/b1_ru_superset_100/independent_eval/test_metrics.json \
  --output /home/chenglong/Uncertain-Nerf-RU-Smoke/console/b1_metrics_regression.json
```

命令执行完成后应看到：`B1_REGRESSION_PASS`，reference/candidate Gaussian 数完全相同，PSNR/SSIM/LPIPS 的绝对差分别不超过 0.05/0.001/0.002。`tensor_differences_diagnostic_only` 允许非空，`tensor_equality_required` 必须为 `false`。

日志位置：`console/b1_ru_superset_100.log`、两个 `independent_eval/test_metrics.json` 与 `console/b1_metrics_regression.json`。

如何判断成功：比较器退出码 0 且唯一状态为 PASS。

出现什么情况应立即停止：比较器 FAIL、shape/key 不同、Gaussian 数不同、任一指标超出固定容差或训练/评测错误。不要调宽容差。

是否需要 Git 提交：否。

是否需要服务器 pull：否。

### 步骤 6：DINO/gsplat 环境检查和单图 coarse/fine 前向

本步骤目的：在 L20 上验证固定环境、DINO 冻结、权重 SHA、源码 commit、16×16/36×36 形状和有限值。

执行位置：Linux 服务器隔离副本。

需要打开的目录：同步骤 4。

需要检查的文件：服务器 DINO 源码和权重。

需要执行的命令：

```bash
set -euo pipefail
cd /home/chenglong/Uncertain-Nerf-RU-Smoke/uncertain-nerf
test ! -e /home/chenglong/Uncertain-Nerf-RU-Smoke/console/ru_environment.json
CUDA_VISIBLE_DEVICES=6 /home/chenglong/Uncertain-Nerf/uncertain-nerf/.venv-gsplat153/bin/python tools/check_puri_gs_ru_environment.py --gsplat-dir /home/chenglong/Uncertain-Nerf-RU-Smoke/uncertain-nerf/external/gsplat-v1.5.3-ru-smoke --dino-repo-dir /home/chenglong/Uncertain-Nerf/uncertain-nerf/external/dinov2 --dino-weight-path /home/chenglong/Uncertain-Nerf/uncertain-nerf/data/PURI-GS-assets/dinov2/dinov2_vits14_reg4_pretrain.pth --device cuda:0 --output /home/chenglong/Uncertain-Nerf-RU-Smoke/console/ru_environment.json
```

命令执行完成后应看到：`status=PURI_GS_RU_ENVIRONMENT_READY`；GPU 为 L20；gsplat 1.5.3；DINO trainable count 0；coarse `[1,384,16,16]`；fine `[1,384,36,36]`；`features_finite=true`。

日志位置：`/home/chenglong/Uncertain-Nerf-RU-Smoke/console/ru_environment.json`。

如何判断成功：上述每个字段都严格满足。

出现什么情况应立即停止：缺包、CUDA 不可用、权重 strict load 失败、SHA 不同、DINO 可训练参数不为 0、shape 不符或非有限值。不要自行升级 PyTorch/CUDA 或安装替代 DINO。

是否需要 Git 提交：否。

是否需要服务器 pull：否。

### 步骤 7：预计算 Android 两级特征缓存

本步骤目的：每张 122 张 Android 训练图只提取一次 GT coarse/fine 特征。

执行位置：Linux 服务器隔离副本。

需要打开的目录：同步骤 4。

需要检查的文件：输出目录必须不存在。

需要执行的命令：

```bash
set +e
bash -euo pipefail <<'BASH'
cd /home/chenglong/Uncertain-Nerf-RU-Smoke/uncertain-nerf
mkdir -p /home/chenglong/Uncertain-Nerf/uncertain-nerf/data/PURI-GS-derived/semantic_features
test ! -e /home/chenglong/Uncertain-Nerf/uncertain-nerf/data/PURI-GS-derived/semantic_features/android
test ! -e /home/chenglong/Uncertain-Nerf/uncertain-nerf/data/PURI-GS-derived/semantic_features/android.building
CUDA_VISIBLE_DEVICES=6 /home/chenglong/Uncertain-Nerf/uncertain-nerf/.venv-gsplat153/bin/python tools/cache_puri_gs_features.py --scene android --gsplat-dir /home/chenglong/Uncertain-Nerf-RU-Smoke/uncertain-nerf/external/gsplat-v1.5.3-ru-smoke --data-dir /home/chenglong/Uncertain-Nerf/uncertain-nerf/data/nerf_robustnerf/robustnerf/android --data-factor 4 --output-dir /home/chenglong/Uncertain-Nerf/uncertain-nerf/data/PURI-GS-derived/semantic_features/android --dino-repo-dir /home/chenglong/Uncertain-Nerf/uncertain-nerf/external/dinov2 --dino-weight-path /home/chenglong/Uncertain-Nerf/uncertain-nerf/data/PURI-GS-assets/dinov2/dinov2_vits14_reg4_pretrain.pth --device cuda:0 --train-keyword clutter --test-keyword extra 2>&1 | tee /home/chenglong/Uncertain-Nerf-RU-Smoke/console/cache_android.log
BASH
puri_cache_exit_code=$?
echo "ANDROID-CACHE-EXIT-CODE=${puri_cache_exit_code}"
if [ "${puri_cache_exit_code}" -eq 0 ]; then echo "ANDROID-FEATURE-CACHE-PASS"; else echo "ANDROID-FEATURE-CACHE-FAIL"; fi
```

命令执行完成后应看到：从 `[1/122]` 到 `[122/122]`，最后一行 `FEATURE_CACHE_READY scene=android images=122`。manifest 中 coarse shape `[384,16,16]`、fine `[384,36,36]`。

日志位置：`/home/chenglong/Uncertain-Nerf-RU-Smoke/console/cache_android.log`；manifest 位于服务器 cache 目录。

如何判断成功：命令退出码 0，最终目录存在，`.building` 不存在，manifest 记录 122 张且 SHA 与 runtime 一致。

出现什么情况应立即停止：训练图数量不是 122；名称 mapping 重复/缺失；shape/finite/SHA 错；目标目录预先存在。不要删除旧目录后重跑，先让 Codex检查。

是否需要 Git 提交：否。

是否需要服务器 pull：否。

### 步骤 8：PURI-GS-RU 100-step CUDA smoke

本步骤目的：在 L20 上同时验证 coarse render、DINO、mask head、Gaussian backward、梯度隔离、checkpoint/aux 输出和固定策略构造。

执行位置：Linux 服务器隔离副本。

需要打开的目录：同步骤 4。

需要检查的文件：Android cache manifest；RU smoke 输出目录必须不存在。

需要执行的命令：

```bash
set +e
bash -euo pipefail <<'BASH'
cd /home/chenglong/Uncertain-Nerf-RU-Smoke/uncertain-nerf
test ! -e /home/chenglong/Uncertain-Nerf-RU-Smoke/uncertain-nerf/logs-puri/android_ru_100
PURI_GSPLAT_PYTHON=/home/chenglong/Uncertain-Nerf/uncertain-nerf/.venv-gsplat153/bin/python bash scripts/train_puri_gs_ru.sh /home/chenglong/Uncertain-Nerf-RU-Smoke/uncertain-nerf/external/gsplat-v1.5.3-ru-smoke /home/chenglong/Uncertain-Nerf/uncertain-nerf/data/nerf_robustnerf/robustnerf/android /home/chenglong/Uncertain-Nerf-RU-Smoke/uncertain-nerf/logs-puri/android_ru_100 6 /home/chenglong/Uncertain-Nerf/uncertain-nerf/external/dinov2 /home/chenglong/Uncertain-Nerf/uncertain-nerf/data/PURI-GS-assets/dinov2/dinov2_vits14_reg4_pretrain.pth /home/chenglong/Uncertain-Nerf/uncertain-nerf/data/PURI-GS-derived/semantic_features/android 100 4 clutter extra 2>&1 | tee /home/chenglong/Uncertain-Nerf-RU-Smoke/console/android_ru_100.log
BASH
puri_ru_exit_code=$?
echo "ANDROID-RU-100-EXIT-CODE=${puri_ru_exit_code}"
if [ "${puri_ru_exit_code}" -eq 0 ]; then echo "ANDROID-RU-100-INNER-PASS"; else echo "ANDROID-RU-100-FAIL"; fi
```

命令执行完成后应看到：DINO trainable=0；进度 100/100；无梯度隔离异常；`ckpts/ckpt_99_rank0.pt`；`aux/mask_head_step99.pt`、`mask_optimizer_step99.pt`、`residual_hist_step99.pt`、`training_schedule.json`；`train_metrics.json`、`DINO_time.json` 和五张代表性 PNG。

日志位置：`/home/chenglong/Uncertain-Nerf-RU-Smoke/console/android_ru_100.log`。

如何判断成功：退出码 0，标准 checkpoint 与辅助文件分离，mask updates 大于 0，DINO trainable 参数为 0，曲线无 NaN。

出现什么情况应立即停止：CUDA OOM、NaN/Inf、梯度隔离异常、cache mapping 错、checkpoint 含 head/DINO、输出缺失。

是否需要 Git 提交：否。

是否需要服务器 pull：否。

### 步骤 9：只用标准 checkpoint 独立评测 100-step smoke

本步骤目的：证明评测进程不加载 DINO、mask head、histogram 或 feature cache，且只 rasterize 每张测试图一次。

执行位置：Linux 服务器隔离副本。

需要打开的目录：同步骤 4。

需要检查的文件：步骤 8 checkpoint；`independent_eval` 必须不存在。

需要执行的命令：

```bash
set +e
bash -euo pipefail <<'BASH'
cd /home/chenglong/Uncertain-Nerf-RU-Smoke/uncertain-nerf
test ! -e /home/chenglong/Uncertain-Nerf-RU-Smoke/uncertain-nerf/logs-puri/android_ru_100/independent_eval
/home/chenglong/Uncertain-Nerf/uncertain-nerf/.venv-gsplat153/bin/python run_puri_gs.py --config configs/puri_gs_ru_full30k.yaml --gsplat-dir /home/chenglong/Uncertain-Nerf-RU-Smoke/uncertain-nerf/external/gsplat-v1.5.3-ru-smoke --data-dir /home/chenglong/Uncertain-Nerf/uncertain-nerf/data/nerf_robustnerf/robustnerf/android --result-dir /home/chenglong/Uncertain-Nerf-RU-Smoke/uncertain-nerf/logs-puri/android_ru_100/independent_eval --gpu 6 --data-factor 4 --train-keyword clutter --test-keyword extra --checkpoint /home/chenglong/Uncertain-Nerf-RU-Smoke/uncertain-nerf/logs-puri/android_ru_100/ckpts/ckpt_99_rank0.pt 2>&1 | tee /home/chenglong/Uncertain-Nerf-RU-Smoke/console/android_ru_100_eval.log
BASH
puri_eval_exit_code=$?
echo "ANDROID-RU-EVAL-EXIT-CODE=${puri_eval_exit_code}"
if [ "${puri_eval_exit_code}" -eq 0 ]; then echo "ANDROID-RU-EVAL-INNER-PASS"; else echo "ANDROID-RU-EVAL-FAIL"; fi
```

命令执行完成后应看到：完成 19 张 test 图；输出 PSNR/SSIM/LPIPS；`ru_validation.json` 中 standard load true、imported DINO false、loaded head false、rasterization ratio 1.0。

日志位置：`console/android_ru_100_eval.log`；指标位于 smoke run 的 `independent_eval`。

如何判断成功：`per_image_metrics.csv` 恰好 19 行数据，四项 validation 严格满足。

出现什么情况应立即停止：评测要求提供 DINO/cache；DINO/head 标志为 true；测试图不是 19；重复 rasterization；指标非有限。

是否需要 Git 提交：完成本步骤且 Codex复核所有 smoke 证据后，才达到 Git 里程碑 1。

是否需要服务器 pull：仍否；先由用户完成图形化 commit/push。

## 七、Git 里程碑 1（仅在步骤 2–9 全部通过后）

现在可以使用图形化 Git 提交并 push。

建议提交说明：

```text
implement PURI-GS-RU semantic mask and delayed AbsGrad training
```

提交应包含：本手册第四节“修改文件”和“新增且应进入里程碑 1 Git 的文件”中的全部内容。

提交不应包含：DINO 权重、`external/dinov2`、Android/Room 数据、feature cache、checkpoint、`logs-puri`、PNG、smoke zip、服务器 JSON/CSV、`.venv`、`analysis`。

服务器是否需要 pull：push 完成后，服务器主项目需要 pull 一次；隔离 smoke 副本不需要 pull。

pull 后需要执行什么：先核对服务器主项目仍为 `dev` 且 commit 与图形化客户端刚 push 的 commit 完全相同；然后执行环境检查和 100-step 单元证据复核。未获得用户对正式 30k 的明确确认前，停止。

## 八、正式 30k 前的服务器主项目准备（里程碑 1 后逐步执行）

### 步骤 10：服务器主项目 pull 后准备固定 RU source

本步骤目的：让正式 B1/RU 使用同一个全新、固定 commit、已应用 RU-superset patch 的 gsplat 源码，保留历史 external 源码不被覆盖。

执行位置：Linux 服务器主项目。

需要打开的目录：`/home/chenglong/Uncertain-Nerf/uncertain-nerf`。

需要检查的文件：图形化客户端显示的最新 dev commit；服务器工作区无意外修改。

需要执行的命令：项目 pull 本身使用用户现有的 Git 操作方式，不提供 Git 命令。pull 完成后执行：

```bash
set -euo pipefail
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
git status --short --branch
git rev-parse HEAD
test ! -e /home/chenglong/Uncertain-Nerf/uncertain-nerf/external/gsplat-v1.5.3-ru
bash scripts/prepare_puri_gs_ru.sh /home/chenglong/Uncertain-Nerf/uncertain-nerf/external/gsplat-v1.5.3-ru
mkdir -p /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/phase_r-console
test ! -e /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/phase_r-console/main_environment.json
CUDA_VISIBLE_DEVICES=6 /home/chenglong/Uncertain-Nerf/uncertain-nerf/.venv-gsplat153/bin/python tools/check_puri_gs_ru_environment.py --gsplat-dir /home/chenglong/Uncertain-Nerf/uncertain-nerf/external/gsplat-v1.5.3-ru --dino-repo-dir /home/chenglong/Uncertain-Nerf/uncertain-nerf/external/dinov2 --dino-weight-path /home/chenglong/Uncertain-Nerf/uncertain-nerf/data/PURI-GS-assets/dinov2/dinov2_vits14_reg4_pretrain.pth --feature-cache-dir /home/chenglong/Uncertain-Nerf/uncertain-nerf/data/PURI-GS-derived/semantic_features/android --device cuda:0 --output /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/phase_r-console/main_environment.json
```

命令执行完成后应看到：分支 `dev`；commit 与 push 一致；固定 gsplat commit；`Prepared PURI-GS-RU on gsplat v1.5.3`；环境状态 `PURI_GS_RU_ENVIRONMENT_READY` 且 Android cache 为 122 张。

日志位置：`logs-puri/phase_r-console/main_environment.json`。

如何判断成功：没有修改历史 `external/gsplat-v1.5.3`，新目录反向 patch check 成功。

出现什么情况应立即停止：服务器不是 dev、commit 不同、工作区有未知修改、目标目录已存在、patch 不能干净应用。

是否需要 Git 提交：否。

是否需要服务器 pull：本步骤已经是唯一一次 pull，后续正式四个 run 期间不再 pull。

### 步骤 11：Room 特征缓存

本步骤目的：在正式 Room run 前预计算唯一两级 cache。

执行位置：Linux 服务器主项目。

需要打开的目录：主项目。

需要检查的文件：Room 数据；输出目录和 `.building` 均不存在。

需要执行的命令：

```bash
set -euo pipefail
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
mkdir -p /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/phase_r-console
test ! -e /home/chenglong/Uncertain-Nerf/uncertain-nerf/data/PURI-GS-derived/semantic_features/room
test ! -e /home/chenglong/Uncertain-Nerf/uncertain-nerf/data/PURI-GS-derived/semantic_features/room.building
CUDA_VISIBLE_DEVICES=6 /home/chenglong/Uncertain-Nerf/uncertain-nerf/.venv-gsplat153/bin/python tools/cache_puri_gs_features.py --scene room --gsplat-dir /home/chenglong/Uncertain-Nerf/uncertain-nerf/external/gsplat-v1.5.3-ru --data-dir /home/chenglong/Uncertain-Nerf/uncertain-nerf/data/mipnerf360/360_v2/room --data-factor 4 --output-dir /home/chenglong/Uncertain-Nerf/uncertain-nerf/data/PURI-GS-derived/semantic_features/room --dino-repo-dir /home/chenglong/Uncertain-Nerf/uncertain-nerf/external/dinov2 --dino-weight-path /home/chenglong/Uncertain-Nerf/uncertain-nerf/data/PURI-GS-assets/dinov2/dinov2_vits14_reg4_pretrain.pth --device cuda:0 2>&1 | tee /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/phase_r-console/cache_room.log
```

命令执行完成后应看到：`[272/272]` 和 `FEATURE_CACHE_READY scene=room images=272`。

日志位置：`logs-puri/phase_r-console/cache_room.log`。

如何判断成功：manifest 恰好 272 张，形状/SHA/finite 均正确。

出现什么情况应立即停止：数量不是 272、split 使用了 keyword、输出已存在、mapping/SHA/shape 错。

是否需要 Git 提交：否。

是否需要服务器 pull：否。

## 九、四个正式 run（必须再次获得用户明确确认；现在禁止执行）

开始前统一规则：GPU 必须还是 6=L20；Android 和 Room cache 均已通过；四个结果目录必须不存在；`logs-puri/phase_r-console` 可以存在；正式 run 期间不 pull、不改代码、不改配置、不改阈值、不并行抢同一张 GPU。

### 步骤 12：Android B1 30k

本步骤目的：生成 Android 固定 B1 对照。

执行位置：Linux 服务器主项目。

需要打开的目录：主项目。

需要检查的文件：B1 配置、Android 数据、结果目录不存在。

需要执行的命令：

```bash
set -euo pipefail
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
mkdir -p /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/phase_r-console
test ! -e /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/phase_r/android_b1_30k
PURI_GSPLAT_PYTHON=/home/chenglong/Uncertain-Nerf/uncertain-nerf/.venv-gsplat153/bin/python bash scripts/train_puri_gs.sh configs/puri_gs_b1_full30k.yaml /home/chenglong/Uncertain-Nerf/uncertain-nerf/external/gsplat-v1.5.3-ru /home/chenglong/Uncertain-Nerf/uncertain-nerf/data/nerf_robustnerf/robustnerf/android /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/phase_r/android_b1_30k 6 30000 4 clutter extra 2>&1 | tee /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/phase_r-console/android_b1_30k.log
```

命令执行完成后应看到：30000/30000；`ckpts/ckpt_29999_rank0.pt`；`train_metrics.json`；无 DINO 初始化。

日志位置：`logs-puri/phase_r-console/android_b1_30k.log`。

如何判断成功：退出码 0、step 29999、配置 seed 42/SH3/AbsGrad/grow 0.0006、split 122/19。

出现什么情况应立即停止：输出已存在、非 L20、NaN/OOM、checkpoint 缺失、DINO 被加载、split 不符。

是否需要 Git 提交：否。

是否需要服务器 pull：否。

### 步骤 13：Android B1 独立评测

本步骤目的：生成 Android B1 19 张逐图指标和效率对照。

执行位置：Linux 服务器主项目。

需要打开的目录：`/home/chenglong/Uncertain-Nerf/uncertain-nerf`。

需要检查的文件：步骤 12 的 `ckpt_29999_rank0.pt`；`android_b1_30k/independent_eval` 必须不存在。

需要执行的命令：

```bash
set -euo pipefail
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
test ! -e /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/phase_r/android_b1_30k/independent_eval
/home/chenglong/Uncertain-Nerf/uncertain-nerf/.venv-gsplat153/bin/python run_puri_gs.py --config configs/puri_gs_b1_full30k.yaml --gsplat-dir /home/chenglong/Uncertain-Nerf/uncertain-nerf/external/gsplat-v1.5.3-ru --data-dir /home/chenglong/Uncertain-Nerf/uncertain-nerf/data/nerf_robustnerf/robustnerf/android --result-dir /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/phase_r/android_b1_30k/independent_eval --gpu 6 --data-factor 4 --train-keyword clutter --test-keyword extra --checkpoint /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/phase_r/android_b1_30k/ckpts/ckpt_29999_rank0.pt 2>&1 | tee /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/phase_r-console/android_b1_eval.log
```

命令执行完成后应看到：19 张；四个评测交付文件；DINO/head false。

日志位置：`logs-puri/phase_r-console/android_b1_eval.log`。

如何判断成功：CSV 19 行、指标有限、validation 全通过。

出现什么情况应立即停止：同步骤 9。

是否需要 Git 提交：否。是否需要服务器 pull：否。

### 步骤 14：Android RU 30k 与独立评测

本步骤目的：生成 Android 唯一 RU 候选和独立指标。

执行位置：Linux 服务器主项目。

需要打开的目录：`/home/chenglong/Uncertain-Nerf/uncertain-nerf`。

需要检查的文件：Android cache manifest 恰好 122 张；训练和独立评测目标目录均不存在；DINO SHA 与环境审计一致。

需要执行的训练命令：

```bash
set -euo pipefail
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
test ! -e /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/phase_r/android_ru_30k
PURI_GSPLAT_PYTHON=/home/chenglong/Uncertain-Nerf/uncertain-nerf/.venv-gsplat153/bin/python bash scripts/train_puri_gs_ru.sh /home/chenglong/Uncertain-Nerf/uncertain-nerf/external/gsplat-v1.5.3-ru /home/chenglong/Uncertain-Nerf/uncertain-nerf/data/nerf_robustnerf/robustnerf/android /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/phase_r/android_ru_30k 6 /home/chenglong/Uncertain-Nerf/uncertain-nerf/external/dinov2 /home/chenglong/Uncertain-Nerf/uncertain-nerf/data/PURI-GS-assets/dinov2/dinov2_vits14_reg4_pretrain.pth /home/chenglong/Uncertain-Nerf/uncertain-nerf/data/PURI-GS-derived/semantic_features/android 30000 4 clutter extra 2>&1 | tee /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/phase_r-console/android_ru_30k.log
```

训练通过后才执行评测：

```bash
set -euo pipefail
test ! -e /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/phase_r/android_ru_30k/independent_eval
/home/chenglong/Uncertain-Nerf/uncertain-nerf/.venv-gsplat153/bin/python run_puri_gs.py --config configs/puri_gs_ru_full30k.yaml --gsplat-dir /home/chenglong/Uncertain-Nerf/uncertain-nerf/external/gsplat-v1.5.3-ru --data-dir /home/chenglong/Uncertain-Nerf/uncertain-nerf/data/nerf_robustnerf/robustnerf/android --result-dir /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/phase_r/android_ru_30k/independent_eval --gpu 6 --data-factor 4 --train-keyword clutter --test-keyword extra --checkpoint /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/phase_r/android_ru_30k/ckpts/ckpt_29999_rank0.pt 2>&1 | tee /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/phase_r-console/android_ru_eval.log
```

命令执行完成后应看到：训练 checkpoint/aux/曲线/五图齐全；独立评测 19 张；DINO/head false。

日志位置：`phase_r-console/android_ru_30k.log` 和 `android_ru_eval.log`。

如何判断成功：所有训练/评测文件齐全、无 NaN、validation 全通过。

出现什么情况应立即停止：任一步失败时不继续 Room，不调参、不重跑不同 seed。

是否需要 Git 提交：否。是否需要服务器 pull：否。

### 步骤 15：Room B1 30k 与独立评测

本步骤目的：生成 clean Room 对照。

执行位置：Linux 服务器主项目。

需要打开的目录：`/home/chenglong/Uncertain-Nerf/uncertain-nerf`。

需要检查的文件：Room 数据 split 272/39；训练和独立评测目标目录均不存在。

需要执行的训练命令：

```bash
set -euo pipefail
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
test ! -e /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/phase_r/room_b1_30k
PURI_GSPLAT_PYTHON=/home/chenglong/Uncertain-Nerf/uncertain-nerf/.venv-gsplat153/bin/python bash scripts/train_puri_gs.sh configs/puri_gs_b1_full30k.yaml /home/chenglong/Uncertain-Nerf/uncertain-nerf/external/gsplat-v1.5.3-ru /home/chenglong/Uncertain-Nerf/uncertain-nerf/data/mipnerf360/360_v2/room /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/phase_r/room_b1_30k 6 30000 4 2>&1 | tee /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/phase_r-console/room_b1_30k.log
```

训练通过后执行：

```bash
set -euo pipefail
test ! -e /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/phase_r/room_b1_30k/independent_eval
/home/chenglong/Uncertain-Nerf/uncertain-nerf/.venv-gsplat153/bin/python run_puri_gs.py --config configs/puri_gs_b1_full30k.yaml --gsplat-dir /home/chenglong/Uncertain-Nerf/uncertain-nerf/external/gsplat-v1.5.3-ru --data-dir /home/chenglong/Uncertain-Nerf/uncertain-nerf/data/mipnerf360/360_v2/room --result-dir /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/phase_r/room_b1_30k/independent_eval --gpu 6 --data-factor 4 --checkpoint /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/phase_r/room_b1_30k/ckpts/ckpt_29999_rank0.pt 2>&1 | tee /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/phase_r-console/room_b1_eval.log
```

命令执行完成后应看到：split 272/39；评测 CSV 39 行；validation 全通过。

日志位置：对应两个 `phase_r-console` 日志。

如何判断成功：checkpoint step 29999、39 张指标有限。

出现什么情况应立即停止：split、CUDA、checkpoint 或 validation 任一失败。

是否需要 Git 提交：否。是否需要服务器 pull：否。

### 步骤 16：Room RU 30k 与独立评测

本步骤目的：生成 clean Room RU 结果。

执行位置：Linux 服务器主项目。

需要打开的目录：`/home/chenglong/Uncertain-Nerf/uncertain-nerf`。

需要检查的文件：Room cache manifest 恰好 272 张；训练和独立评测目标目录均不存在；DINO SHA 与环境审计一致。

需要执行的训练命令：

```bash
set -euo pipefail
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
test ! -e /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/phase_r/room_ru_30k
PURI_GSPLAT_PYTHON=/home/chenglong/Uncertain-Nerf/uncertain-nerf/.venv-gsplat153/bin/python bash scripts/train_puri_gs_ru.sh /home/chenglong/Uncertain-Nerf/uncertain-nerf/external/gsplat-v1.5.3-ru /home/chenglong/Uncertain-Nerf/uncertain-nerf/data/mipnerf360/360_v2/room /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/phase_r/room_ru_30k 6 /home/chenglong/Uncertain-Nerf/uncertain-nerf/external/dinov2 /home/chenglong/Uncertain-Nerf/uncertain-nerf/data/PURI-GS-assets/dinov2/dinov2_vits14_reg4_pretrain.pth /home/chenglong/Uncertain-Nerf/uncertain-nerf/data/PURI-GS-derived/semantic_features/room 30000 4 2>&1 | tee /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/phase_r-console/room_ru_30k.log
```

训练通过后执行：

```bash
set -euo pipefail
test ! -e /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/phase_r/room_ru_30k/independent_eval
/home/chenglong/Uncertain-Nerf/uncertain-nerf/.venv-gsplat153/bin/python run_puri_gs.py --config configs/puri_gs_ru_full30k.yaml --gsplat-dir /home/chenglong/Uncertain-Nerf/uncertain-nerf/external/gsplat-v1.5.3-ru --data-dir /home/chenglong/Uncertain-Nerf/uncertain-nerf/data/mipnerf360/360_v2/room --result-dir /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/phase_r/room_ru_30k/independent_eval --gpu 6 --data-factor 4 --checkpoint /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/phase_r/room_ru_30k/ckpts/ckpt_29999_rank0.pt 2>&1 | tee /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/phase_r-console/room_ru_eval.log
```

命令执行完成后应看到：训练/aux/曲线/五图齐全；39 张独立指标；DINO/head false。

日志位置：对应两个 `phase_r-console` 日志。

如何判断成功：同步骤 14，加上 CSV 39 行。

出现什么情况应立即停止：任一错误；不得开始参数变体或第二 seed。

是否需要 Git 提交：否。是否需要服务器 pull：否。

### 步骤 17：阶段 R 唯一汇总与停止

本步骤目的：用四个固定 30k run 执行附件唯一门禁，生成 Markdown 和 JSON 决策。

执行位置：Linux 服务器主项目。

需要打开的目录：`/home/chenglong/Uncertain-Nerf/uncertain-nerf/reports`。

需要检查的文件：四个标准 checkpoint、四组独立评测、两个 RU aux validation；目标报告必须不存在。

需要执行的命令：

```bash
set -euo pipefail
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
test ! -e /home/chenglong/Uncertain-Nerf/uncertain-nerf/reports/PHASE_R_PURI_GS_RU.md
test ! -e /home/chenglong/Uncertain-Nerf/uncertain-nerf/reports/phase_r_puri_gs_ru.json
/home/chenglong/Uncertain-Nerf/uncertain-nerf/.venv-gsplat153/bin/python tools/summarize_puri_gs_ru.py --android-b1 /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/phase_r/android_b1_30k --android-ru /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/phase_r/android_ru_30k --room-b1 /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/phase_r/room_b1_30k --room-ru /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/phase_r/room_ru_30k --output-dir /home/chenglong/Uncertain-Nerf/uncertain-nerf/reports
```

命令执行完成后应看到：唯一结论 `RU_RECONSTRUCTION_PASS` 或 `RU_RECONSTRUCTION_FAIL`；Android、Room、效率门逐项事实；paired bootstrap CI95；训练时长建议、DINO 用时、checkpoint 大小比和 mask 参数量记录。

日志位置：`reports/PHASE_R_PURI_GS_RU.md` 和 `reports/phase_r_puri_gs_ru.json`。

如何判断成功：汇总工具退出码 0，报告引用的测试图数量为 19/39，所有输入 commit/config 一致，结论只有两个允许值之一。

出现什么情况应立即停止：缺文件、逐图名称不同、数量不符、目标报告已存在、除零或指标非有限。不要手工补数。

是否需要 Git 提交：Codex复核报告后才判断是否达到 Git 里程碑 2，并给出一次图形化提交说明。

是否需要服务器 pull：否。

完成后必须停止：即使结论是 `RU_RECONSTRUCTION_PASS`，也不能自动实现或执行阶段 U。等待用户明确确认。

## 十、缓存回存本地 E 盘

Android/Room cache 完成并验证后，继续使用 Windows PowerShell `scp` 下载到：

```text
E:\7-DataSet\PURI-GS-derived\semantic_features\android
E:\7-DataSet\PURI-GS-derived\semantic_features\room
```

下载后核对两个 `manifest.json` 和每图 `.pt` 数量。它们不进入 Git。原始数据仍只在 `E:\7-DataSet`，服务器只保留当前实验所需副本。

## 十一、环境变更说明

修改原因：训练期需要官方 DINOv2 ViT-S/14 registers 源码与权重。

新增依赖：只增加官方 DINOv2 源码目录和单一 `.pth` 权重资产；代码没有修改 requirements，也没有安装第二个 backbone。

对已有结果的影响：无。B1、A1、CVTR、Phase 4A 历史 checkpoint 和报告均不修改；独立推理不加载 DINO。

是否需要重跑：历史结果不重跑；只运行协议规定的四个新 30k run。

回滚方式：不再使用 `external/dinov2`、DINO weight、feature cache 和隔离 smoke 目录即可；不需要降级 PyTorch/CUDA/gsplat，因为这些环境从未改变。不要在未确认路径前执行递归删除。

## 十二、最终本地自检记录

此处应在交付前由 Codex最终更新：

- 分支：`dev`
- commit：`0ae62b10ef4d69c35cfcc1aafdfab37f37bd98fe`
- Python compileall：通过
- Shell `bash -n`：通过
- 定向 RU tests：24 passed
- 全量 pytest：175 passed、1 skipped；里程碑 1 前最终复测为 43.56 秒
- `git diff --check`：通过，仅有 Windows LF/CRLF 提示，无空白错误
- RU patch 对 base source apply/reverse：通过
- 本地 CUDA smoke：未执行；本机 GPU 架构与固定环境不兼容，不静默升级
- L20 B1 base 与 RU-superset 双 100-step：通过；两者 Gaussian 数均为 112790
- L20 B1 独立指标回归：通过；PSNR 差 `+0.00006485`，SSIM 差 `-0.00000179`，LPIPS 差 `0`
- L20 DINO/gsplat 环境及单图前向：通过；状态 `PURI_GS_RU_ENVIRONMENT_READY`
- xFormers：未安装；官方 DINO 发出可选加速警告后使用兼容路径正常完成前向，不作为失败，不改环境
- L20 Android DINO 缓存：通过；122 个 payload 全部加载，coarse/fine 分别为 `[384,16,16]`、`[384,36,36]` 且有限
- L20 RU 100-step CUDA 训练：通过；Gaussian 数 112790、DINO trainable 0、mask updates 100、梯度隔离运行时检查 100、四条曲线各 100 行、五张代表图齐全
- L20 RU 标准 checkpoint 独立评测：通过；19 张测试图，PSNR 17.742794、SSIM 0.663716、LPIPS 0.560165、36.1501 FPS、Gaussian 数 112790
- L20 RU 独立推理隔离：通过；standard checkpoint load true、imported DINO false、loaded mask head false、rasterization ratio 1.0
- Git 里程碑 1：已达到，等待用户图形化 commit/push
- 正式 30k：未启动
- 阶段 U：未实现、未启动
