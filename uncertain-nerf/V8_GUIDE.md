# PURI-NeRF V8 第一批实施与 L6 执行手册

> **归档通知（2026-08-27）：** 本文是旧 NeRF 诊断链的执行记录。CamP + Zip-NeRF
> 最终 aligned PSNR 为 17.4028 dB，路线已完整淘汰。不要再按本文启动 baseline、
> V7.5、CamP 或 Zip-NeRF；当前入口为 `ROBOT_GAUSSIAN_GUIDE.md`。
> 本文其余“允许的后续步骤”均为历史说明，不再代表当前授权。

本文只覆盖已获准的第一批：仓库审计、V5–V7 冻结、无 uncertainty baseline、
V7.5 旁路 UQ。当前禁止进入 SDF、反射、参与介质或 post-hoc UQ。

## 1. 当前边界与停止门

- 本地分支保持 `dev`，没有创建或切换分支；
- V5、V6、V7 源码、入口和历史配置不修改；
- baseline 不创建可学习 uncertainty 参数；
- V7.5 的 uncertainty head 读取 detached hidden，使用独立 optimizer；
- 两组共用相同 reconstruction 初始化、固定 ray sampler、相机、split、loss、
  学习率、训练步数和 reconstruction 随机数流；UQ 不进入密度、RGB、几何、采样、
  背景或曝光；
- 训练第 1 步及此后每 100 步自动检查
  `d L_UQ / d theta_reconstruction = 0`，泄漏会以
  `STOP_UQ_GRADIENT_LEAK` 立即中止，并写入当前实验的
  `reports/STOP_UQ_GRADIENT_LEAK.md`；
- baseline 若在复核数据、相机、bounds 和评测后仍只有约 22 dB，填写
  `reports/STOP_BASELINE_FAILURE.md` 并停止，不训练 V7.5；
- baseline 通过后才运行 V7.5；二者 test PSNR 差值必须不超过 0.3 dB，
  否则填写 `reports/STOP_V75_FAILURE.md`，不进入 V8-Core。

`reports/REPO_AUDIT.md` 记录了一个重要事实修正：V6/V7 已不再用 UQ 缩放
density；V6 的未隔离损失/几何降权和 V7 的 UQ 自适应采样仍会影响重建，但不能再把
V7 的下降简单表述为“density 前向门控”。

## 2. 数据集决定

第一批只需要 fern。已经核对的本地目录是：

```text
E:\7-DataSet\nerf数据集\nerf_llff_data\fern
```

它包含 LLFF 图像、`poses_bounds.npy` 和 `sparse/0`，足以运行 baseline/V7.5。

- RawNeRF：不阻塞第一批；它针对 RAW/HDR/低照成像，当前 V8 第一版没有 RAW
  相机响应模型；
- RobustNeRF：不阻塞第一批；它用于 distractor 鲁棒性压力测试；
- NeRF On-the-go：压缩包已在本地，可作为后续动态干扰压力测试，但 V8 第一版明确
  不做动态实体分支；
- Ref-NeRF synthetic/real：压缩包已在本地，后续反射阶段使用；
- ScatterNeRF：真实参与介质阶段仍缺失，见第 12 节。

现在不需要继续等待 Raw/Robust，也不需要把这些大压缩包上传服务器。只有 fern 在
第一批训练范围内。

## 3. 本地图形化 Git：先冻结，再提交，再推送

你的常用本地终端位置
`PS E:\6-Project\1-UncertainNerf\uncertain-nerf>` 是正确的 Python 项目目录；
Git 仓库根目录实际是它的上一级 `E:\6-Project\1-UncertainNerf`。图形化客户端应打开
这个上一级根目录（即包含 `.git`、`uncertain-nerf/` 和 `训练及渲染结果/` 的目录）。
以下 Git 动作全部在图形化客户端完成，不输入 Git 命令。

### 3.1 冻结 V7 历史提交

1. 打开图形化 Git 客户端，确认当前分支显示 `dev`；不要切换分支。
2. 打开 History/历史，找到提交
   `e02df19db1b3149c95e7b2507e49911313aeecbd`。
3. 对该历史提交创建 annotated tag：`v7-fern-seed0-100k`。
4. 将这个 tag 推送到远程。不要移动已有 V5/V6 tag。

### 3.2 检查 Changes

Changes 中允许出现：

- `.gitignore`、`README.md`、`V8_GUIDE.md`、`requirements-v8.txt`；
- `run_nerf_v8.py`、`v8/`、两个 `configs/v8_*.txt`；
- 四个 `scripts/*v8*`；
- 两个 `tests/test_v8_*.py`；
- `reports/` 下四个 V8 审计/报告文件。

必须排除：

- `训练及渲染结果/`（这是你的本地资产，保持 untracked，不勾选）；
- 任何 `data/`、`.venv-*`、`logs*`、checkpoint、图片或视频；
- V5/V6/V7 源文件、三个历史入口和三个历史配置。

逐文件检查后建议创建两个普通 commit：

1. 只勾选 `reports/REPO_AUDIT.md` 和
   `reports/HISTORICAL_VERSION_MANIFEST.md`，提交说明：
   `Freeze V5-V7 identities and add repository audit`
2. 勾选其余 V8 第一批文件，提交说明：
   `Add matched V8 baseline and V7.5 diagnostic gate`

第二个 commit 完成后再 Push 到远程 `dev`。不要 force push。推送完成后在历史页确认
本地 `dev` 与远程 `origin/dev` 指向同一最新提交。

## 4. 远程下拉前先归档历史 checkpoint

如果尚未登录服务器，从本地 PowerShell 连接时使用已核对的账号和地址：

```powershell
ssh -o ClearAllForwardings=yes chenglong@172.16.55.2
```

`ClearAllForwardings` 用于避开历史 SSH 配置中本地/远程 17890 端口转发冲突，不改变
训练网络。若你已经像本次一样看到
`(base) chenglong@lc-NF5468-M7-A0-R0-00:~/Uncertain-Nerf$`，说明已登录，无需重复连接。

服务器初始提示符在 Git 根目录 `~/Uncertain-Nerf` 时，先进入 Python 项目目录：

```bash
cd ~/Uncertain-Nerf/uncertain-nerf
pwd
```

先只查找，不移动、不删除：

```bash
find ./logs ./logs-v6 ./logs-v7 -type f -path '*/checkpoints/*.pt' \
  -printf '%p\t%s bytes\n' 2>/dev/null
```

把输出保存并发回本任务，以便把 V5/V6/V7 的确切最优 checkpoint 写入冻结清单。
确定文件后，为每一个历史版本复制到仓库外的归档目录并计算 SHA-256；不要使用通配符
覆盖已有归档：

```bash
mkdir -p ~/Uncertain-Nerf-archive/v5-v7-20260820
cp -p <V5确切checkpoint路径> ~/Uncertain-Nerf-archive/v5-v7-20260820/
cp -p <V6确切checkpoint路径> ~/Uncertain-Nerf-archive/v5-v7-20260820/
cp -p <V7确切checkpoint路径> ~/Uncertain-Nerf-archive/v5-v7-20260820/
sha256sum ~/Uncertain-Nerf-archive/v5-v7-20260820/*.pt
```

`<...>` 必须替换成上一步真实路径，不能原样执行。归档只复制，不删除原文件。

## 5. 远程用图形界面 Fetch/Pull

只有本地两个 commit 已 push 且历史 checkpoint 已归档后才执行：

1. 用 VS Code Remote-SSH 打开 Git 根目录
   `/home/chenglong/Uncertain-Nerf`；运行 Python 命令时再进入其下的
   `uncertain-nerf`。
2. 在 Source Control 中确认当前分支是 `dev`。
3. 确认远程 Changes 为空；若有改动，先停止并核对，不能直接 Pull。
4. 点击 Fetch，再点击 Pull。
5. 在历史中确认最新提交与本地刚推送的提交一致。
6. 确认远程已出现 `run_nerf_v8.py`、`v8/`、`V8_GUIDE.md` 和四个 V8 脚本。

数据、日志和 checkpoint 不通过 Git 同步。

## 6. 核对 fern 是否已在服务器

在远程终端运行：

```bash
cd ~/Uncertain-Nerf/uncertain-nerf
find ./data -maxdepth 3 -name poses_bounds.npy -print 2>/dev/null
```

若已有 fern，继续核对：

```bash
test -d ./data/fern/images
test -f ./data/fern/poses_bounds.npy
test -d ./data/fern/sparse/0
find ./data/fern/images -maxdepth 1 -type f | wc -l
du -sh ./data/fern
```

若实际路径是 `./data/nerf_llff_data/fern`，后续命令把 `./data/fern` 全部替换为
该真实路径。若服务器没有 fern，用 VS Code 文件浏览器或 WinSCP 将本地整个 fern
目录上传到：

```text
/home/chenglong/Uncertain-Nerf/uncertain-nerf/data/fern
```

上传后重新执行上面四项核对。不要上传 zip，也不要把数据加入 Git。

## 7. L6 的准确含义与环境复用

本机 `nvidia-smi -L` 已确认有 8 张 L20，编号 0–7。这里的 L6 是物理 GPU index 6：

```text
GPU 6: NVIDIA L20
UUID: GPU-710a34fb-03e7-9d77-d8c6-66c1f3572635
```

命令必须使用 `CUDA_VISIBLE_DEVICES=6`。屏蔽后，PyTorch 进程内部把这张物理 6 号卡
重新编号成 `cuda:0`；看到 `torch.cuda.get_device_name(0)` 是正常的，并不代表用了
物理 0 号卡。

第一批 V8 复用 `.venv-v7`，原因是它只复用已验证的 V7 数据加载、HashGrid 和渲染
依赖，没有新增 CUDA 扩展。检查脚本不会安装或修改任何包：

```bash
cd ~/Uncertain-Nerf/uncertain-nerf
bash scripts/check_v8_environment.sh 6
```

必须同时看到：

- `python` 位于仓库的 `.venv-v7/bin/python`；
- `torch_cuda_available=True`；
- `torch_visible_device_count=1`；
- `logical_cuda_0=NVIDIA L20`；
- physical GPU 行明确为 index 6；显存使用不超过 1024 MiB、利用率不超过 5%。若
  L6 被其他任务占用，脚本会以 `STOP_GPU_BUSY` 退出，不会抢卡启动训练。

若失败，停止，不要直接 `pip install` 到旧环境；把完整输出发回本任务。我会根据缺失
项决定只补小依赖，还是创建独立 `.venv-v8`。到 SDF/新 CUDA 依赖阶段会重新评估，
不会默认沿用旧环境。

## 8. 自动化测试与 baseline smoke test

环境检查通过后先跑第一批测试：

```bash
cd ~/Uncertain-Nerf/uncertain-nerf
source .venv-v7/bin/activate
CUDA_VISIBLE_DEVICES=6 python -m pytest -q \
  tests/test_v8_diagnostic_core.py \
  tests/test_v8_diagnostic_trainer.py
```

测试必须全部通过。它会验证：baseline 无可学习 UQ、同 seed 的 reconstruction 完全
一致、改变 UQ 数值不改变 RGB/density/weights、UQ loss 对 reconstruction 零梯度、
checkpoint 恢复和诊断图生成。

然后只做 baseline 20 步 smoke：

```bash
CUDA_VISIBLE_DEVICES=6 python run_nerf_v8.py \
  --config configs/v8_baseline.txt \
  --mode baseline \
  --datadir ./data/fern \
  --expname fern_v8_baseline_smoke \
  --N_rand 128 \
  --N_samples 32 \
  --N_importance 16 \
  --N_iters 20 \
  --i_print 1 \
  --i_eval 0 \
  --i_weights 20 \
  --i_diagnostics 20 \
  --no_reload
```

smoke 验收：无 NaN/OOM/shape 错误；`loss_uq=0`、
`gradient_norm_uncertainty=0`、`u_mean=0`、`m_mean=1`；生成
`logs-v8/fern_v8_baseline_smoke/checkpoints/latest.pt` 和诊断目录。

任何一项失败都不要开始正式训练；发回测试和 smoke 的完整终端输出。

## 9. baseline 正式训练、监控与恢复

正式训练建议在 tmux 中运行：

```bash
tmux new -s v8-baseline
cd ~/Uncertain-Nerf/uncertain-nerf
bash scripts/train_v8_diagnostic.sh baseline ./data/fern fern 0 6
```

脚本固定从零开始并写入：

```text
logs-v8/fern_v8_baseline_seed0/
```

如果该目录已有 `latest.pt`，脚本会以 `STOP_EXISTING_RUN` 中止，防止覆盖。训练期间
另开终端监控物理 6 号卡：

```bash
watch -n 2 nvidia-smi -i 6
```

按 `Ctrl+B` 再按 `D` 可退出 tmux 而不中断训练；恢复查看：

```bash
tmux attach -t v8-baseline
```

若训练意外中断，不能再用正式训练脚本，因为它强制 `--no_reload`。使用同一配置、
数据、实验名和 seed 显式恢复：

```bash
cd ~/Uncertain-Nerf/uncertain-nerf
source .venv-v7/bin/activate
CUDA_VISIBLE_DEVICES=6 python run_nerf_v8.py \
  --config configs/v8_baseline.txt \
  --mode baseline \
  --datadir ./data/fern \
  --expname fern_v8_baseline_seed0 \
  --seed 0
```

每 100 步写终端/JSONL指标，每 5000 步验证，每 10000 步保存 checkpoint 和诊断图。
重点监控 `loss_rgb`、`loss_geometry`、`opacity_mean`、`depth_mean`、
`mean_weight_sum`、`mean_terminal_transmittance`、density/color/geometry 梯度范数、
GPU 时间和 peak memory。

## 10. baseline 全 split 渲染与评测

训练完成后依次渲染 train、val、test；这三个 split 使用各自真实相机和 GT：

```bash
bash scripts/render_v8_diagnostic.sh baseline ./data/fern fern 0 6 train
bash scripts/render_v8_diagnostic.sh baseline ./data/fern fern 0 6 val
bash scripts/render_v8_diagnostic.sh baseline ./data/fern fern 0 6 test
```

100k 默认输出目录是：

```text
logs-v8/fern_v8_baseline_seed0/render_train_100000
logs-v8/fern_v8_baseline_seed0/render_val_100000
logs-v8/fern_v8_baseline_seed0/render_test_100000
```

用同一评测器计算 train/val/test PSNR 和 SSIM；test 额外计算 LPIPS：

```bash
bash scripts/evaluate_v8_diagnostic.sh \
  ./logs-v8/fern_v8_baseline_seed0/render_train_100000 \
  puri-nerf-v8-baseline 0 6 \
  ./logs-v8/fern_v8_baseline_seed0/training_summary.json

bash scripts/evaluate_v8_diagnostic.sh \
  ./logs-v8/fern_v8_baseline_seed0/render_val_100000 \
  puri-nerf-v8-baseline 0 6 \
  ./logs-v8/fern_v8_baseline_seed0/training_summary.json

bash scripts/evaluate_v8_diagnostic.sh \
  ./logs-v8/fern_v8_baseline_seed0/render_test_100000 \
  puri-nerf-v8-baseline 0 6 \
  ./logs-v8/fern_v8_baseline_seed0/training_summary.json \
  --compute-lpips
```

将三个 `metrics.json`、`training_summary.json`、`train_metrics.jsonl`、最后一次
diagnostics、console log 和 checkpoint 单独归档。填写 `reports/BASELINE_REPORT.md`。

这里必须停下来评审 baseline。若复核无误后仍约 22 dB，不能启动 V7.5，也不能进入
SDF；先生成停止报告并排查相机/尺度/采样/颜色空间/曝光/评测。

## 11. baseline 通过后才运行 V7.5

得到明确的 baseline 通过结论后，才运行：

```bash
tmux new -s v8-v75
cd ~/Uncertain-Nerf/uncertain-nerf
bash scripts/train_v8_diagnostic.sh v7_5 ./data/fern fern 0 6
```

完成后按第 10 节把 `baseline` 换成 `v7_5`，渲染和评测三个 split。V7.5 还必须
核对每个 diagnostics 目录中的：

- `u_histogram.png`、`m_histogram.png`、`opacity_histogram.png`；
- `rgb_residual_vs_uncertainty.png`；
- `geometry_residual_vs_uncertainty.png`；
- `diagnostic_summary.json`。

填写 `reports/V75_DIAGNOSTIC_REPORT.md`。只有
`abs(test_PSNR_v7_5 - test_PSNR_baseline) <= 0.3 dB` 且没有
`STOP_UQ_GRADIENT_LEAK` 才通过。通过也只是允许设计 V8-Core 下一阶段，不代表自动
开始实施；开始 SDF 前再次向你确认。

## 12. ScatterNeRF 与替代选择

ScatterNeRF 官方数据需要通过 Princeton 页面注册并接受数据条款。公开代码仓库的
README 写有 sample 提示，但当前链接目标缺失；代码历史中也没有可恢复的公开数据
URL。因此我不能绕过注册或条款替你直接下载完整数据，也没有可信的匿名直链可用。

优先方案：通过官方注册获取受控数据。它含 clear ground truth、两种 fog density 和
scanner depth，最符合参与介质 clean/observed 分解验收。若页面仍打不开，可换浏览器、
无痕窗口或网络后访问官方注册链接；注册/登录步骤必须由你本人完成。下载完成后只需
告诉我目录，不要自行改格式。

工程替代方案：在进入介质阶段前，用已训练 clear scene 的深度按

```text
I_observed = I_clear * exp(-beta * depth)
           + A * (1 - exp(-beta * depth))
```

生成多组已知 `beta`/空气光 `A` 的 synthetic fog。它可验证渲染公式、浓度尺度、
梯度和 clean/observed 接口，但必须标注为 synthetic，只能作为工程回归测试，不能
替代真实雾数据支撑论文结论。

真实介质辅助集：SeaThru-NeRF 官方公开了约 594 MB 的水下散射多视角数据，具有
COLMAP/LLFF 风格的相机信息，可直接下载且不要求 ScatterNeRF 注册。它适合验证真实
散射介质下的 observed/clean 渲染趋势，优于只用 synthetic fog；但它没有
ScatterNeRF 受控集的同场景 clear/two-fog-density/scanner-depth 配对，而且水下颜色
吸收与空气雾并不完全相同。因此把它作为“真实介质辅助 benchmark”，不能替换
ScatterNeRF 的 paired clean/fog 主验收集。本地目标文件为：

```text
E:\7-DataSet\nerf数据集\SeathruNeRF_dataset.zip
```

已完成的本地校验记录：文件大小 `622647134` bytes，ZIP 共 126 entries、CRC 全部
通过，SHA-256 为
`FF97F52D547286D10B76210DC6EA54551BF8816CAF232094260FED73EC784254`。包含
Curasao、IUI3-RedSea、JapaneseGradens-RedSea、Panama 四个场景。第一批不需要解压或
上传它。

RawNeRF、RobustNeRF、On-the-go 和 Ref-NeRF 都不能替代 paired clear/fog + depth 的
ScatterNeRF 介质监督：它们分别解决 RAW 成像、distractor、移动干扰和反射问题。
因此当前选择是：第一批立即用 fern；反射阶段使用已下载 Ref-NeRF；介质代码在
baseline/V7.5 gate 通过后可先用 synthetic fog 做严格回归，再用 SeaThru-NeRF 做
真实介质辅助评估，而正式的 paired clear/fog 结论仍等待 ScatterNeRF。

## 13. 每轮修改的固定同步顺序

后续任何修复都遵循：本地 `dev` 修改 → 静态/可运行测试 → 图形化检查 diff →
图形化普通 commit → 图形化 Push 到远程 `dev` → 停止旧训练进程 → 远程图形化
Fetch/Pull → 新实验名 smoke/正式训练。服务器上不要直接编辑受 Git 管理的源码和配置，
不要 force push，不要把训练输出加入 Git。
