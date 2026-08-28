# PURI V8 机器人 Gaussian 路线

## 1. 结论与本轮纠正

CamP + Zip-NeRF 路线已经终止。机器人地图主干仍采用固定版本
`gsplat==1.5.3` 的 `DefaultStrategy`，即原始 3DGS 密度控制策略在 gsplat CUDA
实现上的复现。它提供显式 Gaussian 地图和实时 rasterization，适合后续 Fast-LIVO2
位姿接入、局部地图更新和 FPGA 算子拆分。

此前 Fern 30k 结果不能用于淘汰 3DGS。该实验使用 factor=2，并把标准 17 张训练图
再次拆出 3 张验证图，只剩 14 张训练图；同时把日志目录写成了 `seed0`，而固定训练器
实际使用 seed=42。这是实验协议失效，不是 Gaussian 表示失效。

新流程把两个问题严格分开：

1. **实现复现门 R0**：用 gsplat v1.5.3 官方报告过的 Mip-NeRF 360 Garden 检查环境、
   数据、划分和训练器是否正确。
2. **项目精度门 R1**：R0 通过后，再用标准 LLFF Fern factor=4、17 train / 3 test
   判断 DefaultStrategy 是否达到 PURI 的 27.20 dB 目标。

只有 R0 通过后，Fern 结果才有资格用于算法判断。R0 失败时先修复数据或复现协议，
不换算法。

## 2. 固定协议

| 项目 | R0：官方复现 | R1：项目判断 |
|---|---|---|
| 数据 | Mip-NeRF 360 Garden | LLFF Fern |
| 图像 | `images_4`，factor=4 | `images_4`，1008x756 |
| 划分 | 每 8 张取 1 张 test，其余全 train | 每 8 张取 1 张 test，17/3 |
| val | 不另设 val | 不另设 val |
| 方法 | gsplat 1.5.3 DefaultStrategy | 完全相同 |
| 随机种子 | 训练器固定 42 | 训练器固定 42 |
| 训练 | 30,000 steps | 30,000 steps |
| SH | degree 3 | degree 3 |
| 位姿/初始化 | 官方 COLMAP 与 SfM 稀疏点 | Fern COLMAP 与 SfM 稀疏点 |

训练脚本关闭 viewer、视频、TensorBoard 和训练中评测，但不改变优化、学习率、密度
控制或损失。训练完成后由独立脚本只评测 test。`val_every=0` 的补丁只是恢复 gsplat
原始的“非 test 图像全部用于训练”，不会引入新的算法分支。

## 3. R0 官方参考与自动门禁

gsplat v1.5.3 文档给出的 Garden 30k 单场景参考值为：

| PSNR | SSIM | LPIPS | Gaussian 数 |
|---:|---:|---:|---:|
| 27.32 | 0.865 | 0.075 | 5.84 M |

参考实验使用 TITAN RTX；L20 可以影响时间，但不应导致显著质量差异。自动复现门位于
`configs/gsplat153_garden_reproduction_limits.json`，允许小范围实现/硬件差异：
PSNR >= 26.82、SSIM >= 0.850、LPIPS <= 0.100、Gaussian 数 4.5M--7.0M。
门槛是复现异常报警线，不是修改论文参考值。
这里的 Gaussian 数只用于确认密度控制行为与官方一致，不是机器人部署的地图大小门槛；
部署大小仍由后续固定分辨率效率门单独约束。

训练结束后的官方 stats 文件直接送入门禁，不手抄指标：

```bash
./.venv-gsplat153/bin/python run_gsplat_reproduction_gate.py \
  --stats ./logs-gsplat/garden_default_seed42_full30k_test/stats/test_step29999.json \
  --output ./logs-gsplat/garden_default_seed42_full30k_test/reproduction_decision.json
```

若 R0 FAIL，不运行 Fern，也不运行其他候选。依次检查：数据目录/图像数量与尺度、
COLMAP 模型、commit 和 wheel、实际命令与 stats；修复后只重跑 Garden。

## 4. 数据边界

数据不进入 Git。本地数据总目录保持 `E:\7-DataSet`。R0 需要新增官方
Mip-NeRF 360 数据：

```text
E:\7-DataSet\nerf数据集\mipnerf360\360_v2\garden\
  images\
  images_4\
  sparse\0\
```

官方下载地址：

```text
https://storage.googleapis.com/gresearch/refraw360/360_v2.zip
```

压缩包约 11.67 GiB。完整压缩包只保存在本地 E 盘；向服务器仅传 `garden` 场景，
不上传其余六个场景。服务器目标目录固定为：

```text
/home/chenglong/Uncertain-Nerf/uncertain-nerf/data/mipnerf360/360_v2/garden
```

Windows PowerShell 下载与解压命令如下；`curl.exe -C -` 支持中断后续传：

```powershell
$mipRoot = 'E:\7-DataSet\nerf数据集\mipnerf360'
$extractRoot = Join-Path $mipRoot '360_v2'
$archive = Join-Path $mipRoot '360_v2.zip'

New-Item -ItemType Directory -Force -Path $extractRoot | Out-Null

curl.exe -L --retry 5 --retry-delay 5 -C - `
  'https://storage.googleapis.com/gresearch/refraw360/360_v2.zip' `
  -o $archive

(Get-Item -LiteralPath $archive).Length
tar.exe -xf $archive -C $extractRoot

$garden = Join-Path $extractRoot 'garden'
(Get-ChildItem -LiteralPath (Join-Path $garden 'images') -File).Count
(Get-ChildItem -LiteralPath (Join-Path $garden 'images_4') -File).Count
Get-Item -LiteralPath (Join-Path $garden 'sparse\0\cameras.bin')
Get-Item -LiteralPath (Join-Path $garden 'sparse\0\images.bin')
Get-Item -LiteralPath (Join-Path $garden 'sparse\0\points3D.bin')
```

压缩包字节数应为 `12535427936`，两项图像数都应为 `185`。为减少大量小文件传输，
本地只把 Garden 打成一个不重复压缩图片的 tar：

```powershell
tar.exe -cf (Join-Path $mipRoot 'garden.tar') -C $extractRoot garden
Get-Item -LiteralPath (Join-Path $mipRoot 'garden.tar')
```

把这个 `garden.tar` 上传到服务器项目的 `tmp/mipnerf360/`；它是数据传输文件，不进入
Git。上传完成后在服务器解包：

```bash
cd ~/Uncertain-Nerf/uncertain-nerf
mkdir -p ./data/mipnerf360/360_v2
tar -xf ./tmp/mipnerf360/garden.tar -C ./data/mipnerf360/360_v2
```

Fern 继续使用现有目录：

```text
/home/chenglong/Uncertain-Nerf/uncertain-nerf/data/nerf_llff_data/fern
```

## 5. R0 服务器命令

以下命令只在代码里程碑 push 一次、服务器 pull 一次之后执行。已经通过的环境安装和
外部 gsplat 准备不重做。

先检查代码和数据，不启动训练：

```bash
cd ~/Uncertain-Nerf/uncertain-nerf

git branch --show-current
git log -1 --oneline
git status --short

GARDEN=./data/mipnerf360/360_v2/garden

test -d "$GARDEN/images"
test -d "$GARDEN/images_4"
test -s "$GARDEN/sparse/0/cameras.bin"
test -s "$GARDEN/sparse/0/images.bin"
test -s "$GARDEN/sparse/0/points3D.bin"

printf 'images='
find "$GARDEN/images" -maxdepth 1 -type f | wc -l
printf 'images_4='
find "$GARDEN/images_4" -maxdepth 1 -type f | wc -l
```

两项图像数都应为 185。随后启动一次完整、官方可对照的 30k 训练：

```bash
cd ~/Uncertain-Nerf/uncertain-nerf

RESULT=./logs-gsplat/garden_default_seed42_full30k
test ! -e "$RESULT"
mkdir -p "$RESULT"

nohup bash -c '
PURI_GSPLAT_PYTHON=./.venv-gsplat153/bin/python \
bash scripts/train_gsplat_robot_baseline.sh \
./external/gsplat-v1.5.3 \
./data/mipnerf360/360_v2/garden \
./logs-gsplat/garden_default_seed42_full30k \
6 \
30000
train_exit=$?
printf "%s\n" "$train_exit" \
  > ./logs-gsplat/garden_default_seed42_full30k/train.exit
' > "$RESULT/train.log" 2>&1 < /dev/null &

printf '%s\n' "$!" > "$RESULT/train.pid"
printf 'train_pid='
cat "$RESULT/train.pid"
```

训练完成且 `train.exit=0` 后，单独评测 test：

```bash
cd ~/Uncertain-Nerf/uncertain-nerf

TRAIN_RESULT=./logs-gsplat/garden_default_seed42_full30k
TEST_RESULT=./logs-gsplat/garden_default_seed42_full30k_test
test ! -e "$TEST_RESULT"
mkdir -p "$TEST_RESULT"

PURI_GSPLAT_PYTHON=./.venv-gsplat153/bin/python \
bash scripts/evaluate_gsplat_robot_baseline.sh \
./external/gsplat-v1.5.3 \
./data/mipnerf360/360_v2/garden \
"$TEST_RESULT" \
6 \
"$TRAIN_RESULT/ckpts/ckpt_29999_rank0.pt" \
  > "$TEST_RESULT/eval.log" 2>&1

eval_exit=$?
printf 'eval_exit=%s\n' "$eval_exit"
tail -n 40 "$TEST_RESULT/eval.log"
```

最后运行 R0 门禁。只有输出 `decision: PASS` 才进入 R1。

## 6. R1 Fern 命令与判断

先重新执行 Fern 数据审计；新版输出必须包含 `images_4: 20` 和尺寸 1008x756：

```bash
cd ~/Uncertain-Nerf/uncertain-nerf
bash scripts/check_gsplat_server.sh ./data/nerf_llff_data/fern 6
```

然后用与 R0 相同的脚本训练 `fern_default_seed42_full30k`，并用独立 `_test` 目录评测。
R1 记录 PSNR、SSIM、LPIPS、Gaussian 数、checkpoint 大小和固定分辨率性能。

- `PSNR >= 27.20`：DefaultStrategy 进入鲁棒性与机器人效率短筛。
- `PSNR < 27.20`，但 R0 PASS：只能说明原始 DefaultStrategy 未达本项目 Fern 精度目标，
  不能淘汰显式 Gaussian 主干。

若出现第二种情况，只增加一个有官方消融依据、且不引入 MLP/多次前向的改进：
DefaultStrategy AbsGrad（`absgrad=true`、`grow_grad2d=0.0006`）。gsplat v1.5.3
官方消融中它相对 default 同时提高 PSNR/SSIM、降低 LPIPS、Gaussian 数、显存和训练
时间，符合机器人与 FPGA 方向。此改进只有在 R0 PASS、R1 default 完成后才实现，
不预先把多个候选塞进当前脚本。

## 7. 机器人与硬件边界

- 当前 baseline 仍是 RGB-only：图像、内参、RGB SfM 位姿和稀疏点。
- `v8_robot.pose_contract.PosePacket` 保持唯一外部位姿入口；真机阶段可切换到
  Fast-LIVO2 的 `T_world_camera` 和 6x6 协方差。
- LiDAR/IMU/雷达接口保留但默认关闭，不能污染 RGB-only 对照。
- Gaussian mapper 异步消费关键帧；定位线程不等待地图优化。
- FPGA 边界保持 Gaussian 投影、tile binning、有界排序、alpha compositing 和
  RGB/depth/risk 累积；不增加渲染 MLP、ensemble 或运行时采样。
- 不确定性最终必须参与跟踪加权、地图融合、图元增删、下一视角与规划，而不是只输出
  热力图。

## 8. Git 规则

本次协议修正完成本地测试后只做一次里程碑 commit/push，服务器随后只 pull 一次。
数据、日志、checkpoint 和外部 gsplat checkout 均不进入 Git。Garden 或 Fern 的训练
结果只回传 JSON 和关键日志，不为每次调试创建提交。下一次 Git 里程碑应当是 R0/R1
结论明确且确实需要实现 AbsGrad 或机器人评测适配器时。

参考：

- https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/docs/source/tests/eval.rst
- https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/examples/benchmarks/basic.sh
- https://github.com/graphdeco-inria/gaussian-splatting
- https://github.com/hku-mars/FAST-LIVO2
