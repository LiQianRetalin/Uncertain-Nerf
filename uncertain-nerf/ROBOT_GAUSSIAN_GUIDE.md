# PURI V8 机器人 Gaussian 路线

## 1. 当前决定

CamP + Zip-NeRF 已最终失败，完整数值见
`reports/CAMP_ZIPNERF_FINAL_FAILURE.md`。下一主 baseline 固定为：

> **gsplat 1.5.3 DefaultStrategy 的显式 3D Gaussian 地图/渲染核心，COLMAP 稀疏点
> 初始化，固定外部位姿，RGB-only 输入。**

这里选的是地图和渲染主干，不引入另一套大型 SLAM 前端。研究阶段位姿来自 RGB-only
SfM；真机阶段通过同一接口切换为 Fast-LIVO2 的公制相机位姿。项目名称继续使用
UncertainNeRF/PURI V8。

选择 gsplat 的原因是其 Apache-2.0 实现提供 CUDA Gaussian rasterization、标准
3DGS 复现脚本、较低显存和清晰的 Python/CUDA 边界。Splat-SLAM、MonoGS 和
Photo-SLAM 只作论文/结果参考，不把它们的整套跟踪、GUI、预训练深度网络或多进程
框架合并进本仓库。

参考：

- https://github.com/nerfstudio-project/gsplat
- https://github.com/hku-mars/FAST-LIVO2
- https://github.com/google-research/Splat-SLAM
- https://github.com/HuajianUP/Photo-SLAM
- https://github.com/jkulhanek/wild-gaussians

## 2. 不变的系统边界

- 当前算法只消费 RGB、内参和由 RGB SfM 得到的位姿，不使用深度、LiDAR 或 IMU
  监督，保证 RGB-only 结论成立。
- `v8_robot.pose_contract.PosePacket` 是唯一位姿入口。真机时仅把 `pose_source` 从
  `rgb_sfm` 改为 `fast_livo2`，并可携带 6x6 位姿协方差。
- Fast-LIVO2、Gaussian mapper 和规划器保持进程边界，通过时间戳、标定和 SE(3)
  消息连接；不把两套代码静态混编。这样也隔离 ROS/许可证/部署依赖。
- LiDAR 稀疏深度、IMU 和雷达接口必须保留，但默认关闭；启用它们属于后续融合实验，
  不能污染 RGB-only baseline。
- 前端不加载训练仪表盘、Web viewer 或 GUI。训练只写 JSONL/JSON 和必要图像，查看器
  单独按需启动。

## 3. 第一轮只做一个候选

固定版本 `gsplat==1.5.3`，使用官方 simple trainer 的 DefaultStrategy，不同时跑
MCMC、2DGS、Scaffold-GS 或其他变体。固定 SH degree 3；不增加渲染 MLP；不使用
ensemble、MC-dropout 或多次前向。这样若失败，结论可归因于这条主干，而不是候选
组合。

第一轮 Fern 门只读取官方 `stats/*.json` 中的 PSNR/SSIM/LPIPS。通过短筛后，完整
评测适配器必须再统一输出：

```text
rgb_*.png
gt_rgb_*.png
depth_*.npy
acc_*.npy
render_efficiency.json
training_summary.json
```

官方 trainer 原本只有 train/test，会把本项目三张 val 图混入训练。仓库中的固定补丁
`patches/gsplat_v1.5.3_robot_screen.patch` 将其改为与 A0 完全一致的 14/3/3 split，
并让最终 checkpoint 显式评测 test 而非 val。准备外部 checkout：

```bash
bash scripts/prepare_gsplat_robot_baseline.sh ./external/gsplat-v1.5.3
```

环境固定为独立 Python 3.10、PyTorch 2.4.0 + CUDA 12.1，以及 gsplat v1.5.3 官方
`pt24cu121-cp310` 预编译 wheel。这样和服务器 `nvcc 12.1` 一致，也不依赖现有
Python 3.13/PyTorch 2.13 环境。只安装 trainer 实际导入的依赖，不安装 bilagrid、
视频编码器或其他未启用模块。源码 checkout 只承载固定 trainer 和数据划分补丁，
不编译其中的 CUDA 源码。

服务器继续复用已经存在的数据目录 `./data/nerf_llff_data/fern`，数据不进入 Git。
在准备 gsplat 或安装新环境前，先用旧 `.venv-v7` 做只读预检：

```bash
bash scripts/check_gsplat_server.sh ./data/nerf_llff_data/fern 6
```

该命令必须报告 Fern `PASS`，并输出 L20、CUDA compiler、G++、Python 和 PyTorch
版本。gsplat 使用独立的 `.venv-gsplat153`；不得向 `.venv-v7` 安装 gsplat 或其
依赖。服务器若不能访问 GitHub，先把本地生成的单一离线包上传到项目 `tmp/`，核对
SHA-256 后解压；PyPI、PyTorch 和 Anaconda 依赖仍从各自可访问的官方源取得。

离线包就位后执行：

```bash
bash scripts/setup_gsplat_robot_env.sh 6
PURI_GSPLAT_REPOSITORY=./tmp/gsplat153-offline/gsplat-v1.5.3.bundle \
  bash scripts/prepare_gsplat_robot_baseline.sh ./external/gsplat-v1.5.3
```

环境脚本会核对官方 wheel 的 SHA-256，并执行一次 32x32 单 Gaussian GPU
rasterization、fused SSIM 和 LPIPS 自检；只有输出 `decision=PASS` 才能开始训练。
`fused_ssim` 编译被限定为 `_FORTIFY_SOURCE=2`，以兼容服务器较新的 glibc 头文件与
CUDA 12.1；该设置只作用于这个安装子进程，不改变系统编译器、CUDA 或运行时性能。

## 4. 三项短筛

所有门槛在 `configs/v8_robot_screen_limits.json` 中，固定 L20、640x480 runtime
分辨率。只允许完整训练一个候选；先完成短筛再决定是否进入五类正式评估。

| 短筛 | 数据与门槛 |
|---|---|
| 干净精度 | LLFF Fern factor=2、相同 14/3/3 split；PSNR >= 27.20 dB，SSIM/LPIPS 必须记录 |
| 鲁棒性 | NeRF On-the-go Patio High 的静态区域；PSNR >= 20 dB，floater pixel rate <= 0.10 |
| 机器人效率 | 640x480 render >= 30 FPS、p95 <= 33.4 ms；单次局部 map update p95 <= 200 ms；峰值显存 <= 4 GiB；模型 <= 512 MiB |

结构门槛同时要求：RGB-only、外部位姿接口、单次 raster pass、无渲染 MLP、SH
degree <= 3。填充 `configs/v8_robot_screen_metrics.example.json` 的副本后运行：

```bash
python run_v8_robot_gate.py \
  --metrics reports/gsplat_robot_screen.json \
  --output reports/gsplat_robot_screen_decision.json
```

任何硬门槛失败都不跑更长日程。若只缺报告字段，同样视为协议失败，不以人工判断补齐。

## 5. 通过短筛后的五类正式评估

短筛通过后，才评估以下五类，且每类只保留一个代表性协议：

1. 渲染：Fern PSNR/SSIM/LPIPS。
2. 几何与定位：有真值序列的 depth AbsRel/RMSE、ATE、RPE、跟踪失败率。
3. 实时性：渲染、局部更新、端到端消息延迟、显存和模型大小。
4. 鲁棒性：一个固定机器人压力包，覆盖瞬态遮挡、雨雾、稀疏采样和反光，不为罕见
   情况增加独立分支。
5. 不确定性：error correlation、NLL/ECE、AUSE、risk-coverage 和异常检测。

## 6. V8 最小架构

地图图元保留标准 position、scale、rotation、opacity 和 SH color，并只增加四个可
解释标量：认知、偶然、几何和外观不确定性。训练期可以用鲁棒特征或离线教师生成
监督，但部署时四个标量与 RGB/depth/alpha 在同一次 tile rasterization 中累积。

运行时风险由这四个标量在 mapper 后端组合。跟踪/融合可直接使用每像素权重；规划器
只接收一张风险图，不把四个内部通道和大量调试信息推到前端。这保留了分解能力，也
避免 UI 和下游接口变慢。

## 7. Fast-LIVO2 与硬件接口

Fast-LIVO2 提供带时间戳的 `T_world_camera`、可选 6x6 协方差和标定后的相机内参。
Gaussian mapper 异步消费关键帧；定位线程不等待地图优化。位姿协方差进入几何
不确定性，但 LiDAR 深度默认不参与 RGB-only 训练。

硬件化边界固定为：Gaussian 投影、tile binning、有界 per-tile 深度排序、alpha compositing
以及 RGB/depth/risk 通道累积。保持固定 SH 上限、无渲染 MLP、无运行时采样/ensemble，
便于后续定点化、流水线和 FPGA 原型；当前 CUDA 代码只作为行为参考，不直接作为
硬件实现。

## 8. 后续执行顺序

1. 在 L6 上安装固定 gsplat 版本并先做 Fern 1k step smoke。
2. smoke 正常后完成 Fern 短训练与三张 test 渲染；未达 27.20 立即停止。
3. 通过后才跑 Patio High 鲁棒短筛。
4. 通过后测固定分辨率 runtime、显存、模型大小和局部更新延迟。
5. 三项均通过后再实现 V8 的四标量不确定性和 Fast-LIVO2 适配器。

1k smoke 命令如下；通过后把最后一个参数改为 `30000`。脚本强制关闭 viewer 和
视频生成，仅保留训练、JSON 指标和 checkpoint。

```bash
PURI_GSPLAT_PYTHON=./.venv-gsplat153/bin/python \
  bash scripts/train_gsplat_robot_baseline.sh \
  ./external/gsplat-v1.5.3 ./data/nerf_llff_data/fern \
  ./logs-gsplat/fern_default_seed0 6 1000

PURI_GSPLAT_PYTHON=./.venv-gsplat153/bin/python \
  bash scripts/evaluate_gsplat_robot_baseline.sh \
  ./external/gsplat-v1.5.3 ./data/nerf_llff_data/fern \
  ./logs-gsplat/fern_default_seed0 6 \
  ./logs-gsplat/fern_default_seed0/ckpts/ckpt_999_rank0.pt
```
