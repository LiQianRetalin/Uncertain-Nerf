# uncertain-nerf-v6：方法、训练、推理与一区 SCI 实验指南

本文面向第一次训练 NeRF 的使用者。请按章节顺序操作，不要一开始同时修改大量参数。

## 1. V6 要解决什么问题

V5 的不确定性直接乘入密度，并参与 coarse-to-fine 重要性采样。真实但困难的细节可能先被判定为高不确定性，随后得到更少采样，最终进一步变差。V6 的核心原则是：

> 几何密度负责“空间中是否有物体”，不确定性负责“当前观测是否可信”，两者必须解耦。

V6 保留 V5 中有价值的部分：哈希编码、COLMAP 稀疏深度、鲁棒核、教师-学生不确定性和外观变化建模；但重新设计了它们的连接方式。

| V5 | V6 | 目的 |
|---|---|---|
| 两套完整 coarse/fine 哈希场 | coarse/fine 共享一个场 | 减少约一半哈希参数和优化器状态 |
| `sigma_eff = sigma * reliability(u)` | 透明度只使用 `sigma` | 消除密度与不确定性的不可辨识性 |
| 不确定性控制 fine 采样 | fine 采样只依据物理密度权重 | 防止难细节被自我抑制 |
| 最后区间为 `1e10` | near/far 之间的有限 Voronoi 区间 | 避免 AABB 边界不透明壳层 |
| ReLU 密度 | 带负偏置的 softplus 密度 | 避免初始化产生死密度或随机壳层 |
| 主干始终 dropout | 确定性几何/颜色主干 | 避免细节抖动和训练/推理分布差异 |
| MC-dropout 重复渲染 | 表面多视图分歧 + 射线终止熵 | 降低教师计算量 |
| 为每条教师射线复制完整源图 | 直接双线性读取 4 个邻近像素 | 显著降低显存和带宽 |
| 可学习教师融合权重 | 固定、可消融的教师融合比例 | 避免教师迎合学生的循环退化 |
| 二阶坐标梯度平滑 | 表面邻域有限差分 | 避免 `create_graph=True` 的高开销 |
| 每张图一个常量背景 | 全场景统一终端背景 + appearance code | 保证训练和新视角背景一致 |
| 自动损失数值反比标定 | 含义明确的固定系数 | 便于消融和论文复现 |

## 2. V6 数学形式

### 2.1 场表示

对空间点 `x`、观察方向 `d` 和训练图像外观编码 `a_j`，网络输出：

```text
(rgb_raw, sigma_raw, u_logit) = F_theta(x, d, a_j)
sigma = softplus(sigma_raw + density_bias)
u = sigmoid(u_logit)
```

`sigma` 不依赖 `u`。推理到测试视角或自定义轨迹时，不使用未训练的测试图编码，而使用训练图 appearance code 的均值。

### 2.2 有限区间体渲染

V6 在射线与扩展 AABB 的有效交段内采样。默认使用逆深度采样，以便给靠近相机的细小结构更多采样点。

```text
alpha_i = 1 - exp(-sigma_i * delta_i)
T_i = product_{j<i}(1-alpha_j)
w_i = T_i * alpha_i
C = sum_i w_i*c_i + (1-sum_i w_i)*C_bg
```

`delta_i` 只覆盖真实 near/far 区间，不再使用 `1e10`。深度用归一化权重计算，避免低透明度时深度被错误拉向 0。

### 2.3 不确定性学习

V6 使用三个互补信号：

1. **异方差颜色似然**：预测风险对应颜色方差，误差项除以方差，同时保留 `log variance`，因此网络不能无代价地把不确定性全部增大。
2. **表面教师**：只在当前渲染表面点计算多视图颜色分歧，并先用每张图的颜色均值和标准差做曝光归一化。
3. **射线终止熵**：权重沿射线越分散或剩余透射率越大，几何越不确定。

教师目标直接位于 `[0,1]`，不再对非负量额外执行导致下界为 0.5 的 Sigmoid。

### 2.4 几何先验

COLMAP 深度仍使用 MAD 和伪 Huber。区别是几何置信度使用停止梯度：

```text
confidence = exp(-gamma * stop_gradient(u))
L_geo = mean(confidence * pseudo_huber(depth - scale*depth_prior))
```

这样不确定性可以降低坏深度的影响，但不能通过主动增大自身来逃避几何监督。

### 2.5 浮游物抑制

V6 加入 mip-NeRF 360 风格的 distortion loss，鼓励同一条射线的密度集中于紧凑表面，减少多层薄雾和 floaters。实现使用前缀和，复杂度为 `O(N)`。

### 2.6 总损失

```text
L = L_rgb
  + lambda_nll        * L_nll
  + lambda_geo        * L_geo
  + lambda_teacher    * L_teacher
  + lambda_spatial    * L_spatial
  + lambda_distortion * L_distortion
  + lambda_appearance * L_appearance
```

初始权重已经写入 `configs/llff_colmap_v6.txt`。这些权重是实验起点，不是论文结论；必须通过消融验证。

### 2.7 每次迭代的完整算法

```text
输入：训练图像、相机、COLMAP 稀疏深度、当前步数 g

1. 分层抽取一批训练射线：至少 prior_ray_ratio 来自有效 COLMAP 像素。
2. 射线与扩展 AABB 求交，在有效 near/far 内按逆深度分层采样。
3. 共享场输出颜色、物理密度和观测不确定性。
4. 仅用物理密度计算 coarse 权重；按 coarse 权重抽取 fine 样本。
5. 合并 coarse/fine 样本，用有限区间重新渲染 RGB、深度、透明度和不确定性。
6. 每一步计算 RGB MSE、异方差 NLL、COLMAP 几何损失和 distortion loss。
7. 每 teacher_interval 步：
   a. 从有效射线抽取 teacher_ray_ratio 子集；
   b. 只取每条射线的期望表面点；
   c. 投影至四个邻近源视图，直接读取所需像素；
   d. 将鲁棒光度分歧与射线终止熵融合成停止梯度的教师目标；
   e. 计算学生蒸馏损失和表面邻域有限差分。
8. 按固定、可解释的 lambda 系数组合总损失。
9. AMP 反向传播、梯度裁剪、Adam 更新和 warmup-cosine 学习率更新。
10. 周期性在独立验证集计算 PSNR；测试集不参与调参或 time-to-PSNR。

输出：共享辐射场、训练图 appearance codes、全局终端背景和校准风险图。
```

## 3. 代码目录

```text
run_nerf_v6.py                 V6 训练和推理入口
configs/llff_colmap_v6.txt     默认配置
v6/model.py                    解耦场网络
v6/rendering.py                有限区间体渲染、采样和 distortion loss
v6/teacher.py                  低开销表面教师
v6/losses.py                   异方差 NLL 和停止梯度几何损失
v6/trainer.py                  训练、验证、检查点和渲染
v6/evaluation.py               图像、几何、伪影、不确定性和效率评测
v6/pointcloud.py               深度反投影点云
v6/aggregate.py                三随机种子均值与 95% 置信区间
v6/compare.py                  多方法论文表格 CSV
scripts/*.sh                   Ubuntu 操作脚本
```

V5 检查点不能加载到 V6，因为网络结构和物理积分均已改变。

## 4. Ubuntu 环境安装

### 第 1 步：检查 NVIDIA 驱动

```bash
nvidia-smi
```

如果命令不存在或没有显示显卡，先安装 NVIDIA 驱动。不要在此时开始训练。

### 第 2 步：进入项目

```bash
cd /你的路径/1-UncertainNerf/uncertain-nerf
```

### 第 3 步：创建环境

兼容性优先、使用纯 PyTorch 哈希编码：

```bash
bash scripts/setup_v6_ubuntu.sh
```

速度优先、同时编译 tiny-cuda-nn：

```bash
nvcc --version
INSTALL_TCNN=1 bash scripts/setup_v6_ubuntu.sh
```

tiny-cuda-nn 需要完整 CUDA Toolkit 和 `nvcc`，只有显卡驱动与 PyTorch CUDA runtime 不够。如果 `nvcc` 不存在，先按 NVIDIA 官方方法安装与 PyTorch CUDA 版本兼容的 Toolkit；不要盲目混装多个 CUDA 主版本。

如果你的驱动不支持默认 CUDA wheel，先到 PyTorch 官方安装页选择对应索引，例如：

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
  bash scripts/setup_v6_ubuntu.sh
```

安装完成必须看到：

```text
CUDA available: True
GPU: 你的显卡名称
```

### 第 4 步：激活环境

每次打开新终端都执行：

```bash
source .venv-v6/bin/activate
```

### 第 5 步：运行测试

```bash
pytest -q
```

测试通过只能证明程序接口和基本数学性质正常，不能证明重建质量好。

## 5. 数据准备

每个场景目录必须是：

```text
data/leaves/
├── images/
│   ├── IMG_0001.png
│   └── ...
├── poses_bounds.npy
└── sparse/0/
    ├── cameras.bin
    ├── images.bin
    └── points3D.bin
```

要求：

1. `poses_bounds.npy` 和 `sparse/0` 必须来自同一次 COLMAP 重建。
2. 图片名称必须与 COLMAP 的 `images.bin` 一致。
3. 第一轮实验不要同时改变图像、COLMAP 参数和 V6 参数，否则无法判断问题来源。
4. 论文最终实验必须固定训练/验证/测试划分。

V6 默认先按 `llffhold=8` 留出测试视角，再从剩余候选训练视角中按 `val_every=8` 留出独立验证视角。验证视角不会进入训练射线、COLMAP 先验或教师源视图。数据量很少时应重新设计固定划分并如实报告，不能把测试集当验证集调参。

## 6. 第一次训练：先做冒烟测试

冒烟测试只检查能否运行，不用于比较质量：

```bash
source .venv-v6/bin/activate
CUDA_VISIBLE_DEVICES=0 python run_nerf_v6.py \
  --config configs/llff_colmap_v6.txt \
  --datadir ./data/leaves \
  --expname leaves_v6_smoke \
  --seed 0 \
  --factor 4 \
  --N_rand 128 \
  --N_samples 16 \
  --N_importance 16 \
  --N_iters 20 \
  --i_print 1 \
  --i_eval 0 \
  --i_weights 20 \
  --no_reload
```

成功标准：

- 能完成 20 步；
- `loss/psnr/nll/geo` 均为有限数；
- 出现 `logs-v6/leaves_v6_smoke/checkpoints/latest.pt`。

## 7. 正式训练

### 7.1 选择哈希后端

如果已经成功安装 tiny-cuda-nn，把配置改成：

```text
hash_backend = tcnn
```

否则保持：

```text
hash_backend = torch
```

纯 PyTorch 后端用于兼容和验证；论文效率实验应优先使用同一台机器上的 tcnn 后端，并对所有 V6 消融保持一致。

### 7.2 单随机种子训练

```bash
bash scripts/train_v6.sh ./data/leaves leaves 0 0
```

五个参数依次表示：数据目录、场景名、随机种子、GPU编号、可选配置文件。

训练结果位于：

```text
logs-v6/leaves_v6_seed0/
├── args.json
├── config.txt
├── train_metrics.jsonl
├── training_summary.json
└── checkpoints/
```

### 7.3 中断后继续训练

不要加 `--no_reload`，程序会自动读取 `latest.pt`：

```bash
CUDA_VISIBLE_DEVICES=0 python run_nerf_v6.py \
  --config configs/llff_colmap_v6.txt \
  --datadir ./data/leaves \
  --expname leaves_v6_seed0 \
  --seed 0
```

### 7.4 最终高分辨率训练

默认 `factor=2` 适合开发。论文最终结果建议另开实验使用：

```bash
CUDA_VISIBLE_DEVICES=0 python run_nerf_v6.py \
  --config configs/llff_colmap_v6.txt \
  --datadir ./data/leaves \
  --expname leaves_v6_fullres_seed0 \
  --seed 0 \
  --factor 1 \
  --hash_backend tcnn \
  --no_reload
```

不要用 factor=2 的检查点继续 factor=1；数据指纹和网络空间分辨率的实验条件不同，应重新训练并单独记录。

### 7.5 三随机种子

```bash
bash scripts/train_v6_three_seeds.sh ./data/leaves leaves 0
```

这会依次训练 seed 0、1、2。显存允许且有多张显卡时可在三个终端分别指定 GPU，但不要让多个训练进程争用同一张显卡。

## 8. 推理与渲染

### 8.1 渲染测试集

测试集渲染会额外保存 GT 图像，供 PSNR/SSIM/LPIPS 使用：

```bash
bash scripts/render_v6.sh ./data/leaves leaves 0 0 test
```

输出目录类似：

```text
logs-v6/leaves_v6_seed0/render_test_100000/
├── rgb_000.png
├── gt_rgb_000.png
├── depth_000.npy
├── uncertainty_000.npy
├── acc_000.npy
└── render_efficiency.json
```

### 8.2 渲染 LLFF 新视角轨迹

```bash
bash scripts/render_v6.sh ./data/leaves leaves 0 0 path
```

### 8.3 指定检查点

```bash
CUDA_VISIBLE_DEVICES=0 python run_nerf_v6.py \
  --config configs/llff_colmap_v6.txt \
  --datadir ./data/leaves \
  --expname leaves_v6_seed0 \
  --render_only \
  --render_split test \
  --ft_path ./logs-v6/leaves_v6_seed0/checkpoints/step_050000.pt
```

## 9. 截图中“必须报告”的指标如何完成

### 9.1 图像：PSNR、SSIM、LPIPS

先完成 test 渲染，然后执行：

```bash
python -m v6.evaluation \
  --pred_dir ./logs-v6/leaves_v6_seed0/render_test_100000 \
  --method uncertain-nerf-v6 \
  --seed 0 \
  --compute_lpips \
  --device cuda \
  --training_summary ./logs-v6/leaves_v6_seed0/training_summary.json
```

含义：

- PSNR 越高越好；
- SSIM 越高越好；
- LPIPS 越低越好。

所有方法必须使用完全相同的分辨率、测试图和颜色空间。不能让 V6 用 factor=1，而基线用 factor=4。

### 9.2 几何：AbsRel、RMSE、Chamfer、F-score

AbsRel 和深度 RMSE 需要每个测试视角的真值深度：

```text
ground_truth/leaves/depth/depth_000.npy
ground_truth/leaves/depth/depth_001.npy
...
```

真值深度必须与 V6 输出采用相同坐标尺度和相同“射线参数深度”定义。若 LiDAR 使用米，而 NeRF 使用归一化尺度，必须先用 COLMAP/LLFF 的世界变换转换，不能在评测脚本里偷偷对每张图单独拟合尺度。

导出预测点云：

```bash
python -m v6.pointcloud \
  --datadir ./data/leaves \
  --render_dir ./logs-v6/leaves_v6_seed0/render_test_100000 \
  --factor 2 \
  --device cuda \
  --output ./logs-v6/leaves_v6_seed0/predicted_points.npy
```

然后统一评测：

```bash
python -m v6.evaluation \
  --pred_dir ./logs-v6/leaves_v6_seed0/render_test_100000 \
  --method uncertain-nerf-v6 \
  --seed 0 \
  --compute_lpips --device cuda \
  --gt_depth_dir ./ground_truth/leaves/depth \
  --pred_pointcloud ./logs-v6/leaves_v6_seed0/predicted_points.npy \
  --gt_pointcloud ./ground_truth/leaves/lidar_points.npy \
  --fscore_threshold 0.01
```

`fscore_threshold` 必须按场景统一尺度预先确定并写进论文，例如归一化场景尺度的 1%，不能根据每个方法的结果选择。

### 9.3 伪影：纯背景误透明度、floaters 像素率

需要为测试图准备二值纯背景掩码：

```text
background_mask_000.png
background_mask_001.png
...
```

白色表示确定没有目标/几何的纯背景，黑色表示不参与背景伪影统计。把掩码复制到对应 render 目录。

V6 定义：

```text
纯背景误透明度 = 背景掩码内 acc 的均值
floaters 像素率 = 背景掩码内 acc > threshold 的像素比例
```

默认 `threshold=0.1`。论文中必须固定阈值，并补充若干可视化结果。对于没有“纯背景”的场景，该指标应标为 N/A，不能任意把远景树木标为背景。

### 9.4 不确定性：NLL、AUSE/AURG、risk-coverage、错误检测 AUROC、校准误差

`v6.evaluation` 会自动读取：

- `rgb_*.png`；
- `gt_rgb_*.png`；
- `uncertainty_*.npy`。

输出包括：

- `NLL`：异方差高斯负对数似然，越低越好；
- `AUSE`：模型稀疏化曲线与理想 oracle 的面积差，越低越好；
- `AURG`：相对随机剔除带来的收益，越高越好；
- `AURC`：risk-coverage 曲线下面积，越低越好；
- `bad_pixel_AUROC`：把误差最高 20% 像素作为错误像素，不确定性检测它们的 AUROC，越高越好；
- `UCE`：预测 RMSE 与经验 RMSE 的分箱校准误差，越低越好。

同时生成：

```text
risk_coverage.csv
risk_coverage_calibration.csv
```

论文中至少画两张图：risk-coverage 曲线和 predicted-RMSE/empirical-RMSE 校准曲线。

### 9.5 效率：达到指定 PSNR 的时间、总训练时间、峰值显存、FPS

训练器自动记录：

```text
training_summary.json
```

字段包括：

- `time_to_target_psnr_seconds`：验证集达到 `target_psnr` 的纯优化时间；
- `total_training_seconds`：包含周期验证的总墙钟时间；
- `optimization_seconds_excluding_validation`：扣除验证的优化时间；
- `peak_gpu_memory_mib`：PyTorch 峰值显存。

推理自动记录：

```text
render_efficiency.json
```

字段包括 FPS 和 rays/s。论文比较时必须：

1. 使用同一 GPU、驱动和 CUDA 环境；
2. 使用相同输出分辨率；
3. 明确是否包含模型加载和磁盘写入；V6 的 `fps` 使用渲染与 CPU 取回时间，不含模型加载和 PNG/视频写入，完整含 I/O 时间另存为 `render_wall_seconds_with_io`；
4. target PSNR 必须在实验开始前固定；
5. 所有方法采用相同验证集合。

### 9.6 至少三个随机种子和 95% 置信区间

分别完成三个 seed 的渲染和评测，得到三个 `metrics.json`，然后：

```bash
python -m v6.aggregate \
  --metrics \
    logs-v6/leaves_v6_seed0/render_test_100000/metrics.json \
    logs-v6/leaves_v6_seed1/render_test_100000/metrics.json \
    logs-v6/leaves_v6_seed2/render_test_100000/metrics.json \
  --output results/leaves/uncertain-nerf-v6_summary.json
```

程序使用 Student-t 置信区间，输出 JSON 和 CSV。三个记录的 seed 必须不同，否则程序拒绝汇总。

### 9.7 与七种关键方法比较

必须至少比较：

1. Instant-NGP；
2. Mip-NeRF 360；
3. Zip-NeRF；
4. RobustNeRF；
5. NeRF-W；
6. Bayes' Rays；
7. 原始 NeRF 或本仓库 V5。

公平协议：

- 使用同一原始图像和相机参数；
- 使用同一 train/test split；
- 每种方法至少三个随机种子；
- 图像指标使用同一脚本重新计算，而不是抄论文数字；
- 没有不确定性输出的方法，在不确定性表格标记 N/A；
- Bayes' Rays 是 post-hoc 方法，训练时间和后处理时间应分开报告；
- RobustNeRF 是动态干扰/floaters 关键基线；
- 所有方法记录实际 GPU 时间和显存，不能把不同论文不同 GPU 的时间放入同一张公平对比表。

每种外部方法渲染后，把文件整理成 V6 评测命名：

```text
rgb_000.png
gt_rgb_000.png
depth_000.npy              如果方法输出深度
uncertainty_000.npy        如果方法输出不确定性
acc_000.npy                如果方法输出透明度
```

然后使用 `v6.evaluation` 和 `v6.aggregate`。最后合并多方法汇总：

```bash
python -m v6.compare \
  --summaries \
    results/leaves/instant-ngp_summary.json \
    results/leaves/mipnerf360_summary.json \
    results/leaves/zipnerf_summary.json \
    results/leaves/robustnerf_summary.json \
    results/leaves/nerfw_summary.json \
    results/leaves/bayesrays_summary.json \
    results/leaves/uncertain-nerf-v6_summary.json \
  --output results/leaves/method_comparison.csv
```

外部基线需要分别按照其官方仓库安装；本仓库没有复制或伪造这些实现。

## 10. 论文消融实验顺序

不要只报告“完整 V6 比 V5 好”。至少做以下逐步实验：

| 编号 | 设置 | 要回答的问题 |
|---|---|---|
| B0 | 原始 V5 | 当前问题基线 |
| B1 | V6 渲染器，关闭所有 V6 辅助损失 | 有限区间、softplus、共享场是否解决壳层和速度问题 |
| B2 | B1 + appearance code | 曝光建模是否减少颜色残差 |
| B3 | B2 + NLL | 异方差似然是否提高质量和校准 |
| B4 | B3 + teacher | 表面教师是否提升不确定性排序 |
| B5 | B4 + geometry | 深度先验是否提升几何 |
| B6 | B5 + distortion | 是否降低 floaters |
| B7 | 完整 V6 + tcnn | 加速后是否保持质量 |

B1 可使用：

```bash
python run_nerf_v6.py ... \
  --lambda_nll 0 \
  --lambda_geo 0 \
  --lambda_teacher 0 \
  --lambda_spatial 0 \
  --lambda_distortion 0 \
  --lambda_appearance 0 \
  --teacher_ray_ratio 0
```

每加入一个模块，保持其他配置、随机种子、训练步数和数据划分不变。

## 11. 建议的论文表格

### 表 1：新视角质量

```text
Method | PSNR↑ | SSIM↑ | LPIPS↓
```

### 表 2：几何与伪影

```text
Method | AbsRel↓ | RMSE↓ | Chamfer↓ | F-score↑ | False opacity↓ | Floater rate↓
```

### 表 3：不确定性质量

```text
Method | NLL↓ | AUSE↓ | AURG↑ | AURC↓ | Error AUROC↑ | UCE↓
```

### 表 4：效率

```text
Method | Time-to-PSNR↓ | Optimization time↓ | Peak VRAM↓ | FPS↑
```

每个单元格写成 `均值 ± 95% CI`，并在表注中列出三个种子、GPU、分辨率、训练步数和 F-score/floater 阈值。

## 12. 一区 SCI 主线建议

建议论文主线命名为：

> Decoupled Risk-Calibrated Radiance Fields for Robust Robotic Reconstruction under Degraded Visibility

论文贡献不要写成“加入了七个模块”，而应围绕一个因果链：

1. 退化观测会让传统 NeRF 的观测风险与几何密度纠缠；
2. 这种纠缠造成细节自抑制和空背景 floaters；
3. V6 用解耦渲染、可校准风险似然和表面教师打断该闭环；
4. 在烟尘、反光、低照度和遮挡数据上，同时改善几何、伪影和不确定性校准；
5. 风险输出可以用于机器人下一视角选择或安全阈值决策。

仅在 LLFF 上提高 PSNR 通常不足以支撑该主线。建议建立或采集真实机器人退化数据，并提供同步 LiDAR/高质量离线扫描真值。至少设置：正常、低照度、反光、轻烟尘、重烟尘、动态遮挡六类条件。

## 13. 当前 V6 的边界

- V6 已实现可选 tcnn 哈希编码，但没有把整个渲染器改成 fully-fused CUDA，也没有 occupancy-grid 空空间跳过；因此速度会明显优于 V5 教师路径，但不等同于 Instant-NGP 的极限速度。
- 表面教师进行了曝光归一化和鲁棒聚合，但没有真实源视图深度时，无法完成严格的 z-buffer 遮挡判断。拥有 LiDAR/MVS 深度时，应进一步加入前后向深度一致性。
- V6 使用点采样哈希网格，没有完整复现 Zip-NeRF 的锥体抗混叠。Zip-NeRF 必须作为细节/混叠基线。
- 任何“达到一区”的结论都必须由数据、对比、消融、统计显著性和机器人任务验证共同支持；代码结构本身不能保证发表等级。
