# Uncertain-NeRF V5

这是基于 `nerf-pytorch` 的不确定性感知 NeRF 实现，面向具有反光、烟尘、遮挡、弱纹理和噪声深度先验的真实多视图场景。

主要改动包括：

- 纯 PyTorch 多分辨率哈希编码和小型 MLP；
- coarse/fine 两阶段采样；
- 每个采样点预测 RGB、密度和不确定性；
- 可靠性因子参与不透明度及透射率计算；
- MC-dropout 与多视图光度一致性构造教师不确定性；
- COLMAP 稀疏深度先验、尺度对齐、MAD、Tukey biweight 和伪 Huber；
- 每张训练图像的可学习背景；
- 推理一次前向同时输出 RGB、深度和学生不确定性。

## 1. Ubuntu 环境

推荐 Ubuntu 22.04/24.04、Python 3.10 以上和 NVIDIA GPU。PyTorch wheel 已包含对应 CUDA 运行时，但主机仍需安装兼容的 NVIDIA 驱动。

```bash
sudo apt update
sudo apt install -y python3-venv ffmpeg

cd uncertain-nerf
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

pip install torch==2.12.1 torchvision==0.27.1 \
  --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

如果显卡不支持 CUDA 12.6，请从 PyTorch 官方安装页选择与驱动匹配的 wheel；项目本身不编译自定义 CUDA 扩展。

## 2. 数据准备

每个场景采用以下结构：

```text
data/<scene>/
├── images/
│   ├── image000.png
│   └── ...
├── poses_bounds.npy
└── sparse/0/
    ├── cameras.bin
    ├── images.bin
    └── points3D.bin
```

也支持 `cameras.txt`、`images.txt`、`points3D.txt`。若同一目录同时存在两种格式，优先使用二进制。

要求：

- `poses_bounds.npy` 与 `sparse/0` 必须来自同一次 COLMAP/LLFF 重建；加载时会核对图像名和相机中心。
- 相机类型支持 `PINHOLE`、`SIMPLE_PINHOLE` 和 `SIMPLE_RADIAL`；`SIMPLE_RADIAL` 会在射线生成时反解径向畸变，并在多视图重投影时应用对应畸变，无需预先去畸变。
- 图像分辨率一致；`factor` 缩放图由 OpenCV 自动生成，不依赖 ImageMagick。
- 稀疏点默认要求轨迹长度至少为 3、重投影误差不超过 2 像素。

## 3. 训练

复制并修改 `configs/llff_colmap_v5.txt`，或在命令行覆盖场景参数：

```bash
CUDA_VISIBLE_DEVICES=0 python run_nerf.py \
  --config configs/llff_colmap_v5.txt \
  --datadir ./data/<scene> \
  --expname <scene>_v5
```

日志和检查点保存在 `logs/<scene>_v5/`。程序默认自动读取 `checkpoints/latest.pt`；指定检查点可使用：

```bash
python run_nerf.py --config configs/llff_colmap_v5.txt \
  --datadir ./data/<scene> --expname <scene>_v5 \
  --ft_path ./logs/<scene>_v5/checkpoints/step_050000.pt
```

若显存不足，依次减小 `N_rand`、`teacher_ray_ratio`、`N_importance` 和 `netchunk`。`--amp` 仅在 CUDA 上生效。

## 4. 推理

渲染 LLFF 轨迹：

```bash
CUDA_VISIBLE_DEVICES=0 python run_nerf.py \
  --config configs/llff_colmap_v5.txt \
  --datadir ./data/<scene> \
  --expname <scene>_v5 \
  --render_only --render_split path \
  --ft_path ./logs/<scene>_v5/checkpoints/latest.pt
```

渲染测试视角：

```bash
python run_nerf.py --config configs/llff_colmap_v5.txt \
  --datadir ./data/<scene> --expname <scene>_v5 \
  --render_only --render_split test
```

自定义相机轨迹应保存为 `[N,3,4]` 或 `[N,4,4]` 的 NumPy 数组：

```bash
python run_nerf.py --config configs/llff_colmap_v5.txt \
  --datadir ./data/<scene> --expname <scene>_v5 \
  --render_only --render_poses ./poses/my_path.npy
```

输出目录中包含：

- `rgb_*.png`、`rgb.mp4`；
- `depth_*.npy`、`depth_*.png`、`depth.mp4`；
- `uncertainty_*.npy`、`uncertainty_*.png`、`uncertainty.mp4`。

深度采用 LLFF/NeRF 场景归一化单位，不代表公制距离；不确定性范围为 `[0,1]`。推理时网络处于 `eval` 模式，不执行 MC-dropout 或多视图教师计算。

## 5. 测试

```bash
pytest -q
```

CPU 可运行单元测试和小型端到端测试；完整训练建议使用 CUDA。原版 NeRF `.tar` 检查点与 V5 网络结构不兼容，加载时会明确报错。
