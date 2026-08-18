# Uncertain-NeRF V7：严格实施、远程训练与渲染指南

## 1. V7 的版本边界

V7 与 V6 并列存在，不覆盖 V6。入口、模块、默认配置、日志目录和检查点版本分别是：

- `run_nerf_v7.py`；
- `v7/`；
- `configs/llff_colmap_v7.txt`；
- `logs-v7/`；
- `uncertain-nerf-v7-1`。

V5/V6 检查点不能加载到 V7。V7 关闭了静态 LLFF 的 appearance embedding，模型结构也加入了 mip-aware 输入，所以必须从头训练。

## 2. 九项修改的严格顺序及代码落点

1. uncertainty 梯度隔离：`v7/model.py` 对 uncertainty head 的 trunk 输入执行停止梯度；`v7/rendering.py` 对 uncertainty 聚合所用终止权重停止梯度；`v7/losses.py` 对 NLL 中预测 RGB 停止梯度。Teacher、NLL 和 spatial（虽然默认关闭）只能更新 uncertainty head。
2. 静态 LLFF 关闭 appearance：V7 强制 `appearance_dim=0`，优化器和检查点均没有 appearance embedding。
3. `lambda_spatial=0`：默认配置和参数校验都强制为零；当前纠错阶段不允许误开。
4. Teacher 延迟和可见性：默认 `teacher_start_step=10000`；投影到源视图后，再渲染源视图射线，以深度一致性、累积不透明度、画面边界和相机正面共同判定可见性。
5. geometry 不由 uncertainty 降权：V7 几何损失完全不读取 uncertainty。
6. 终止权重分布监督：COLMAP 深度被转换为沿 ray 的窄高斯目标 PMF，直接监督全部体渲染终止权重和终端透射率，不再只监督期望深度。
7. 自适应采样：batch 由 sparse-prior、Sobel 边缘、在线 uncertainty EMA 图和均匀射线混合组成。
8. mip-aware HashGrid：每条 ray 根据相机内参得到像素锥半径，每个样本按深度得到 footprint；分辨率高于 footprint 支持范围的 HashGrid level 会平滑衰减。
9. 最后加速：`hash_backend=tcnn` 使用 TCNN；occupancy EMA grid 标记空区域；渲染时只压缩并查询占用样本，实现稀疏 HashGrid/MLP ray marching。默认正确性配置不打开该项，加速配置才打开。

## 3. 本地完成后，何时做 Git 操作

本节所有 Git 操作都用图形化客户端，不需要输入 Git 命令。

### 3.1 现在不要切换分支

保持本地当前 `dev` 分支。不要创建 V7 分支，不要 checkout 其他分支，不要合并 V6 分支。先在图形化客户端顶部确认当前分支显示为 `dev`。

### 3.2 第一次检查改动

在图形化客户端的 Changes/更改面板中检查：

- 新增 `v7/`、`run_nerf_v7.py`、两个 V7 配置、三个 V7 脚本、V7 测试和本指南；
- `README.md` 只增加 V7 入口说明；
- 不应出现 `data/`、`logs-v7/`、检查点、渲染图片、视频、`.venv-v7/` 或本地缓存；
- 不应出现 V5/V6 源文件被修改。

逐文件查看 diff。特别确认默认配置包含 `appearance_dim=0`、`lambda_spatial=0`、`teacher_start_step=10000`，且默认 `hash_backend=torch`、未启用 occupancy grid。

### 3.3 什么时候 commit

只有在本地自动化测试通过、你完成上述 diff 检查后再 commit。建议用一个完整提交，提交说明可写：

`Add gradient-corrected Uncertain-NeRF V7`

在图形化客户端中勾选上述 V7 文件和 README；不要勾选数据、日志或实验输出。点击 Commit 后，再确认 Changes 面板为空。

### 3.4 什么时候 push

commit 完成后、远程服务器开始任何 V7 安装或训练之前 push。图形化客户端中确认目标是远程仓库的 `dev`，然后点击 Push/推送。推送完成后，在客户端历史页确认远程 `dev` 指针与本地刚才的 V7 commit 位于同一提交。

不要在远程 V7 训练已经开始后改写或强推这个提交。后续参数修改使用新的普通提交。

## 4. 远程服务器什么时候下拉

必须在本地 V7 commit 已成功 push 之后、创建 V7 环境和启动训练之前下拉。

推荐用 VS Code Remote-SSH 或服务器上的图形化 Git 面板：

1. 连接远程服务器并打开服务器上的 `uncertain-nerf` 仓库目录。
2. 在 Source Control/源代码管理中确认远程当前分支也是 `dev`。
3. 确认远程 Changes 面板为空。若有远程本地改动，先停止，不要直接 Pull；用图形界面提交到单独提交或暂存，确认不会覆盖服务器文件后再继续。
4. 点击 Fetch/获取，看到远程 `dev` 有新提交后点击 Pull/拉取。
5. 在历史页确认最新提交就是本地刚推送的 V7 commit。
6. 在远程文件浏览器中确认 `run_nerf_v7.py`、`v7/` 和 `V7_GUIDE.md` 已出现。

数据集和训练输出不通过 Git 同步。把数据集单独放到服务器，例如仓库下的 `data/leaves`，并确保 `images/`、`poses_bounds.npy`、`sparse/0/` 都齐全。

## 5. 远程环境安装

进入远程仓库的 `uncertain-nerf` 目录。先做正确性版本，不装 TCNN：

```bash
bash scripts/setup_v7_ubuntu.sh
source .venv-v7/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

若服务器 PyTorch 需要别的 CUDA wheel，在运行安装脚本前设置与服务器驱动兼容的 `TORCH_INDEX_URL`。上面的检查必须显示 CUDA 为 `True`。

## 6. 先做远程 smoke test，不直接跑十万步

smoke test 使用独立实验名，避免污染正式检查点：

```bash
CUDA_VISIBLE_DEVICES=0 python run_nerf_v7.py \
  --config configs/llff_colmap_v7.txt \
  --datadir ./data/leaves \
  --expname leaves_v7_smoke \
  --N_rand 128 --N_samples 32 --N_importance 16 \
  --N_iters 20 --i_print 1 --i_eval 0 --i_weights 20 \
  --no_reload
```

smoke test 验收：

- 没有 NaN、CUDA OOM 或 shape 错误；
- `teacher_count` 始终为 0，因为尚未到 10000 步；
- `logs-v7/leaves_v7_smoke/checkpoints/latest.pt` 存在；
- `args.json` 中三项纠错参数分别是 0、0、10000（appearance、spatial、Teacher 起点）。

若 smoke test 失败，不要开始正式训练，也不要为“试试看”打开 TCNN/occupancy。先在本地修复，测试通过后用图形化客户端创建新 commit 并 push；服务器再按第 4 节 Fetch/Pull。每轮都保持这个同步顺序。

## 7. 正确性 V7 正式训练（第 1～8 项）

第一次正式训练使用默认 PyTorch HashGrid，不启用 occupancy grid，以便先判断质量变化：

```bash
bash scripts/train_v7.sh ./data/leaves leaves 0 0 configs/llff_colmap_v7.txt
```

脚本使用 `--no_reload`，用于确保正式实验从零开始。建议在 tmux/screen 会话中运行，并同时用 GPU 监控工具观察显存和利用率。

训练阶段检查点：

- 1～9999 步：`teacher_count` 必须为 0；先让 RGB、密度和几何稳定。
- 10000 步：Teacher 首次具备启动条件；只有通过有效 ray 和源视图可见性判断的样本才计入 `teacher_count`，所以它可以小于目标比例，偶尔为 0 也不是错误。
- 10000～20000 步：重点观察 `geo`、`teacher`、PSNR 和叶片边缘；如果 teacher 突然造成损失尖峰，先检查 visibility 容差，不要提前恢复 spatial。
- 每 10000 步：生成检查点；不要删除 `latest.pt` 指向的实际 step 文件。
- 100000 步：完成默认正式训练。

如果训练中断，需要原实验名恢复时，不再使用 `train_v7.sh`，因为它强制从零开始。激活环境后运行：

```bash
CUDA_VISIBLE_DEVICES=0 python run_nerf_v7.py \
  --config configs/llff_colmap_v7.txt \
  --datadir ./data/leaves \
  --expname leaves_v7_seed0 \
  --seed 0
```

程序会自动读取 `logs-v7/leaves_v7_seed0/checkpoints/latest.pt`。恢复时必须使用同一数据、划分、配置和实验名。

## 8. 正确性版本渲染

测试视角：

```bash
bash scripts/render_v7.sh ./data/leaves leaves 0 0 test configs/llff_colmap_v7.txt
```

路径视频：

```bash
bash scripts/render_v7.sh ./data/leaves leaves 0 0 path configs/llff_colmap_v7.txt
```

输出位于 `logs-v7/leaves_v7_seed0/render_test_100000/` 或 `render_path_100000/`，包括 RGB、depth、uncertainty、acc、MP4 和效率 JSON。先比较：

- 深度边缘是否从双层重影变成单一终止峰；
- 空中假叶片和 floaters 是否下降；
- 细叶片在斜视角和远景中是否减少闪烁/锯齿；
- uncertainty 是否定位困难区域，而不是通过破坏密度来自我实现。

## 9. 最后才启用 TCNN、occupancy grid 和稀疏 marching

只有默认 V7 的几何和叶片质量已达到预期，才执行本节。先安装 TCNN：

```bash
source .venv-v7/bin/activate
INSTALL_TCNN=1 bash scripts/setup_v7_ubuntu.sh
python -c "import tinycudann; print('TCNN ready')"
```

然后用独立实验名从头训练，不要加载 PyTorch HashGrid 检查点：

```bash
CUDA_VISIBLE_DEVICES=0 python run_nerf_v7.py \
  --config configs/llff_colmap_v7_fast.txt \
  --datadir ./data/leaves \
  --expname leaves_v7_fast_seed0 \
  --seed 0 --no_reload
```

加速版到 10000 步前会持续更新 occupancy EMA，但不把它用于剔除；达到 warmup 后才启用稀疏查询，避免早期空网格永久丢掉细叶片。比较速度时必须使用相同 GPU、分辨率、视角、sample 数和计时口径。

加速版渲染直接使用 V7 入口，因为标准脚本固定了正式实验名：

```bash
CUDA_VISIBLE_DEVICES=0 python run_nerf_v7.py \
  --config configs/llff_colmap_v7_fast.txt \
  --datadir ./data/leaves \
  --expname leaves_v7_fast_seed0 \
  --seed 0 --render_only --render_split test \
  --ft_path ./logs-v7/leaves_v7_fast_seed0/checkpoints/latest.pt
```

## 10. 后续参数修改的同步纪律

每次只改变一个有明确实验目的的参数组。推荐顺序是 visibility 容差、终止分布宽度、adaptive sampling 比例、mip 衰减；不要同时改四组。

每轮流程固定为：本地 `dev` 修改 → 本地测试 → 图形化 diff 检查 → 图形化 commit → 图形化 push 到远程 `dev` → 停止服务器旧进程 → 服务器图形化 Fetch/Pull → 使用新实验名训练。不要在服务器直接修改受 Git 管理的配置，否则下一次 Pull 容易冲突，也无法准确复现实验。

训练生成的 `args.json` 和 `config.txt` 是该次实验的真实配置快照。论文表格中的每个结果都应保留对应快照、commit 标识、随机种子、GPU 型号、训练时间和渲染效率 JSON。
