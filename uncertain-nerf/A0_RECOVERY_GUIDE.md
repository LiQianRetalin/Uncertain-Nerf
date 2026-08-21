# 原始 NeRF A0 恢复门执行手册

## 1. 为什么停止旧 baseline

`fern_v8_baseline_seed0` 不是原始 A0。它是 V7 HashGrid/Mip/终止分布几何/
固定混合采样重建链路去掉 uncertainty 后的诊断版本。100000 步 test PSNR 为
21.9691 dB；20000 步早停检查点为 22.6042 dB。早停减轻了后期退化，但没有恢复
健康基线。因此禁止继续 V7.5 或 V8-Core。

本手册的新 A0 独立使用：

- 原始 coarse/fine 两个 8 x 256 NeRF MLP；
- 位置编码 `multires=10`、方向编码 `multires_views=4`；
- LLFF forward-facing NDC；
- 均匀图像/像素采样，不读取 uncertainty、边缘或 COLMAP 深度；
- fine RGB MSE + 等权 coarse RGB MSE；
- Adam 5e-4 和原始指数学习率衰减；
- 原始训练期 density noise `raw_noise_std=1.0`；
- 与原始实现一致使用 FP32，不在固定 A0 配置中开启 AMP；
- 与 V6/V7/V8 完全相同的 fern factor=2 split：train 14、val 3、test 3。

原始 NeRF 训练日程为 200000 步，因此正式 A0 保留 200000 步；同时必须单独评测
100000 步检查点，提供与 V6/V7/V8 的等步数对照。不能只报告 200000 步结果。

## 2. Git 图形化操作

本地 Git 客户端打开 `E:\6-Project\1-UncertainNerf`，保持 `dev`，不要创建或
切换分支。提交前只勾选以下 A0 文件：

```text
uncertain-nerf/.gitignore
uncertain-nerf/README.md
uncertain-nerf/V8_GUIDE.md
uncertain-nerf/a0/
uncertain-nerf/run_nerf_a0.py
uncertain-nerf/configs/a0_fern.txt
uncertain-nerf/scripts/train_a0.sh
uncertain-nerf/scripts/render_a0.sh
uncertain-nerf/scripts/evaluate_a0.sh
uncertain-nerf/tests/test_a0_core.py
uncertain-nerf/tests/test_a0_trainer.py
uncertain-nerf/reports/BASELINE_REPORT.md
uncertain-nerf/reports/STOP_BASELINE_FAILURE.md
uncertain-nerf/A0_RECOVERY_GUIDE.md
```

不要勾选工作区根目录的 `.gitignore` 或 `训练及渲染结果/`。建议提交说明：

```text
Recover original NeRF A0 baseline
```

提交后 Push 到远程 `dev`，不要 force push。

## 3. 服务器下拉与环境复用

服务器终端执行：

```bash
cd ~/Uncertain-Nerf/uncertain-nerf
git status --short
```

只有确认服务器没有未提交源码修改后，才在 VS Code 的源代码管理界面执行 Pull。
Pull 完成后回到终端：

```bash
cd ~/Uncertain-Nerf/uncertain-nerf
git log -1 --oneline
source .venv-v7/bin/activate
CUDA_VISIBLE_DEVICES=6 python -m pytest -q \
  tests/test_a0_core.py tests/test_a0_trainer.py
```

新 A0 只依赖已经验证过的 `.venv-v7` 包，不重新安装环境。

## 4. L6 smoke test

先确认物理 GPU 6 空闲：

```bash
bash scripts/check_v8_environment.sh 6
```

运行 20 步 smoke；它使用独立目录，不会覆盖正式训练：

```bash
CUDA_VISIBLE_DEVICES=6 python run_nerf_a0.py \
  --config configs/a0_fern.txt \
  --datadir ./data/nerf_llff_data/fern \
  --expname fern_a0_original_smoke \
  --N_rand 128 \
  --N_samples 16 \
  --N_importance 8 \
  --N_iters 20 \
  --netdepth 2 \
  --netwidth 32 \
  --netdepth_fine 2 \
  --netwidth_fine 32 \
  --i_print 1 \
  --i_eval 0 \
  --i_weights 20 \
  --no_reload
```

通过条件：20 步结束、没有 traceback/NaN/Inf/OOM，并生成：

```text
logs-a0/fern_a0_original_smoke/checkpoints/step_000020.pt
logs-a0/fern_a0_original_smoke/training_summary.json
```

smoke 通过前不要开始正式训练。

## 5. L6 正式训练

确认没有同名历史 A0 检查点后创建 tmux：

```bash
tmux new -s a0-fern
```

在 tmux 中执行：

```bash
cd ~/Uncertain-Nerf/uncertain-nerf
source .venv-v7/bin/activate
bash scripts/train_a0.sh ./data/nerf_llff_data/fern fern 0 6
```

看到训练开始后按 `Ctrl+B`，松开，再按 `D` 退出 tmux。重新查看：

```bash
tmux attach -t a0-fern
```

不要同时在物理 GPU 6 启动其他训练。A0 的经典 MLP 明显慢于 HashGrid；保留
10000 步检查点，验证每 5000 步执行一次，不能只看最后一步。训练到 100000 步时
不要终止，保留该检查点后继续到原始日程的 200000 步。

## 6. 正式训练完成后的渲染与评测

先评测 100000 步等步数对照，使用独立实验目录，避免改写正式训练的参数文件：

```bash
CUDA_VISIBLE_DEVICES=6 python run_nerf_a0.py \
  --config configs/a0_fern.txt \
  --datadir ./data/nerf_llff_data/fern \
  --expname fern_a0_original_step100k_review \
  --render_only \
  --render_split test \
  --ft_path ./logs-a0/fern_a0_original_seed0/checkpoints/step_100000.pt

bash scripts/evaluate_a0.sh \
  ./logs-a0/fern_a0_original_step100k_review/render_test_100000 \
  0 6
```

然后使用 200000 步 `latest.pt` 依次渲染正式三个 split：

依次渲染三个 split：

```bash
bash scripts/render_a0.sh ./data/nerf_llff_data/fern fern 0 6 train
bash scripts/render_a0.sh ./data/nerf_llff_data/fern fern 0 6 val
bash scripts/render_a0.sh ./data/nerf_llff_data/fern fern 0 6 test
```

评测：

```bash
bash scripts/evaluate_a0.sh \
  ./logs-a0/fern_a0_original_seed0/render_train_200000 \
  0 6 ./logs-a0/fern_a0_original_seed0/training_summary.json

bash scripts/evaluate_a0.sh \
  ./logs-a0/fern_a0_original_seed0/render_val_200000 \
  0 6 ./logs-a0/fern_a0_original_seed0/training_summary.json

bash scripts/evaluate_a0.sh \
  ./logs-a0/fern_a0_original_seed0/render_test_200000 \
  0 6 ./logs-a0/fern_a0_original_seed0/training_summary.json
```

## 7. 决策门

- 若 A0 达到历史正常质量并接近/超过 25 dB：以 A0 为重建主干建立严格匹配的
  V7.5 旁路 UQ，再检查二者 test PSNR 差值是否不超过 0.3 dB。
- 若 A0 仍只有约 22 dB：继续保持 `STOP_BASELINE_FAILURE`，优先核对原始数据、
  NDC、位姿、图像预处理和评测，不进入 V7.5/V8-Core。
- 不以 train batch PSNR 替代完整 val/test 指标，不以 200000 步自动覆盖 best val
  检查点结论。100000 步检查点还必须使用独立 review 实验名完成 test 渲染和评测，
  以便与 V6/V7/V8 做等步数比较。
