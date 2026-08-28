# PURI-GS Phase 2：A1 鲁棒静态责任

## 实现范围

实现文件：`puri_gs/responsibility.py`。gsplat 集成通过 `patches/gsplat_v1.5.3_robot_screen.patch` 完成。

实现了：

1. detached RGB 平均 L1 residual；
2. 每图 median 与 `1.4826 * MAD + epsilon`；
3. 只处理高于中位数的正向标准残差；
4. threshold=1.5 的 Huber-style responsibility；
5. `q_min=0.2`；
6. 单次 3x3 average pooling；
7. responsibility-normalized weighted L1；
8. 3000-step warm-up；
9. B0/B1 关闭时直接调用原始 `F.l1_loss`；
10. 最后一步保存 responsibility map，不增加 rasterization；
11. 显式 `clutter`/`extra` 数据划分并保存实际文件名。

保持不变：DSSIM、`ssim_lambda=0.2`、优化器、学习率、Gaussian 参数、DefaultStrategy、densification、pruning、rasterizer 和推理前向。

## 测试结果

### 单元与回归

```text
18 passed in 22.68s
```

其中新增 PURI-GS 测试 11 个，覆盖：

- q 的范围、低侧不降权、高侧异常降权、极值 clamp；
- q 无梯度、常量残差无 NaN、pooling 形状和范围；
- A1 关闭/未到起始步时与原始 L1 精确一致；
- 单步反向参数梯度存在且有限；
- checkpoint 保存/加载；
- 三份配置差异；
- 数据清单及动态关键字划分。

其余 7 个为既有 gsplat 协议和 reproduction gate 回归测试。

仓库在增加 smoke 生效配置记录与安装验证后，最终全量回归结果：

```text
72 passed in 24.51s
```

### Fern 真实图像 10-step loss smoke

```json
{
  "steps": 10,
  "initial_loss": 0.1719341725,
  "final_loss": 0.0347473435,
  "psnr": 17.0674583,
  "responsibility_min": 0.2000000179,
  "responsibility_max": 1.0,
  "responsibility_requires_grad": false,
  "checkpoint_roundtrip": true
}
```

输出位于忽略目录 `tmp/puri_gs_a1_loss_smoke/`，包括 RGB、residual map、responsibility map、checkpoint 和 JSON 指标。

此 smoke 只证明损失、反向、checkpoint 和指标路径，不包含 gsplat rasterizer，不用于算法结论。

## 真实 gsplat 实现验证状态

本地已按授权创建固定的 `.venv-gsplat153`，并真实调用 gsplat CUDA
`projection_ewa_3dgs_packed_fwd`。调用在第一步前停止：本机 GPU 为 `sm_120`，
而固定 PyTorch 和 gsplat wheel 只支持至 `sm_90`；wheel 也不含 PTX。

用户授权后，以上验证已在兼容固定 wheel 的 NVIDIA L20（`sm_89`）上完成，
没有改变 PyTorch、CUDA、gsplat 或核心 rasterizer。已验证 gsplat/tyro import、
真实 CUDA rasterization、Gaussian 参数反向、DefaultStrategy 训练前后处理、
官方 trainer checkpoint 保存和独立加载，以及 A1 最终 responsibility map。

本轮没有采用针对本地 `sm_120` 更换 PyTorch/CUDA 或重编译 rasterizer 的路径。
正式 A1 配置仍保持 `responsibility_start_step=3000`；10-step smoke 启动器可显式
覆盖为 3，并把实际生效配置写入结果目录，保证短 smoke 真正覆盖 A1 路径。

经用户明确授权，服务器验证统一由 `scripts/run_puri_gs_cuda_smoke.sh` 执行。
脚本会在相同 Fern split、相同 10-step 预算和同一物理 L20 上顺序运行三种配置，
随后独立加载各 checkpoint 评测，并生成 `smoke_summary.json`。A1 必须额外存在
`renders/responsibility_step0009.png`；脚本记录其形状、最小值、最大值和唯一值数量。

L20 首次运行在 B0 step 0 前因源码包遮蔽固定 wheel 而停止。修复只调整 Python
导入路径，不改变 A1 损失、配置、DefaultStrategy、CUDA wheel 或实验预算。
第二次运行的 B0/B1/A1 均完成 10 step，并分别保存、加载和评测 checkpoint。

A1 输出的 `responsibility_step0009.png` 形状为 `755 x 1007`，像素值范围
`51..255`，共有 202 个唯一值。下界 51 与 `q_min=0.2` 的 8-bit 表示一致，
同时非恒定图证明责任计算和平滑路径真实生效。A1 测试指标为 PSNR 11.218639、
SSIM 0.426121、LPIPS 0.936116；这些仅是 10-step 可运行性证据，不用于算法判断。

## A1 门禁状态

- 代码级与纯 PyTorch smoke：PASS。
- 实际 gsplat L20 smoke：PASS（B0/B1/A1 真实 CUDA 10 step、反向、checkpoint 与评测）。
- 动态 10k short screening：NOT RUN。
- clean 门禁：BLOCKED BY MIPNERF360 PARTIAL。
- A2/A3：未实现，符合阶段约束。

因此当前不能判断 A1 算法效果，也不能进入 A2。

## 预备的 Android 10k 精确命令

以下命令只在本地里程碑提交并 push、服务器 pull 后执行。当前仅记录，不应立即运行。

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf

PURI_GSPLAT_PYTHON=./.venv-gsplat153/bin/python bash scripts/train_puri_gs.sh configs/puri_gs_b0_default.yaml ./external/gsplat-v1.5.3 ./data/nerf_robustnerf/robustnerf/android ./logs-puri/android_b0_seed42_short10k 6 10000 4 clutter extra

PURI_GSPLAT_PYTHON=./.venv-gsplat153/bin/python bash scripts/train_puri_gs.sh configs/puri_gs_b1_absgrad.yaml ./external/gsplat-v1.5.3 ./data/nerf_robustnerf/robustnerf/android ./logs-puri/android_b1_seed42_short10k 6 10000 4 clutter extra

PURI_GSPLAT_PYTHON=./.venv-gsplat153/bin/python bash scripts/train_puri_gs.sh configs/puri_gs_a1_responsibility.yaml ./external/gsplat-v1.5.3 ./data/nerf_robustnerf/robustnerf/android ./logs-puri/android_a1_seed42_short10k 6 10000 4 clutter extra
```

预期日志均包含 `Dataset split: train=122, test=19`、`Model initialized`、有限 loss，并在成功后产生 `ckpts/ckpt_9999_rank0.pt`。A1 额外产生 `renders/responsibility_step9999.png`。

任一命令出现以下情况立即停止：split 数量不符、import/CUDA 错误、NaN/Inf、OOM、checkpoint 缺失、A1 责任图大面积压低静态区域，或相同预算/配置无法保证。

## 最小服务器数据

只上传 `android` 的以下内容，不上传其他 RobustNeRF 场景：

```text
android/images/
android/images_4/
android/sparse/0/cameras.bin
android/sparse/0/images.bin
android/sparse/0/points3D.bin
```

不上传 `images_2`、`images_8`、`sparse.tmp`、其他场景或原始压缩包。
