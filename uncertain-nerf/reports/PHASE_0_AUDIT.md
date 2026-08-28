# PURI-GS Phase 0：仓库、环境与数据审计

审计日期：2026-08-28

## 结论

- 开始时位于 `dev`，commit 为 `7baf0db443dc8be2e9f7d4dba591da162171435d`，工作区干净；全程未切换或新建分支/worktree。
- 当前唯一 Gaussian 主干是固定 commit 的 `gsplat v1.5.3`；外部源码目录不进入 Git，项目通过一个可逆补丁部署最小修改。
- 未修改 PyTorch、CUDA、gsplat、编译器或环境。
- Mip-NeRF 360 当前目录存在但为空，状态为 `PARTIAL`。此状态不阻塞审计、配置、A1 实现和损失测试，但阻塞 clean 正式比较及稀疏视角实验。
- RobustNeRF 有 4 个当前加载器可用场景和 1 个部分场景；首个动态场景推荐 `android`。

## Git 审计

| 项目 | 结果 |
|---|---|
| 分支 | `dev` |
| 开始 commit | `7baf0db443dc8be2e9f7d4dba591da162171435d` |
| 开始工作区 | clean |
| 分支操作 | 无 |
| rebase / merge | 无 |

## 实际代码位置

| 功能 | 实际位置 |
|---|---|
| trainer 入口 | `external/gsplat-v1.5.3/examples/simple_trainer.py`，由 `run_puri_gs.py` 调用 |
| evaluation 入口 | 同一 trainer 的 `Runner.eval()`；checkpoint-only 模式单独评测 |
| L1 / DSSIM | `simple_trainer.py` 训练循环；L1 接入 `puri_gs/responsibility.py`，DSSIM 仍为 `fused_ssim` |
| DefaultStrategy 调用 | `Runner.rasterize_splats()`、`step_pre_backward()`、`step_post_backward()` |
| DefaultStrategy 实现 | `external/gsplat-v1.5.3/gsplat/strategy/default.py` |
| densification 梯度统计 | `DefaultStrategy._update_state()`；读取 `means2d.grad/absgrad`、`radii`、`gaussian_ids` |
| rasterizer | `external/gsplat-v1.5.3/gsplat/rendering.py::rasterization` |
| 可用 rasterizer 输出 | `radii`、`means2d`、`depths`、`gaussian_ids`、`opacities`、图像宽高等 |
| checkpoint 保存/加载 | `torch.save` 训练 checkpoint；`torch.load(..., weights_only=True)` 评测加载 |
| 数据加载器 | `examples/datasets/colmap.py::Parser/Dataset` |
| 标准相机张量边界 | 每个 batch 输出 `K [B,3,3]` 与 `camtoworld [B,4,4]`；trainer 不依赖 LLFF `poses_bounds.npy` 才能训练 |
| 现有外部位姿接口 | `v8_robot/pose_contract.py`（本阶段未改） |
| 服务器项目路径 | `/home/chenglong/Uncertain-Nerf/uncertain-nerf` |

## gsplat v1.5.3 行为

- `DefaultStrategy.grow_grad2d` 默认 `0.0002`，`absgrad=False`。
- AbsGrad 要求 rasterization 同时启用 `absgrad`；现有 trainer 已从 strategy 字段传递给 rasterizer。
- densification 统计在 Python strategy 层完成，本阶段 A1 不修改该统计、不修改 CUDA。
- 非 packed 模式的可见性信号为 `info["radii"] > 0`；这可供未来 A3 审计，但本阶段不实现 A3。

## 数据审计摘要

完整机器可读清单见 `reports/local_dataset_manifest.json`，人读版本见 `reports/local_dataset_manifest.md`。

### Mip-NeRF 360

```text
status: PARTIAL
missing_items: [bicycle, bonsai, counter, garden, kitchen, room, stump]
reason: 下载/解压尚未形成任何正式场景目录
```

### RobustNeRF

固定协议为文件名含 `clutter` 的图像训练、含 `extra` 的干净图像测试。禁止使用 every-8 划分。

| 场景 | 状态 | images_4 | 注册相机 | train clutter | test extra |
|---|---:|---:|---:|---:|---:|
| android | READY | 263 | 263 | 122 | 19 |
| crab1 | READY | 143 | 143 | 72 | 71 |
| crab2 | READY | 412 | 412 | 109 | 194 |
| statue | PARTIAL | 416 | 415 | 255 | 19 |
| yoda | READY | 420 | 420 | 109 | 202 |

`android` 被选为首个动态场景：COLMAP/下采样图完整、训练/测试关键字明确、训练图数量比 Crab2/Yoda 更小，且不需要格式转换。

RobustNeRF 的下采样文件保留 JPG 扩展名但实际为 PNG payload。补丁使用 8 字节文件签名只读检测，避免 gsplat 错误地再次下采样和写出派生目录。

### 其他关键数据

- smoke：`nerf_example_data/nerf_llff_data/fern`，当前 COLMAP loader 可读。
- NeRF On-the-go：图像 + `transforms.json`，当前为 `UNSUPPORTED_FORMAT`；本阶段不转换。
- Mip-NeRF 360 下载完成后 clean 候选固定为 `garden`、`room`。

## 环境结论

详见 `reports/environment_before.txt`。本地已有 WSL PyTorch/CUDA 可完成 A1 纯 PyTorch 测试，但没有可用的 gsplat 安装；现有离线 wheel 与本地 PyTorch ABI 不匹配。未自动安装或编译。

## Phase 0 状态

仓库、环境和数据审计本身完成。由于同一工作区还包含尚未经过真实 gsplat CUDA smoke 的 A1 接入代码，当前不建议把整批改动作为已验证功能提交。
