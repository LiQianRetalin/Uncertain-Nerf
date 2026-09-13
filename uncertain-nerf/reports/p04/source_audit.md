# P04 冻结源码机制核查

审计时本地与服务器均在 `ru-part`、HEAD `41a50f5f3ba7429b9b41ea771e13a8ebc9a42780`，工作区起点清洁。原 Android RU 训练记录 commit `9e29230952be700dc5b527008e2f60f05b82717d`，Room factor4 历史 RU 为 `da5089cd34bad85780e974f732c2bc722b05c77f`；P03 主工程来源 `795830a4a55f3cb363cef5ac674abd8190421d42` 只作身份参考，不回滚。gsplat 1.5.3 源 HEAD `937e29912570c372bed6747a5c9bf85fed877bae`。服务器已安装包与 `/external/gsplat-v1.5.3-ru` 的 `rendering.py`、`strategy/default.py`、`RasterizeToPixels3DGSFwd.cu`、`RasterizeToIndices3DGS.cu` 分别哈希一致；诊断使用已安装包而不是缺少 GLM 子模块的源码副本。

另用 `git show` 只读对照两个原训练 commit 的 `puri_gs/delayed_absgrad.py` 与 RU 补丁：两次旧源码的 `_update_state` 从 0 开始、10000–19999 事件和 reset 顺序一致。当前 HEAD 加入了其它工作包的拓扑事件记录/可选窗口；这些新增分支没有被套用于两个旧 checkpoint。`semantic_mask.py` 相对旧 commit 未变；当前 `ru_training.py` 新增其它包控制，但本包引用的原 Mask 生成/损失语义经旧补丁核对一致。

## 原策略统计与事件顺序

真实调用链：`examples/simple_trainer.py:662` 的 `Runner.rasterize_splats` → `gsplat.rendering.rasterization` → 原 CUDA 光栅化；训练 `simple_trainer.py:982` Gaussian 光度 `loss.backward()` → `puri_gs/delayed_absgrad.py:196` 的 `DelayedAbsGradStrategy.step_post_backward` → `gsplat/strategy/default.py:203` 的 `_update_state` → 到期 `_grow_gs/_prune_gs` → 清空 `grad2d/count/radii` → 到期 opacity reset → Gaussian optimizer。RU 的策略在 optimizer 前，普通 B1 在 optimizer 后，不能交换时间口径。

`DefaultStrategy._update_state` 取 `info[key_for_gradient]`（本配置为 `means2d`）的 `.absgrad`，分别乘 `width/2*n_cameras`、`height/2*n_cameras` 转为屏幕尺度，再取二维向量范数累加进 `grad2d`；`count` 对非 packed 路径中 `info['radii'] > 0` 两轴皆真的投影各加 1（`default.py:221-253`）。本包原 RU `packed=false`。它不检查像素级 alpha/前景遮挡、Mask 是否接受或该 Gaussian 是否合成了实际像素，因此“投影计数稀释”在源码层面确实可能发生；这不是对实际场景强度的证明。原增长分数为 `grad2d/count.clamp_min(1)`，阈值 `grow_grad2d=0.0006`；P04 的比较候选不能先用该阈值裁掉。

日程实际为 Mask 光度生效 step 500，增密 step 10000 至 19999、每 100 步，统计从 step 0 至 19999，首个 step 10000 事件含此前累积而非仅 100 张最近视图。每次增密事件在本次 `_update_state` 后执行，随后清空统计；opacity reset 在 step 15000、18000，若与增密同一步则在增密/清空之后执行。Mask 头更新暂停的是 reset 后 300 步（15001–15300、18001–18300），不是停止 Gaussian 投影统计。对应 `delayed_absgrad.py:46-88,196-239`；正式运行 `config.yaml` 与 `aux/training_schedule.json` 同值。20k 后停止增密和策略统计，DINO 渲染由 coarse 224×224 切为 fine；P04 只有同一步 29999 Gaussian+头+残差历史，属于终点静态状态，不是把 step 改写成增密期。

## 硬 Mask 与梯度时序

`ru_training.py:226-265`：取训练图的冻结 DINO 特征缓存（30k 终点 36×36 fine），经 `StaticResponsibilityHead` 的 384→16→1 MLP+Sigmoid、双线性上采样，再 `semantic_mask.py:147-154` 以概率严格大于 0.25 二值化、7×7 最小池化形成硬 Mask。当前图的硬 Mask 在 Gaussian 光度 backward 前 detach；不靠当前图 residual 直接生成。头的学习来自 `complete_mask_supervision`：coarse/fine 渲染的 DINO 相似度目标、当前 RGB residual 与历史直方图 0.60/0.80 分位、静态先验和权重项，再在 Gaussian 光度 backward 后执行独立 Mask backward 和头 optimizer（`ru_training.py:269-345`；`simple_trainer.py:982-1000`）。因此终点头和缓存能恢复终点 A；虽然 residual_hist 同步存在，它不需要在诊断中更新。重放、拟合或随机头均不合格。

`semantic_mask.py:156-177` 的 L1 是 `(M*abs(render-target)).mean()`，分母为完整 B×H×W×3，不除以接受像素数。DSSIM 是先将 render 与 target 分别乘 M，再对整幅图调用 `fused_ssim(..., padding='valid')`；局部 SSIM 窗跨接受/拒绝边界会改变接受侧的梯度，但乘 M 对完全拒绝像素的直接导数为零。故若精确合成接受贡献 `bM=0` 而投影 AbsGrad `g>0`，需实查数值阈值、边界与其他损失，不能只用“SSIM 泄漏”断言，也不能用小 epsilon 自动放大。

## 光栅化精确口径

`gsplat/cuda/csrc/RasterizeToPixels3DGSFwd.cu:18-178` 逐 tile 按深度排序，从 `T=1` 开始，`alpha=min(0.999, opacity*exp(-sigma))`；`sigma<0` 或 `alpha<1/255` 跳过，若 `T*(1-alpha)<=1e-4` 则该 Gaussian **不计入**并结束该像素，反之其合成权重为 `T*alpha` 后更新 T。仅投影 `radii>0` 不保证贡献。P04 独立探针按此规则扫描全部前方 Gaussian，只对固定至多 512 个 ID 汇总 b、bM、m；任何与原 alpha 不一致的视图应判诊断失败，不降级成中心采样近似。

## 资产等级与适用结论

有界检查 P01 登记的 Android `/logs-puri/ru-generalization-rerun-9e292309/android_ru_30k`、Room `/logs-puri/phase_r/room_ru_30k`：两者 `ckpts/` 仅 `ckpt_29999_rank0.pt`，`aux/` 有同一步 Mask 头、Mask optimizer、residual_hist、配置/日程和 DINO 环境。无增密期 checkpoint，因此均为 `MATCHED_TERMINAL_STATE`。Android 训练协议 factor4 1007×755，Room 历史 factor4 779×519；绝不借用 Room factor2。终点静态排序无法复原增密窗统计或预测 P05 最终 PSNR。
