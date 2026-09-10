# P02 共同协议与方法原生配方差异

## 共同且不可变的输入

- Android 与 Patio-High 均使用 P01 生成并验证的物理 factor4 COLMAP 共同数据，方法加载器参数固定为 factor1；图像是 1007×755，禁止重跑 SfM、替换相机、点云或拆分。
- 两种方法都从仓库外的隔离数据视图读取共同文件；图像和三个 COLMAP `.bin` 只读链接到共同数据。RobustSplat 首次读取时生成的 `points3D.ply` 只会写入隔离视图，不会落入共同数据目录。
- 训练集仅为文件名含 `clutter` 的图像，测试集仅为含 `extra` 的图像；`clean` 与其他排除图像不进入训练、测试、特征或监督。
- 四个正式身份均为 seed42、作者官方 30,000 更新；不调参、不加 seed。

## RobustSplat

- 固定源提交：`a130281d6d0c004032a9a57e8d6a14962d9836d3`。
- 保留作者 30k 优化、mask、bootstrap、reset、densification 和 DINO 两级特征配方。Scene 内部的 `[1.0, 4.0]` 多尺度是方法原生特征分支，不是更换共同输入或再次物理降采样。
- 仅加显式 seed42、测试集浮点预测导出、原生 `render()` 边界计时。未改变损失、数据选择或优化超参数。

## SpotLessSplats / SLS-mlp

- 固定源提交：`0caae3cc45bb1fddf86bd47e4a521888f5c49889`。
- 固定 `--loss_type robust --semantics --no-cluster`，不传 `--ubp`，因此是 SLS-mlp-no-UBP，不是 SLS-agg。
- Android 的作者边界为 0.5/0.9，Patio-High 的作者 benchmark 边界为 0.3/0.8；只为精确共同训练图像生成 Stable Diffusion 2.1 特征，特征时间单列。
- 仅加显式 seed42、测试集浮点预测导出、原生 `rasterize_splats()` 边界计时。

## 独立评测与计时

- 预测先验证完整冻结测试文件名、1007×755 尺寸、有限值和 `[0,1]` clamp；统一用 P01 的 torchmetrics PSNR、SSIM、LPIPS-Alex（`normalize=True`）逐图计算并算术平均。
- FPS 为 10 次 warmup 后，对完整测试集重复 3 次；每次前后 CUDA 同步，计时区间不含保存图像与指标计算。
- 方法原生世界归一化允许保留并记录，但不得改变像素、相机、初始点、拆分或评测尺寸。
