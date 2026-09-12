# P03 Corner 预检

状态：`PASS / FROZEN_BEFORE_TRAINING`

- 原始 Corner 共 122 帧；官方 `clutter` 101 帧训练、`extra` 20 帧测试，frame 121 排除。
- 物理图像只做一次 factor 4 缩放；共同外部加载器使用 factor 1。
- 三个身份共享 10,490 个由固定相机位姿和训练帧生成的初始点，未重跑 SfM。
- 原生 Corner 像素经过畸变校正及 ROI 裁剪后，与共同 COLMAP 输入逐像素完全一致，最大绝对差为 0。
- 代表测试帧在训练前固定为：`extra_0000_IMG_7106.png`、`extra_0010_IMG_7137.png`、`extra_0120_IMG_7262.png`。
- SLS-mlp 的 Corner 配方来自固定提交的一般场景循环，使用默认边界 0.5/0.9；不沿用 Patio-High 的 0.3/0.8 特例。
- 测试 GT 仅用于训练完成后的独立评测，不参与训练或调参。

详细逐文件哈希与位姿、投影矩阵核验见 `input_validation.json` 和 `protocol_manifest.json`。
