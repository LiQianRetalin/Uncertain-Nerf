# CamP + Zip-NeRF 最终淘汰报告

状态：**FAIL，路线永久停止，不再调参、不再延长训练、不进入 V8。**

## 最终结果

| 指标 | 结果 |
|---|---:|
| raw PSNR mean | 21.0918986003 dB |
| raw SSIM mean | 0.6440748771 |
| Procrustes-aligned PSNR mean | 17.4028199514 dB |
| Procrustes-aligned SSIM mean | 0.4254284402 |
| aligned PSNR 相对 27.20 dB 门槛 | -9.7971800486 dB |
| aligned PSNR gate | FAIL |
| aligned SSIM gate | FAIL |

逐图 raw PSNR 为 20.2580776215、21.0591716766、21.9584465027 dB；逐图
aligned PSNR 为 16.9808311462、17.4804954529、17.7471332550 dB。对齐不但没有
恢复质量，反而进一步下降，不能再用“相机标定误差尚未充分优化”解释当前差距。

## 决定

- CamP、Zip-NeRF、CamP + Zip-NeRF 不再作为下一 V8 的候选或教师。
- 不再进行相机参数、训练步数、对齐方式或评测后处理的追加试验。
- 27.20 dB 继续作为 Fern 干净场景入口，不因本次失败降低。
- 项目转入显式 Gaussian 地图主干的机器人短筛。
- 旧 A0/V7.5/V8-Core 的 NeRF 门控链保留为历史证据，不再是当前实施入口。

该结论使用 2026-08-27 用户提供的最终训练与评测输出记录。
