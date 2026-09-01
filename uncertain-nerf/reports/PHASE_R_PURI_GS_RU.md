# Phase R：PURI-GS-RU 决策报告

最终结论：`RU_RECONSTRUCTION_FAIL`

| 门禁 | 结果 | 关键事实 |
| --- | --- | --- |
| Android 鲁棒性 | PASS | ΔPSNR=1.2065 dB，ΔSSIM=0.028468，LPIPS 相对改善=3.77%，逐图改善=100.00%，bootstrap CI95=[0.8821, 1.5805] |
| Room clean | FAIL | ΔPSNR=0.7480 dB，ΔSSIM=0.013096，ΔLPIPS=-0.006660，最差逐图 ΔPSNR=-3.8602 dB |
| 推理效率与独立路径 | FAIL | 逐场景事实见 JSON |
| 训练时长建议（非核心门） | INFO | Android=1.010×，Room=1.025×；完整 DINO 用时、checkpoint 比例和 mask 参数量见 JSON |

本报告只执行附件规定的唯一门禁；不会自动改变 B1、调整阈值或进入阶段 U。
