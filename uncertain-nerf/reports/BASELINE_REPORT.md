# A0 Baseline 恢复报告

状态：旧 diagnostic baseline 已停止；等待原始 A0 在 L6 正式训练。

原 `configs/v8_baseline.txt` 运行已失败并停止。完整证据见
`reports/STOP_BASELINE_FAILURE.md`。该配置实际复用了 V7 重建主干，不能作为原始
A0 验收。新的 A0 固定配置为 `configs/a0_fern.txt`，执行手册为
`A0_RECOVERY_GUIDE.md`。

验收场景：fern，factor=2，seed=0  
代码版本：待 A0 恢复提交后填写。

| 指标 | 结果 |
|---|---:|
| PSNR_train | 待运行 |
| PSNR_val(best/final) | 待运行 |
| PSNR_test | 待运行 |
| SSIM_test | 待运行 |
| LPIPS_test | 待运行 |
| opacity_mean | 待运行 |
| depth_mean | 待运行 |
| gradient_norm_density | 待运行 |
| gradient_norm_color | 待运行 |
| gradient_norm_geometry_loss | 待运行 |
| mean_weight_sum / terminal T | 待运行 |
| GPU time | 待运行 |
| peak memory | 待运行 |

停止条件：若使用核对无误的相机、split、bounds 和评测代码后仍只有约 22 dB，生成
`STOP_BASELINE_FAILURE.md`，不进入 V7.5 或 V8 SDF。
