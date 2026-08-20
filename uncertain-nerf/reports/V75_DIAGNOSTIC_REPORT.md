# V7.5 诊断阶段报告

状态：必须在 baseline 通过后运行。

固定配置：`configs/v8_v7_5.txt`  
验收场景：与 baseline 完全相同的 fern/factor=2/seed=0/split。

实现保证：`m=1`，UQ 不进入 density、RGB、geometry、sampling、background；UQ head
使用 detached hidden 和独立 optimizer，训练时自动检查 `dL_UQ/dtheta_recon=0`。

| 指标 | baseline | V7.5 | 差值 |
|---|---:|---:|---:|
| best val PSNR | 待运行 | 待运行 | 待运行 |
| final val PSNR | 待运行 | 待运行 | 待运行 |
| test PSNR | 待运行 | 待运行 | 待运行 |
| mean opacity | 待运行 | 待运行 | 待运行 |
| mean terminal T | 待运行 | 待运行 | 待运行 |
| mean u | N/A | 待运行 | N/A |

通过条件：`abs(PSNR_V7.5 - PSNR_baseline) <= 0.3 dB`。失败时生成
`STOP_V75_FAILURE.md`，不进入 V8 SDF。
