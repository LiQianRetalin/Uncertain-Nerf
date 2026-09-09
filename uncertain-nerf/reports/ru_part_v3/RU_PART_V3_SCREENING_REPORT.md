# RU-PART-V3 Garden 开发场景筛选报告

状态：**COMPARISON_INCOMPLETE**

实现仅增加 0.8 × mean((1−M)C × RGB绝对误差)，step 500 开启。基础 RU/DSSIM/责任头和常规 ADC 保留。
此处协议字段描述固定设计；是否完成训练以 training_actually_completed 和阶段状态文件为准。

阶段进度（用户提供的服务器输出）：原配对profiler中位数16.2875/18.0096毫秒，比值1.105733（增加10.57%），曾超过1.08检查线。优化后的单次V3复测中位数17.320367507636547毫秒，与保留Parent相比为1.0634138514755154（增加6.34%），短测开销检查通过。调用计数600/1200、禁用项0、参数梯度检查正常。历史标准Parent在GPU 0重评exitcode=0，24张名单匹配，PSNR差为0；代码已增加复用登记，尚待在服务器实际登记。600步smoke不代表正式30k训练完成，不改变下列最终验收项状态。

重评Parent指标为PSNR26.650571823120117、SSIM0.8573052287101746、LPIPS0.09050631523132324；DSC07988 PSNR14.080384254455566。Gaussian数2274197。GPU 0预热10次、24个原始推理样本，FPS143.3705007690993，p50/p95延迟6.898403/7.451010毫秒，推理显存1.854206GiB。标准检查点加载通过，DINO/head关闭，单raster，ROI数组均存在。表中的Parent数值属于此次重评，最终复用仍须实际配置与文件登记通过。

首次V3尝试曾在launcher参数检查阶段失败：`FAILED / exit_code=1 / last_step=-1`。修复该参数问题后的、尚未优化传输的checkpoint SHA-256为`e8eba8d15f9ca3da2c19497683e8c5c252df7c5c6922ad29abbe2ffc9853d20e`，训练统计11.5152秒、子进程总墙钟43.9004秒。该原始运行应保留在归档中，不能把它的checkpoint哈希或总耗时标成最新优化复测结果。

| 指标 | B1 历史参考 | 重评 Parent（待登记） | V3 | V3−Parent |
|---|---:|---:|---:|---:|
| psnr | 27.716728 | 26.650572 | 未测 | 未测 |
| ssim | 0.873957 | 0.857305 | 未测 | 未测 |
| lpips | 0.068151 | 0.090506 | 未测 | 未测 |
| DSC07988_psnr | 20.491301 | 14.080384 | 未测 | 未测 |

固定恢复门：psnr=NOT_ASSESSABLE；ssim=NOT_ASSESSABLE；lpips=NOT_ASSESSABLE；DSC07988_psnr=NOT_ASSESSABLE

Parent 质量保护：psnr=NOT_ASSESSABLE；ssim=NOT_ASSESSABLE；lpips=NOT_ASSESSABLE

资源门：gaussian_count=NOT_ASSESSABLE；training_time_ratio=NOT_ASSESSABLE；single_rasterization=NOT_ASSESSABLE；fps_ratio=NOT_ASSESSABLE

早期证据激活：`NO_Q_SAMPLED / SUPERVISION_INACTIVE_AT_AUDITED_STATE`。两张固定训练图C非零，但Mask拒绝率=0，因此Q=0；固定张量功能检查梯度误差=0。最新优化复测仍无真实采样激活，正式训练期激活与最终覆盖未测。51项相关CPU测试及最新CUDA短测开销检查通过，不把Q=0解释为C恒零或整个训练过程永不激活。

孔洞 ROI：{"status": "NOT_ASSESSABLE"}

成本：{"cache_prepare_seconds": null, "original_cache_build_seconds": null, "training_plus_original_cache_seconds": null}

缺失项：历史Parent实际配置及文件登记；v3: TRAIN_COMPLETE；v3: EVAL_COMPLETE；同口径完整训练计时。历史训练GPU为1，不能将1347.454996秒直接当成当前GPU 0的同口径成本。历史图像/SfM字节哈希与30k相机序列未记录，当前清单不冒充历史证据。

单 seed 只能提供候选信号，不能证明统计显著性、跨场景泛化、收敛或静态标签正确性。Garden 已用于开发选择。
报告完成后停止，不自动开展下一算法、多 seed、跨场景、Oracle 或重放。

```json
{
  "BRANCH": "ru-part",
  "TRAINING_FROM_SCRATCH": true,
  "RESUMED_FAILED_SNAPSHOT": false,
  "HISTORICAL_REPLAY_STATUS": "REPLAY_NOT_EQUIVALENT",
  "HISTORICAL_REPLAY_RECLASSIFIED": false,
  "NEW_PROTOCOL": "V3_FROM_SCRATCH_SCREENING",
  "EXACT_200_STEP_REPLAY_REQUIRED": false,
  "NEW_ALGORITHM": "RU-PART-V3",
  "DEDICATED_BIRTH_ENABLED": false,
  "PART_CAP_ENABLED": false,
  "EXTRA_TOPOLOGY_SCORING_ENABLED": false,
  "MULTI_SEED_RUN": false,
  "CROSS_SCENE_RUN": false,
  "NEXT_ALGORITHM_STARTED": false
}
```
