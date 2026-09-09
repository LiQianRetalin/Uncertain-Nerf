# RU-PART-V3 Garden 开发场景筛选报告

状态：**COMPARISON_INCOMPLETE**

实现仅增加 0.8 × mean((1−M)C × RGB绝对误差)，step 500 开启。基础 RU/DSSIM/责任头和常规 ADC 保留。
此处协议字段描述固定设计；是否完成训练以 training_actually_completed 和阶段状态文件为准。

| 指标 | B1 历史参考 | 可比 Parent | V3 | V3−Parent |
|---|---:|---:|---:|---:|
| psnr | 27.716728 | 未测 | 未测 | 未测 |
| ssim | 0.873957 | 未测 | 未测 | 未测 |
| lpips | 0.068151 | 未测 | 未测 | 未测 |
| DSC07988_psnr | 20.491301 | 未测 | 未测 | 未测 |

固定恢复门：psnr=NOT_ASSESSABLE；ssim=NOT_ASSESSABLE；lpips=NOT_ASSESSABLE；DSC07988_psnr=NOT_ASSESSABLE

Parent 质量保护：psnr=NOT_ASSESSABLE；ssim=NOT_ASSESSABLE；lpips=NOT_ASSESSABLE

资源门：gaussian_count=NOT_ASSESSABLE；training_time_ratio=NOT_ASSESSABLE；single_rasterization=NOT_ASSESSABLE；fps_ratio=NOT_ASSESSABLE

证据激活：未测

孔洞 ROI：{"status": "NOT_ASSESSABLE"}

成本：{"cache_prepare_seconds": null, "original_cache_build_seconds": null, "training_plus_original_cache_seconds": null}

缺失项：parent: TRAIN_COMPLETE；parent: EVAL_COMPLETE；v3: TRAIN_COMPLETE；v3: EVAL_COMPLETE

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
