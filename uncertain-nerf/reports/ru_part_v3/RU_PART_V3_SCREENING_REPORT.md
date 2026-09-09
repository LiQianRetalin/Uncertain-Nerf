# RU-PART-V3 Garden 开发场景筛选报告

状态：**COMPARISON_INCOMPLETE**

实现仅增加 0.8 × mean((1−M)C × RGB绝对误差)，step 500 开启。基础 RU/DSSIM/责任头和常规 ADC 保留。
此处协议字段描述固定设计；是否完成训练以 training_actually_completed 和阶段状态文件为准。

阶段进度（用户提供的服务器输出）：GPU 0上的`smoke-parent`与修复后的`smoke-v3`均达到`SMOKE_COMPLETE`，exit_code=0、last_step=599，标准checkpoint存在且可加载，Gaussian数均为138766。配对profiler中位数为16.2875/18.0096毫秒，比值1.105733（增加10.57%），超过1.08的实现开销检查线。两次600步smoke不代表正式30k训练完成，也不改变下列最终验收项状态。

首次V3尝试曾在launcher参数检查阶段失败：`FAILED / exit_code=1 / last_step=-1`。`--track-cache`被旧PART参数保护误拒绝的问题修复后，47项相关CPU测试通过，服务器后续V3 smoke已完成。最新checkpoint SHA-256为`e8eba8d15f9ca3da2c19497683e8c5c252df7c5c6922ad29abbe2ffc9853d20e`；训练统计11.5152秒、子进程总墙钟43.9004秒，两者均不能代替配对profiler或完整30k成本验收。

| 指标 | B1 历史参考 | 可比 Parent | V3 | V3−Parent |
|---|---:|---:|---:|---:|
| psnr | 27.716728 | 未测 | 未测 | 未测 |
| ssim | 0.873957 | 未测 | 未测 | 未测 |
| lpips | 0.068151 | 未测 | 未测 | 未测 |
| DSC07988_psnr | 20.491301 | 未测 | 未测 | 未测 |

固定恢复门：psnr=NOT_ASSESSABLE；ssim=NOT_ASSESSABLE；lpips=NOT_ASSESSABLE；DSC07988_psnr=NOT_ASSESSABLE

Parent 质量保护：psnr=NOT_ASSESSABLE；ssim=NOT_ASSESSABLE；lpips=NOT_ASSESSABLE

资源门：gaussian_count=NOT_ASSESSABLE；training_time_ratio=NOT_ASSESSABLE；single_rasterization=NOT_ASSESSABLE；fps_ratio=NOT_ASSESSABLE

早期证据激活：`NO_Q_SAMPLED / SUPERVISION_INACTIVE_AT_AUDITED_STATE`。两张固定训练图C非零，但Mask拒绝率=0，因此Q=0；固定张量功能检查梯度误差=0。正式训练期激活与最终覆盖仍未测。实现已消除重复Q计算并改用当前视图的锁页内存异步传输，51项相关CPU测试通过，CUDA开销复测待执行。

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
