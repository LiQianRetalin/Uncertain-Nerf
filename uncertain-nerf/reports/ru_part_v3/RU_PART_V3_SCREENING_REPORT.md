# RU-PART-V3 Garden 开发场景筛选最终报告

**最终状态：QUALITY_RECOVERY_FAIL / NO_GO。本轮训练和独立评测正常完成；相对历史标准RU Parent有小幅整体改善和固定孔洞ROI改善，但四项固定质量恢复门全部失败，不晋级。**

本报告依据用户于2026-09-09提供的服务器状态和JSON字段回读，以及此前已审计的实现。没有直接登录服务器或重新运行实验。原始回读和复算结果保存在[本地证据归档](E:/7-DataSet/ru_part_v3/server_readback_20260909.json)，来源明确标记为用户粘贴的服务器结果。

V3在GPU 0上从step0训练至29999，`TRAIN_COMPLETE / exit_code=0`；独立评测为`EVAL_COMPLETE / exit_code=0 / error=null`。服务器报告的`missing=[] / invalid=[]`，质量恢复门均为false，Parent的PSNR、SSIM、LPIPS保护门均为true。时间门仍为null；文件完整性检查没有报缺失，不等于所有资源门可判定。

| 质量指标 | B1历史参考 | 历史Parent重评 | V3 | V3−Parent | 固定恢复门 | V3距恢复门 | 判定 |
|---|---:|---:|---:|---:|---:|---|---|
| PSNR ↑ | 27.716728 | 26.650572 | 26.808758 | +0.158186 | ≥27.566728 | 还需提高0.757970 dB | FAIL |
| SSIM ↑ | 0.873957 | 0.857305 | 0.858221 | +0.000916 | ≥0.868957 | 还需提高0.010736 | FAIL |
| LPIPS ↓ | 0.068151 | 0.090506 | 0.090359 | −0.000147 | ≤0.073151 | 还需降低0.017208 | FAIL |
| DSC07988 PSNR ↑ | 20.491301 | 14.080384 | 14.745125 | +0.664741 | ≥19.491301 | 还需提高4.746176 dB | FAIL |

B1列使用协议规定的历史值。Parent是原标准RU的同一29999检查点，在当前GPU 0、warmup=10下重评，平均PSNR与原结果之差为0。当前记录支持相对Parent的单次开发筛选比较；这些小幅均值变化不能证明统计显著性。Garden整体与关键视角均未达到恢复要求，尤其DSC07988仍相差4.746 dB。不能把Parent本来低于B1目标解释为Parent发生历史回退。

当前实现仅在原RU Gaussian损失上增加`0.8 × mean((1−M.detach()) × C.detach() × |render−target|)`，step500开启，使用完整RGB元素均值。M沿用实际最终Mask，C沿用原`track_evidence_binary`及双线性上采样。实现版本为`v3-single-q-pinned-transfer-v1`；未改责任头、Mask规则、DSSIM、普通ADC或原调度。以下记录支持新增项已经进入实际训练，不把它解释为因果重放已经成功。

| 运行检查 | 实际结果 | 解释 |
|---|---:|---|
| Gaussian backward | 30000 | 每步一次 |
| 训练rasterization | 50000 | 与Parent既有训练路径的预期计数一致 |
| birth / probe / scoring / cap / lineage | 全部0 | 禁用路径未被调用 |
| extra_training_rasterization / per_step_matching | 全部0 | 未增加这两类调用 |
| 责任头更新 | 29400 | 与两次reset后各暂停300步的调度一致 |
| 已记录参数梯度检查 | step500、599均有限且存在非零梯度 | 这是两个检查时点，不能写成29999步末梯度检查 |
| 标准检查点 | 存在且可加载 | SH3、step29999、N2266597 |

新增监督在完整训练中为`ACTIVE`：2265个训练步出现非零Q，占step500..29999的29500个可启用步骤约7.68%。这里的`active_samples`统计含任意非零Q的训练更新次数，不是有效像素数量。附加损失累计值为0.0874202773；缺少同口径基础损失累计值，不能由该数值单独量化它相对主损失的强度。600步smoke的`NO_Q_SAMPLED`只适用于早期状态，不能延伸为30k全程未激活。

最终证据范围为`FINAL_TRAINING_STATE`，只涉及预先固定的两张训练图：

| step29999训练图 | C非零像素比例 | Mask拒绝比例 | Q非零像素比例 | 最高20%残差像素中的Q覆盖 | mean Q |
|---|---:|---:|---:|---:|---:|
| DSC07987.JPG | 2.002423% | 14.886919% | 0% | 0% | 0 |
| DSC07989.JPG | 2.610420% | 0.843614% | 0.071961% | 0.289129% | 0.000095163 |

在DSC07987上，C非零与Mask拒绝区域没有形成非零Q交集；在DSC07989上，交集和高残差区域覆盖都很小。`ACTIVE`汇总只要求两图中至少一张有非零Q，不表示两张图均获得充分监督。这些记录直接显示诊断视图上的有效覆盖有限，与局部小幅恢复的结果相符；尚不能证明它是整体恢复不足的唯一原因，也不能由两张图代表全部161张训练图或测试视角DSC07988。没有根据测试结果反向修改C、Mask阈值或训练规则。

固定孔洞ROI复用历史定义`old top_connected_component((B1>=.8)&((B1−RU_TAR)>=.5))`，共77812像素，占图像7.142123%；没有从V3结果重新挑选区域。低alpha统计阈值保持`alpha≤0.3`。

| 固定ROI指标 | Parent | V3 | 变化 |
|---|---:|---:|---|
| RGB绝对误差均值 ↓ | 0.517779 | 0.448519 | 降低0.069260，即13.38% |
| alpha均值 ↑ | 0.200694 | 0.293748 | 增加0.093054 |
| 低alpha像素数 ↓ | 58168 | 48358 | 减少9810，即16.86% |
| 低alpha面积占ROI比例 ↓ | 74.7545% | 62.1472% | 下降12.6073个百分点 |

固定ROI的误差和alpha覆盖指标均改善，说明保留了有用的局部信号。不过仍有62.15%的ROI像素处于低alpha范围，关键视角质量也远低于目标，不能描述为孔洞已修复。alpha变化本身不证明几何正确。此处没有混用S_cov等旧误差分解份额与孔洞面积。

ROI来源标识：B1数组SHA-256 `360ce99d7246478debf005c2f6c3f675c044c65d150a9ae261d08e8e262cb736`；RU-TAR数组SHA-256 `7a36e8b0bac777f5d4f1d734d0a4529703d9dcac23d8f490b5ccef5492caa079`。记录的`bbox_rc=[0,118,120,1248]`原样保留。

| 资源门 | 本轮结果 | 固定要求 | 判定 |
|---|---|---|---|
| Gaussian数量 | 2266597；比Parent少7600，距上限余45405 | ≤2312002 | PASS |
| 独立推理rasterization | 1 | 1 | PASS |
| FPS / Parent | 143.230355 / 143.370501 = 0.999022 | ≥0.95 | PASS |
| 同口径完整训练时间 / Parent | 未建立可比时间基线 | <1.10 | NOT_ASSESSABLE |

两次独立推理均使用GPU 0、warmup10、24个原始计时样本；报告中评测口径比较通过。V3 FPS约低0.098%，本轮记录基本持平。Parent/V3的p50延迟为6.898403/6.906867毫秒，p95为7.451010/7.456243毫秒；推理显存为1.854206/1.850505 GiB。

V3训练段时间1212.198883秒，训练峰值显存4.667715 GiB；历史Parent训练段时间1347.454996秒，峰值4.680006 GiB。历史训练GPU为1，本轮为0，训练计时及记录方式的可比性未成立，因此展示原始值但不报告“训练提速百分比”或时间预算通过。子进程总墙钟1244.831186秒还覆盖初始化、加载及证据导出，另行保存，不混入训练段时间。

本轮缓存准备0.580099秒；原静态缓存构建561.269638秒，已复用而非本轮重建。训练段加原缓存构建成本为1773.468521秒，该总数属于包含既有缓存构建成本的记账口径，不是本轮新增墙钟。此前缓存清单记录的特征提取8.505113秒保留为独立历史成本，不假定它与构建成本互不重叠或再加一次。

可追溯路径以服务器目录`/home/chenglong/Uncertain-Nerf/uncertain-nerf`为根：

- V3配置：`configs/puri_gs_ru_part_v3_v3_garden30k.yaml`；Parent配置：`configs/puri_gs_ru_part_v3_parent_garden30k.yaml`。
- 数据：`data/mipnerf360/360_v2/garden`，factor4、train161/test24、seed42；规范化/SfM/图像标识见本轮`v3_input_manifest.json`。
- C缓存：`data/PURI-GS-derived/static_tracks/garden_factor4_v1/garden_factor4_static_tracks.pt`；完整缓存标识及配置见`logs-puri/ru_part_v3_screening/v3/v3_run_manifest.json`。
- 历史Parent：`logs-puri/ru-generalization-rerun-9e292309/garden_ru_30k`，源commit `9e29230952be700dc5b527008e2f60f05b82717d`；本轮重评目录`logs-puri/ru_part_v3_screening/eval-existing-parent`。
- 本轮训练/评测：`logs-puri/ru_part_v3_screening/v3`和`logs-puri/ru_part_v3_screening/eval-v3`，逐图指标保存在独立评测目录；完整报告JSON保存在`logs-puri/ru_part_v3_screening/screening_result.json`。
- 标准V3检查点：`logs-puri/ru_part_v3_screening/v3/ckpts/ckpt_29999_rank0.pt`；SHA-256 `14563fbbcfbb722879929be31fd19ad5e6a02939e8d2409e6a6855ce909d417c`。
- 本轮GPU0 UUID：`GPU-e75e3686-bd4e-bbaf-9417-28a4a31d5744`。

历史Parent复用是基于原配置/路径/split、实际标准策略及固定检查点复现的记录审计。历史图像/SfM字节哈希和30k相机序列未记录，当前哈希不是历史字节身份的追溯证明；此前重评没有直接记录当时的GPU UUID。这些限制继续保留。本地归档含摘要而非服务器完整逐图文件、叠图或检查点；未声称本机重新核验了这些二进制产物。

本轮结论限定为Garden单seed开发筛选：新增L1监督确实激活，整体指标与固定ROI有改善，但恢复幅度不足，四项固定质量门均失败。单次结果不能推出统计显著性、训练轨迹因果等价、跨场景泛化、长期收敛或静态标签正确性。最终目标中的Android、On-the-go和Room提升未在本轮验证。

**按既定停止条件结束本轮。保留配置、日志、检查点、独立评测、证据与负结果报告；不自动重训、改阈值、放宽门槛、多seed、跨场景或启动下一算法。**

```text
BRANCH = ru-part
TRAINING_FROM_SCRATCH = true
RESUMED_FAILED_SNAPSHOT = false
HISTORICAL_REPLAY_STATUS = REPLAY_NOT_EQUIVALENT
NEW_ALGORITHM = RU-PART-V3
DEDICATED_BIRTH_ENABLED = false
PART_CAP_ENABLED = false
EXTRA_TOPOLOGY_SCORING_ENABLED = false
MULTI_SEED_RUN = false
CROSS_SCENE_RUN = false
NEXT_ALGORITHM_STARTED = false
TRAINING_ACTUALLY_COMPLETED = true
EVALUATION_ACTUALLY_COMPLETED = true
FINAL_STATUS = QUALITY_RECOVERY_FAIL / NO_GO
```
