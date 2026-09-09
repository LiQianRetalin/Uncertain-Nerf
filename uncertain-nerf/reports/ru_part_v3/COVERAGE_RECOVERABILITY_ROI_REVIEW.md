# V3 准备回读与 ROI 可用性复核

2026-09-09 收到用户从服务器导出的 `review_bundle.zip`。本次复核状态为 **ROI_NOT_READY**：在固定优化视图与所提供训练候选图中，尚未找到可清晰确认、四视图共同可见且能产生新增监督的同一静态表面。此为静态 ROI 选择尚未完成，**不代表局部不可恢复，也不是 Va/Vb/O 的实验结论**。

## 已核实的准备结果

压缩包完整，29 个成员的 CRC 检查通过。25 张 PNG 的 SHA 与 preparation.json 所记录的 SHA 全部一致。实际归档 SHA：

```text
7ddfc3423cff2e356d765da71b1e6e47aeff704937be9ec239ba4c26704330a3
```

源 checkpoint SHA 与协议固定值一致，preparation.json 的 problems 为空。环境记录为 GPU 0、NVIDIA L20、Torch 2.4.0+cu121、CUDA 12.1、gsplat 1.5.3+pt24cu121。原 head 和 fine 缓存产生的终态 M 在 161 张训练图上全部可用，test 图像解码尝试数为 0。

准备计数为 2 次 source render、0 次 Gaussian backward、0 次 optimizer update、0 次 head update、0 次 topology event。临时 CUDA 更新预检与三组正式诊断尚未由本次复核启动。该包未包含 prepare.status.json，因此这里只确认准备产物和记录，不额外宣称已核实服务器后台进程的最终退出状态。

means 终态常量学习率为 `1.962743359643857e-06`，其余参数组为 scales=0.005、quats=0.001、opacities=0.05、sh0=0.0025、shN=0.000125。

## 覆盖统计

| 统计项 | 161 张终态训练图结果 |
|---|---:|
| C 非零像素比例，逐图等权 | 1.608740% |
| M 拒绝像素比例，逐图等权 | 1.206149% |
| Q 非零像素比例，逐图等权 | 0.011493% |
| mean C | 0.004203090918 |
| mean Q | 0.000026268068 |
| Q 非零的训练图数量 | 11 / 161 |
| 全部拒绝像素中 Q 非零的比例 | 0.952846% |
| 全部 C 权重落在拒绝区的比例 | 0.624970% |

这些数值来自原始 coverage CSV/JSON，重新累计核对：175,406,280 个像素，2,115,661 个拒绝像素，20,159 个 Q 非零像素。C 权重总和等于接受区 C 权重与拒绝区 C 权重之和。所有图尺寸相同，因此本批逐图等权与像素加权均值相同。

固定两张优化图的情况：

| 图像 | M 拒绝比例 | Q 非零比例 |
|---|---:|---:|
| DSC07987.JPG | 14.886919% | 0% |
| DSC07989.JPG | 0.843613% | 0.071961% |

历史窗口的激活步数分别为 518、1012、735，总计 2265，与原训练记录一致，占 29500 个启用阶段更新的 7.677966%。窗口新增 loss 只有稀疏采样点，不能报告完整窗口 loss 总和或把采样均值当作完整均值。

## 原图与终态 M 的视觉复核

下图只为显示，把导出 M 的拒绝区域着红色。没有从 PNG 恢复数值 M，没有把该显示图用于 loss、定量干预判定或训练。

![两张固定优化图的顶部拒绝区域](E:/7-DataSet/ru_part_v3/coverage_recoverability_diag/review_bundle_7ddfc3423cff/inspection/mask_scope_review.png)

DSC07987 的拒绝范围包含天空、植被及后方房屋上部。房屋上部是清晰建筑候选，但在 DSC07989 原图中已不在画面内，不能为固定两张优化图绘制该同一表面的有效非空 S。

DSC07989 的拒绝范围主要在右上角，包含枝叶和收拢遮阳伞的局部。该伞未在 DSC07987 中清晰可见；不能用另一株树干或前景枝叶冒充同一静态表面，也不能仅因被 M 拒绝就把枝叶标为静态。

桌面、陶罐、前景铺地和可见砖墙是共同可见的静态候选，但在当前导出 M 中呈接受区。对 M=1 的像素，无论 C 或 S 如何，新增权重 `(1-M)(1-C)S` 均为 0。为这些区域画四个框并不能自动形成有效 O 干预。

几何候选前两名为 DSC08024.JPG、DSC08025.JPG。它们清楚显示同一桌面和陶罐，但不能补上固定两张优化图之间缺失的房屋/遮阳伞对应关系。其他 6 个候选的概览也已查看；此次没有根据训练残差挑选检查图。

![固定优化图与前两名几何候选，仅为原图复核，没有 S 边界](E:/7-DataSet/ru_part_v3/coverage_recoverability_diag/review_bundle_7ddfc3423cff/inspection/four_training_views_review.png)

## 状态和下一步边界

当前为 **ROI_NOT_READY**，不是数值判定的 `NO_ORACLE_INTERVENTION`。本包未包含 prepared 目录下的原始 M/C/source RGB 数组，也尚未产生合法 S，因此不能声称已经完成正式的新增 Q 权重和或残差加权量检查。

本次没有生成正式 S、多边形 proposal、静态标签确认或预注册，没有启动 Va/Vb/O，没有修改训练代码、源输出、固定优化视图或 Mask。原 V3 仍为 `QUALITY_RECOVERY_FAIL / NO_GO`，历史重放仍为 `REPLAY_NOT_EQUIVALENT`。

按当前协议，后续需要先有能满足共同静态表面与有效新增监督要求的可复核 ROI。如果无法找到，应保留本覆盖审计和 ROI_NOT_READY 记录，结束这次局部诊断准备；不能以全图 S、树叶标签、替换优化图或改变 M 来绕过该条件。

完整复核数据保存在 [roi_eligibility_review.json](E:/7-DataSet/ru_part_v3/coverage_recoverability_diag/review_bundle_7ddfc3423cff/roi_eligibility_review.json)，原始压缩包和解压文件保存在同目录。本次只补充文档和本地复核产物，不需要重新训练或重新执行 prepare。
