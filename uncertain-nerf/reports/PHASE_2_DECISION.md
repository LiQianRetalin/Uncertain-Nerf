# PURI-GS Phase 2 最终决策

## 最终结论

```text
A1_REJECT
```

该结论只适用于当前唯一 A1 配置和本阶段固定协议，不外推为所有责任建模方法均
无效。结论由预先定义的门禁产生，不进行结果后调参。

## 输入证据

| 证据 | 结论 |
|---|---|
| Android 19 图逐图配对 | `PAIRWISE_STABLE` |
| Responsibility 8 帧选择性 | `MODERATE_EDGE_SENSITIVITY` |
| garden/room clean 执行 | `PASS_EXECUTION` |
| clean 主要基线 | B1，`efficiency-strong baseline` |
| clean 数值门禁 | `CLEAN_FAIL` |
| clean 18 图视觉抽样 | 未见灾难性损伤，但不能推翻数值失败 |

Android 上，A1 相对 B1 的 PSNR/SSIM/LPIPS 均跨图稳定改善，PSNR 为
15/19 张改善，Bootstrap 95% CI 为 `[+0.106852,+0.429576] dB`。责任图
没有被判为高边缘敏感，但普通静态高频结构受到中等程度响应。

Clean 场景中，B1 被选为主要基线。A1 相对 B1 的两场景平均 PSNR 为
`-0.172801 dB`，room 为 `-0.449688 dB`。两者分别越过 `-0.15 dB` 总体门
和 `-0.30 dB` 单场景门，直接触发 `CLEAN_FAIL`。

## 决策矩阵

预先定义的矩阵规定，只要 clean 门失败即为 `A1_REJECT`。Android 的稳定改善、
更少 Gaussian、更低显存和更高 FPS 都不能覆盖这一硬失败。因此不使用
`A1_CONFIRMED_CANDIDATE` 或 `A1_BORDERLINE_RETAIN`。

## 后续动作

1. 冻结当前 A1，保留为历史消融；
2. 不进入 A2；
3. 不实现基于 `1-q` 的 A3；
4. 不修改 threshold、min weight、warm-up、pooling、优化器或学习率；
5. 不自动生成 A1.1，不运行 30k 或第二随机种子；
6. 不上传或运行第二个动态场景来挽救本次失败；
7. 若以后重启该研究方向，先重新设计“动态选择性”证据与机制，再建立新的、
   预先注册的阶段协议。

当前阶段已经到达终止条件，无需继续服务器 GPU 实验。

