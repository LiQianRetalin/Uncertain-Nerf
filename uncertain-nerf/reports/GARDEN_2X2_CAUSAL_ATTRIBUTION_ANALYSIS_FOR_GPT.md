# PURI-GS-RU Garden 2×2 因果归因分析报告

## 0. 报告用途与边界

本报告用于向后续 GPT 提供 Garden 退化问题的实验事实和因果归因证据。

本报告只回答以下问题：

> Garden 上的质量下降，主要来自语义静态 Mask 子系统、延迟拓扑调度子系统、二者负交互，还是二者均独立造成损失？

本报告不提出新算法、不选择改进方案、不修改配置，也不提供下一阶段实施计划。后续方案讨论应作为独立任务进行。

## 1. 最终归因结论

唯一归因标签：

```text
GARDEN_TOPOLOGY_DOMINANT
```

结论的直接证据是：延迟拓扑调度在无 Mask 和有 Mask 两种条件下均产生了实质、同向且逐图配对置信区间不跨零的质量损失；当前 Mask 子系统在标准 B1 拓扑下的平均质量则接近中性。

跨条件归因证据：

```text
Mask 在两种拓扑下均造成实质同向损失 = False
拓扑在有无 Mask 时均造成实质同向损失 = True
```

这意味着 2×2 试验支持“拓扑子系统是 Garden 退化的主导来源”，但不支持把全部退化单独归因给 Mask，也不支持把结果仅解释为二者负交互。

## 2. 因果设计

两个因素定义如下：

- `M=0`：关闭 DINO、feature cache、mask head、mask loss 和 masked photometric loss，使用标准 B1 光度损失。
- `M=1`：保持正式 RU 的语义静态 Mask 子系统。
- `T=0`：使用实际 B1 DefaultStrategy 拓扑与 opacity-reset 行为。
- `T=1`：使用正式 RU 的延迟 densification、split、clone、prune、统计窗口和 opacity-reset 行为。

四个象限：

| 象限 | 方法 | M | T | 训练与评测状态 | 训练/评测 commit |
|---|---|---:|---:|---|---|
| Y00 | B1 | 0 | 0 | 复用既有 30k 结果 | `9e29230952be700dc5b527008e2f60f05b82717d` |
| Y01 | DG-only | 0 | 1 | 新增 30k 与独立评测 | `01e647a135e8a3b1e2d0b9bbd2c4003bb0e67785` |
| Y10 | Mask-only | 1 | 0 | 新增 30k 与独立评测 | `d30bc63ccd2bd8da59ccea3f629a109bd4f59787` |
| Y11 | RU | 1 | 1 | 复用既有 30k 结果 | `9e29230952be700dc5b527008e2f60f05b82717d` |

最终汇总与归因代码 commit：

```text
6e54008e244722b977a9a2b2c54e10207413b991
```

## 3. 固定实验条件与可比性

四组共同固定：

```text
scene = Garden
dataset = Mip-NeRF 360 Garden / COLMAP
train images = 161
test images = 24
data factor = 4
total steps = 30000
seed = 42
SH degree = 3
absgrad = true
grow_grad2d = 0.0006
gsplat = 1.5.3
PyTorch = 2.4.0+cu121
GPU = NVIDIA L20
```

四组训练和独立评测使用完全相同的 24 张测试图及顺序：

```text
DSC07956.JPG, DSC07964.JPG, DSC07972.JPG, DSC07980.JPG,
DSC07988.JPG, DSC07996.JPG, DSC08004.JPG, DSC08012.JPG,
DSC08020.JPG, DSC08028.JPG, DSC08036.JPG, DSC08044.JPG,
DSC08052.JPG, DSC08060.JPG, DSC08068.JPG, DSC08076.JPG,
DSC08084.JPG, DSC08092.JPG, DSC08100.JPG, DSC08108.JPG,
DSC08116.JPG, DSC08124.JPG, DSC08132.JPG, DSC08140.JPG
```

可比性审计均通过：

- 四组训练/测试划分和图像顺序一致；
- 四组固定基线字段一致；
- 每组训练 commit 与对应独立评测 commit 一致；
- checkpoint 均为 step 29999，且只包含标准 Gaussian 推理字段 `splats` 与 `step`；
- 独立评测不导入 DINO、不加载 mask head、不读取 feature cache；
- 独立评测 rasterization count ratio 为 `1.0`；
- DG-only 的拓扑事件与 RU 一致；
- Mask-only 的拓扑事件与 B1 一致；
- 新增两象限之间的训练代码语义等价性审计通过。

## 4. 实际拓扑事件与 Mask 更新

| 象限 | refine 事件 | 统计窗口 | opacity reset | post-backward 顺序 | Mask 更新/暂停 |
|---|---:|---|---|---|---|
| Y00 B1 | 144；step 600–14900，每 100 step | `[0,14999]` | 无实际事件 | Gaussian optimizer 之后 | 不适用 |
| Y01 DG-only | 100；step 10000–19900，每 100 step | `[0,19999]` | 15000、18000 | Gaussian optimizer 之前 | 不适用 |
| Y10 Mask-only | 144；step 600–14900，每 100 step | `[0,14999]` | 无实际事件 | Gaussian optimizer 之后 | 30000 次更新、0 次暂停 |
| Y11 RU | 100；step 10000–19900，每 100 step | `[0,19999]` | 15000、18000 | Gaussian optimizer 之前 | 29400 次更新、600 step 暂停 |

Y11 的 Mask 暂停区间为：

```text
15001–15300
18001–18300
```

Y10 没有实际 opacity-reset 事件，因此没有静默加入 Mask 暂停。

## 5. 四象限观测结果

质量指标中 PSNR、SSIM 越高越好，LPIPS 越低越好。

| 象限 | PSNR↑ | SSIM↑ | LPIPS↓ | Gaussian 数量 | 训练时间(s)↓ | 训练峰值显存(GiB)↓ | FPS↑ | p50/p95(ms)↓ | 推理显存(GiB)↓ | checkpoint bytes↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Y00 B1 | 27.716728 | 0.873957 | 0.068151 | 3,798,503 | 1631.620 | 5.646 | 32.907 | 10.810 / 11.594 | 2.993 | 896,449,250 |
| Y01 DG-only | 27.287037 | 0.859978 | 0.084880 | 2,708,397 | 931.863 | 4.634 | 33.518 | 7.879 / 8.660 | 2.161 | 639,184,226 |
| Y10 Mask-only | 27.744446 | 0.873914 | 0.068342 | 3,773,259 | 2225.062 | 6.730 | 32.363 | 10.800 / 11.445 | 2.978 | 890,491,746 |
| Y11 RU | 26.650572 | 0.857305 | 0.090506 | 2,274,197 | 1347.455 | 4.680 | 39.691 | 6.873 / 7.407 | 1.837 | 536,713,186 |

直接观察：

- Mask-only 与 B1 的平均质量非常接近；PSNR 略高，SSIM 和 LPIPS 略差。
- DG-only 在三项质量指标上均明显差于 B1。
- RU 在三项质量指标上均明显差于 Mask-only，也明显差于 B1。
- 延迟拓扑对应更少的 Gaussian、更小的 checkpoint、更低的推理显存及更低延迟，但同时对应明显质量损失。
- Gaussian 数量只能作为伴随观测，不能单独证明具体退化机制。

## 6. Garden clean tolerance

固定阈值：

```text
PSNR 下降不超过 0.15 dB
SSIM 下降不超过 0.005
LPIPS 增加不超过 0.01
任一测试图 PSNR 下降不超过 1.0 dB
```

| 对比 | 判定 | ΔPSNR | ΔSSIM | ΔLPIPS | 最差单图 ΔPSNR | 主要触发项 |
|---|---|---:|---:|---:|---:|---|
| DG-only − B1 | FAIL | −0.429691 | −0.013979 | +0.016729 | −1.928871 | 四项均越界 |
| Mask-only − B1 | FAIL | +0.027718 | −0.000042 | +0.000191 | −1.103930 | 仅最差单图 PSNR 越界 |
| RU − B1 | FAIL | −1.066156 | −0.016651 | +0.022355 | −6.410916 | 四项均越界 |

Mask-only 的 `FAIL` 需要谨慎解释：它的三个平均质量指标均处在 clean tolerance 内，唯一失败原因是最差单图 PSNR 为 `−1.103930 dB`，比 `−1.0 dB` 阈值多下降约 `0.103930 dB`。因此该结果不等价于“Mask 在标准拓扑下造成普遍质量下降”。

## 7. 四项条件效应与交互项

定义：

```text
Mask@T0       = Y10 − Y00
Mask@T1       = Y11 − Y01
Topology@M0   = Y01 − Y00
Topology@M1   = Y11 − Y10
Interaction   = Y11 − Y10 − Y01 + Y00
```

### 7.1 标量质量效应

| 指标 | Mask@T0 | Mask@T1 | Topology@M0 | Topology@M1 | Interaction |
|---|---:|---:|---:|---:|---:|
| PSNR↑ | +0.027718 | −0.636465 | −0.429691 | −1.093874 | −0.664182 |
| SSIM↑ | −0.000042 | −0.002673 | −0.013979 | −0.016609 | −0.002630 |
| LPIPS↓ | +0.000191 | +0.005626 | +0.016729 | +0.022164 | +0.005435 |

解释边界：

- `Topology@M0` 三项指标均恶化。
- `Topology@M1` 三项指标均恶化，且幅度大于 `Topology@M0`。
- `Mask@T0` 平均效应接近零。
- `Mask@T1` 三项指标均朝不利方向变化。
- 交互项在三项指标上均朝不利方向变化，说明组合存在额外负交互；但由于拓扑在 M=0 时已经独立造成实质损失，负交互不是唯一或主导归因标签。

### 7.2 标量效率与规模效应

| 指标 | Mask@T0 | Mask@T1 | Topology@M0 | Topology@M1 | Interaction | 方向说明 |
|---|---:|---:|---:|---:|---:|---|
| FPS | −0.544 | +6.173 | +0.611 | +7.328 | +6.717 | 越高越好 |
| Gaussian 数量 | −25,244 | −434,200 | −1,090,106 | −1,499,062 | −408,956 | 原始数量差 |
| p50 延迟(ms) | −0.011 | −1.006 | −2.932 | −3.927 | −0.995 | 越低越好 |
| p95 延迟(ms) | −0.149 | −1.253 | −2.934 | −4.038 | −1.104 | 越低越好 |
| 推理显存(GiB) | −0.015 | −0.323 | −0.832 | −1.141 | −0.308 | 越低越好 |
| 训练时间(s) | +593.442 | +415.592 | −699.757 | −877.607 | −177.850 | 越低越好 |
| 训练峰值显存(GiB) | +1.083 | +0.046 | −1.012 | −2.050 | −1.038 | 越低越好 |

这些结果说明质量归因与效率结果并不相同：延迟拓扑提高了压缩/速度相关表现，但 Garden 质量明显下降。本试验的归因标签依据质量规则确定，不依据 Gaussian 数量或 FPS 单独确定。

## 8. 24 张图的逐图配对统计

所有比较均使用相同的 24 张测试图。95% CI 使用固定 seed `42` 的 10,000 次 paired bootstrap；CI 对应 24 张图平均原始差的不确定性。

“有利图数”按指标方向计算：PSNR/SSIM 增加为有利，LPIPS 减少为有利。

| 效应 | 指标 | 平均差 | 中位数差 | 有利图数/24 | paired bootstrap 95% CI | 最差单图变化 |
|---|---|---:|---:|---:|---|---:|
| Mask@T0 | PSNR | +0.027717 | +0.026539 | 13/24 | [−0.101774, +0.136568] | −1.103930 |
| Mask@T0 | SSIM | −0.000042 | −0.000121 | 8/24 | [−0.000474, +0.000350] | −0.003111 |
| Mask@T0 | LPIPS | +0.000191 | +0.000235 | 8/24 | [−0.000239, +0.000596] | +0.001968 |
| Mask@T1 | PSNR | −0.636465 | −0.016258 | 10/24 | [−1.281943, −0.129652] | −5.499318 |
| Mask@T1 | SSIM | −0.002673 | −0.000639 | 4/24 | [−0.005002, −0.000851] | −0.020890 |
| Mask@T1 | LPIPS | +0.005626 | +0.000611 | 8/24 | [+0.001085, +0.011702] | +0.054319 |
| Topology@M0 | PSNR | −0.429692 | −0.327414 | 5/24 | [−0.645966, −0.238910] | −1.928871 |
| Topology@M0 | SSIM | −0.013979 | −0.011891 | 0/24 | [−0.017340, −0.011350] | −0.042082 |
| Topology@M0 | LPIPS | +0.016729 | +0.016517 | 0/24 | [+0.014657, +0.019166] | +0.037060 |
| Topology@M1 | PSNR | −1.093874 | −0.301227 | 4/24 | [−1.919449, −0.423749] | −6.669452 |
| Topology@M1 | SSIM | −0.016609 | −0.012377 | 0/24 | [−0.021858, −0.012452] | −0.055117 |
| Topology@M1 | LPIPS | +0.022164 | +0.016574 | 0/24 | [+0.016222, +0.029863] | +0.078754 |
| Interaction | PSNR | −0.664182 | −0.063996 | 8/24 | [−1.346428, −0.109586] | −5.757854 |
| Interaction | SSIM | −0.002630 | −0.001009 | 5/24 | [−0.004975, −0.000685] | −0.019875 |
| Interaction | LPIPS | +0.005435 | +0.000233 | 11/24 | [+0.000802, +0.011476] | +0.052351 |

逐图证据强度：

- `Mask@T0` 的三项 CI 均跨零，不能证明标准拓扑下存在稳定的平均 Mask 损失。
- `Topology@M0` 的三项 CI 均不跨零，并全部位于不利方向。
- `Topology@M1` 的三项 CI 均不跨零，并全部位于不利方向。
- `Mask@T1` 和交互项的三项 CI 也都不跨零并位于不利方向，表明延迟拓扑条件下 Mask 相关效应及二阶交互均为负面。

## 9. 代表图

选择规则：按 `RU−B1 PSNR` 从低到高排序全部 24 张图，再取六个等间距排序位置 `round(i×23/5), i=0..5`。

| 排序位置（0-based） | 测试索引（0-based） | 图像名 | RU−B1 PSNR |
|---:|---:|---|---:|
| 0 | 4 | DSC07988.JPG | −6.410916 |
| 5 | 8 | DSC08020.JPG | −0.731262 |
| 9 | 13 | DSC08060.JPG | −0.477282 |
| 14 | 18 | DSC08100.JPG | −0.276787 |
| 18 | 10 | DSC08036.JPG | −0.061394 |
| 23 | 21 | DSC08124.JPG | +0.230917 |

这些图仅用于覆盖从最差到最好变化的代表位置，不是人工挑选的成功或失败案例。

## 10. 因果判定链

1. 四组数据、固定训练字段、测试图顺序、checkpoint 加载和 evaluator 条件通过配对审计，因此可以计算四象限条件效应。
2. `Mask@T0` 的三个平均效应接近零，且三项逐图 paired-bootstrap CI 全部跨零；没有证据支持 Mask 在标准拓扑下产生稳定、普遍的质量损失。
3. `Topology@M0` 的 PSNR、SSIM、LPIPS 均实质恶化，三项 CI 全部不跨零。
4. `Topology@M1` 的 PSNR、SSIM、LPIPS 也均实质恶化，三项 CI 全部不跨零。
5. 因此“拓扑在有无 Mask 时均造成实质同向损失”为真，满足 `GARDEN_TOPOLOGY_DOMINANT` 的固定归因规则。
6. `Mask@T1` 和交互项也显示额外负面效应，但它们不推翻拓扑已经在 M=0 条件下独立造成损失这一事实。
7. 所以唯一归因标签为 `GARDEN_TOPOLOGY_DOMINANT`。

## 11. 本试验能够确定和不能确定的内容

### 能够确定

- 在本次固定 Garden、固定 seed、固定 30k 配置下，延迟拓扑子系统是质量下降的主导归因。
- 当前 Mask 子系统在标准 B1 拓扑下没有表现出稳定的平均质量损失。
- Mask 与延迟拓扑组合存在额外负交互。
- 延迟拓扑的效率/规模收益与 Garden 质量损失同时存在。

### 不能确定

- `T` 因素同时包含 densification 起止、统计窗口、split/clone/prune 时刻、reset 时刻和 optimizer 顺序等差异；本次 2×2 不能把拓扑归因进一步拆到其中某一个具体机制。
- 本次只有一个训练 seed；逐图 bootstrap 反映图像级配对不确定性，不代表跨随机种子的训练方差。
- 本试验只针对 Garden，不能把该因果标签直接外推到 Android、Room 或 Patio-High。
- Gaussian 数量与质量同时变化，但本试验不能仅凭相关性断言“Gaussian 数量减少”就是唯一机制。
- Mask-only 虽然平均表现接近 B1，但有一张图超过最差单图 clean-tolerance 阈值；不能把 Mask 描述为对每张图都无害。

## 12. 可追溯证据

服务器最终正式报告：

```text
/home/chenglong/Uncertain-Nerf/uncertain-nerf/reports/PHASE_R_GARDEN_CAUSAL_2X2.md
SHA-256: 0549beb63d2e96e6bfdf2c6b83c742dd809e3cb34a6be22718aac304c38053f6

/home/chenglong/Uncertain-Nerf/uncertain-nerf/reports/phase_r_garden_causal_2x2.json
SHA-256: dae9e5907b677aad4dc57bb3091a4d97c4bc29b86c4641a43cb7598742a1fe20

/home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/phase_r_causal-console/garden_causal_summary.log
SHA-256: 851b0a6a2ed1f83636671a30fe0de47e99d6a489d4549ddbc5b24788599f141e
```

最终审计标志：

```text
GARDEN-FINAL-PAIRING-AUDIT-PASS
GARDEN-FINAL-EFFECT-FORMULA-AUDIT-PASS
GARDEN-FINAL-ATTRIBUTION-RULE-AUDIT-PASS
GARDEN-FINAL-STOP-BOUNDARY-AUDIT-PASS
GARDEN-FINAL-REPORT-INDEPENDENT-AUDIT-PASS
GARDEN-2X2-CAUSAL-EXPERIMENT-COMPLETE
GARDEN-FINAL-ATTRIBUTION-FINISHED-SUCCESSFULLY
```

## 13. 强制停止声明

本次未执行 Garden 3×3，未运行额外随机种子，未实现或启动阶段 U，未编写 RU 新版本算法，也未修改正式 RU 配置。

本报告在因果归因处结束，不包含新方案或改进方向。
