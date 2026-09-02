# PURI-GS-RU-v1 方法来源审计

审计状态：`RU_PROVENANCE_COMPLETE`

审计对象为 `dev` 的 `c014f3148326eb3b6302ef4a4d08fd0286230764`。该状态表示 RU-v1 的代码构成与已发现的直接上游来源已经逐项登记；它不表示这些模块可以作为项目原创贡献。

## 1. RU 的精确定义

RU-v1 是固定 gsplat v1.5.3 B1/AbsGrad 主干上的**训练期 RobustSplat 适配**：B1 保持 30,000 步、seed 42、factor 4、SH 3、SSIM 0.2、`DefaultStrategy(absgrad=True, grow_grad2d=0.0006)`；RU 增加冻结的 DINOv2 ViT-S/14-register 特征、384→16→1 静态掩码 MLP、历史残差与 DINO 特征共同监督、16→36 级联尺度，以及 10k–20k 延迟增密。正式 checkpoint 仍只有 Gaussian splats，独立推理不加载 DINO、mask head 或 feature cache。

简化流程：

```text
读取训练图与已缓存的 GT DINO 特征
for step = 0 ... 29999:
    标准 gsplat 全分辨率渲染
    if step < 20000:
        额外做一次 224×224 渲染供 coarse DINO 特征使用
    静态 MLP(GT DINO feature) -> soft static probability
    threshold 0.25 + 7×7 erosion -> detached hard static mask
    step < 500: 标准 B1 L1+DSSIM
    step >= 500: hard-mask L1+DSSIM
    Gaussian loss backward
    用 detached render 的 DINO cosine 与 0.60/0.80 历史残差区间监督 MLP
    在 opacity reset 后固定窗口内暂停 MLP update
    10000 <= step < 20000 且 step % 100 == 0: stock gsplat grow/prune
    step in {15000, 18000}: stock opacity reset
保存 splats-only checkpoint；mask/head/histogram 仅保存到 aux
```

## 2. B1 到 RU 的完整算法差异

| 模块 | B1 | RU-v1 | 来源判定 |
| --- | --- | --- | --- |
| Gaussian 主干 | gsplat DefaultStrategy | 相同 | `GSPLAT_STOCK` |
| 增密梯度 | AbsGrad，0.0006 | 相同 | gsplat/AbsGS |
| 增密时序 | stock B1 时序 | 10k 开始，20k 停止，每 100 步 | `ROBUSTSPLAT_DERIVED` |
| opacity reset | stock 时序 | 15k、18k | `ROBUSTSPLAT_DERIVED` |
| 图像损失 | 全图 L1+DSSIM | 500 步后改为 0.25 threshold、7×7 erosion 的 masked L1+DSSIM | `ROBUSTSPLAT_DERIVED` |
| 语义特征 | 无 | 冻结 DINOv2 ViT-S/14-register | `OTHER_PUBLISHED_METHOD_DERIVED` |
| mask head | 无 | Linear(384,16)-ReLU-Linear(16,1)-Sigmoid | `ROBUSTSPLAT_DERIVED`；上游谱系含 SpotLessSplats |
| residual 监督 | 无 | 10k-bin、0.95 历史直方图、0.60/0.80 分位数、3×3 邻域 | `ROBUSTSPLAT_DERIVED` |
| feature 监督 | 无 | `clamp(2*cos-1,0,1)`，权重 0.5 | `ROBUSTSPLAT_DERIVED` |
| 静态先验 | 无 | `2 exp(-step/2000) mean(1-mask)` | `ROBUSTSPLAT_DERIVED` |
| 尺度级联 | 无 | step <20k 用 16×16，否则 36×36 | `ROBUSTSPLAT_DERIVED` |
| UBP | 无 | **仍无** | 不得称为 SpotLessSplats UBP |
| 训练/推理解耦 | 无 RU 辅助状态 | 离线 cache、梯度隔离、aux 独立保存、splats-only 推理 | `PROJECT_ORIGINAL` 工程适配 |
| 审计输出 | 标准结果 | RU 曲线、DINO 用时、标准推理路径验证 | `MEASUREMENT_ONLY` |

逐项字段、代码行、适配方式、消融名和许可证备注见 [ru_method_provenance.json](ru_method_provenance.json)。

## 3. 与 RobustSplat 的代码级对应关系

以下项目不是基于名称猜测，而是与 RobustSplat 官方仓库 `a130281d6d0c004032a9a57e8d6a14962d9836d3` 的 `train.py`、`utils/mask_utils.py` 和 `arguments/__init__.py` 对照所得：

- DINOv2 `dinov2_vits14_reg`、384→16→1 MLP、Adam `1e-3`；
- mask 从第 500 步开始，阈值 0.25，7×7 erosion；
- 10,000-bin residual histogram，`0.95×history + current`，0.60/0.80 分位数与 3×3 规则；
- cosine target、0.5 cosine + 0.5 residual + 2.0 decaying static prior；
- 16/36 特征尺度、20k 切换；
- 10k 延迟 Gaussian growth、20k 停止、15k 起 opacity reset、reset 后 mask 暂停。

RU 是跨代码库适配而非逐文件复制，但部分数学表达和实现结构非常接近。RU 还存在明确适配差异：早期只有 DINO render feature 使用额外 224×224 rasterization，residual evidence 仍来自全分辨率；zero-based 第一次 refine 为 step 10000；post-reset pause 为完整的 300 个 step；B1 AbsGrad/0.0006 被保留。

## 4. SpotLessSplats 与 UBP

SpotLessSplats 是 RobustSplat mask MLP、robust residual masking 的上游谱系。RU 的 MLP 结构、`.data` max-weight regularizer、历史直方图和 3×3 inlier 规则均能追溯到这一谱系。

但 RU **没有 UBP**：没有 `ubp` rasterizer 标志、没有 squared-gradient 利用率统计、没有 `ubp_thresh`，pruning 仍调用 gsplat `DefaultStrategy` 的 opacity/scale pruning。分类 `SPOTLESSSPLATS_UBP_DERIVED` 在当前 RU 中没有活动项。

## 5. 项目自行设计或适配的内容

可以登记为项目实现工作的内容：

- 将 RobustSplat 训练配方移植到固定 gsplat v1.5.3 `DefaultStrategy`，并与 B1 AbsGrad 组合；
- 离线缓存 GT DINO feature，并校验 DINO weight SHA、repository commit、payload shape 和有限性；
- 显式隔离 Gaussian photo gradient 与 mask/DINO gradient；
- mask/head/histogram 保存到 `aux`，标准 Gaussian checkpoint 和独立推理完全不依赖它们；
- 仅对送入 DINO 的 detached render 副本做 `[0,1]` clamp；
- 训练曲线、耗时和独立推理路径审计。

这些首先是工程适配、复现与测量设计，不应在没有新算法消融和更广泛文献审查时写成新的鲁棒重建原理。

## 6. 训练与推理开销

归档 Phase-R 记录中，mask head 为 6,177 个参数。DINO render feature 占 Android/Room 训练时间的 34.71%/35.77%；总训练时间比值记录为 1.010×/1.025×。这些是已有运行观察，不是本阶段的新门禁。

推理期没有 DINO、head 或额外 rasterization。Room 严格 3+3 的 RU/B1 FPS 中位比为 1.6071，性能变化来自 Gaussian 数量变化，而非额外 RU 推理模块。

## 7. 论文声明边界

当前可以声明：

- 对已公开 RobustSplat 配方进行了 gsplat v1.5.3/AbsGrad 适配，并建立严格的配对、泛化和效率审计；
- 离线缓存、梯度隔离、标准 checkpoint 推理解耦和审计属于项目实现与可复现性工作；
- 只有后续 Android、Garden、On-the-go 门全部通过后，才能陈述对应的实证结果。

当前不可以声明：

- DINO mask MLP、残差 histogram、16→36 级联、0.25/7×7 mask 或 10k 延迟增密是 PURI-GS 原创；
- RU 使用了 SpotLessSplats UBP；
- RU 包含新的 uncertainty 或推理期鲁棒模块；
- 在所有规定门通过前称 RU 为通用鲁棒主干。

## 8. 许可证审计

当前仓库根 `LICENSE` 是 MIT，但最接近的 RobustSplat 上游实现使用 Gaussian-Splatting 的研究/非商业许可证；现有 RU 文件没有逐组件上游归属头。SpotLessSplats、gsplat 与标准 DINOv2 为 Apache-2.0，AbsGS 仍需按其仓库声明处理。

因此，在论文代码公开或任何外部分发前必须：补充 RobustSplat、SpotLessSplats、DINOv2、AbsGS 的明确引用与代码归属，并由维护者或法律审查确认本地重实现是否受 RobustSplat 仓库许可证约束。不得把当前根 MIT 文件当作已经消除该风险。

