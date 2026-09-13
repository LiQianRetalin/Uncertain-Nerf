# 工作包 4：OAC 有界机制与可实现性检查

日期：2026-09-13

状态：`COMPLETE_LIMITED`（执行完成；不是机制阳性或正式训练验证）
直接 Parent：原 GS-RU，不更换方法、不新建训练身份；P05/R 未启动。

## 1. 执行边界与主要结论

在冻结 Android RU 30k 与 Room 历史 factor4 RU 的**终点 step 29999**，各用按 TRAIN 文件名等间距冻结的 8 张训练图，构造 A 真 RU Mask、B 全接受、C 固定 8×8 删失、D 首图中央人工误接受。使用同一 512 个按首图可投影集合等间距确定的 Gaussian ID；每个探针的贡献权重扫描原 tile 深度顺序中的**所有前方遮挡者**，没有逐 Gaussian 重渲染或 N×像素矩阵。比较原 RU `G0`、Pixel 式、ReAct 式、简单 `a·u` 和固定 OAC，均复用每条件的一次 photometric backward；没有参数更新或拓扑事件。全部已扣诊断 F/B **231/300**，成功 229，失败 2 次仍扣账；另两次过严的验证断言所触发的完整核验调用也在 231 内。使用 L20 GPU 7，未占用原有 GPU 6 任务。

结论是：原计数确实允许“有投影但无像素贡献/无接受监督”被计入；Android 终点真实 Mask 也呈现可量化的选择性，Room 主要是模型遮挡/投影混杂。然而 OAC 在这个小探针上相对简单控制只有很小的排序稳定性差异；D 还显示“同条件相对 G0 至多 2 倍”**不能**限制条件间最大倍率。更关键的是两个场景都只有配套终点状态，缺失 step 10000–19999 的同一步 Gaussian+Mask+残差快照，因此不能把终点结果变成增密期证据或宣称 P05 训练收益。本包**不建议仅凭现有证据开启 P05 正式四场景筛选**；由方案助手决定是否接受这一有限结论、要求合法的训练期状态，或停止当前 OAC 定义。

## 2. 冻结身份、输入与资产等级

起点本地/服务器均为 `ru-part`、HEAD `41a50f5f3ba7429b9b41ea771e13a8ebc9a42780`、工作区清洁；没有 switch、reset、rebase。Android 源训练 commit `9e29230952be700dc5b527008e2f60f05b82717d`，checkpoint SHA-256 `11737109…6f73a`；Room 源 commit `da5089cd34bad85780e974f732c2bc722b05c77f`，本包现场算得 checkpoint SHA-256 `fc3cf97b…29d6`。两次运行各有同一步 Mask 头、残差直方图、配置、split 与 DINO 特征缓存，但 `ckpts/` 只有 29999；均分级 `MATCHED_TERMINAL_STATE`。Room 绝不冒称 factor2 或增密期恢复。详细路径、完整哈希、16 张选图的索引/字节 SHA、公式参数、D 注入像素及脚本版本见 `diagnostic_manifest.json`。

Android TRAIN 122 张，选中索引 `0,17,34,51,69,86,103,121`；Room TRAIN 272 张，选中 `0,38,77,116,154,193,232,271`。选图在任何分数产生前冻结，不用测试图。Android 评测/训练图实际 1007×755；Room 历史 factor4 为 779×519。P01 `protocol_manifest.json` 的两场景 `cameras.bin/images.bin/points3D.bin` SHA 均与服务器既有副本现场完全相符；split 与源训练 `dataset_split.json` 一致；16 张实际读取的训练图在 preflight 记录哈希并在运行时逐张重核。没有搬动 `E:\7-DataSet`、重新下载、训练、特征再生成、测试图选优或借用 Corner/Room factor2。

原 RU Mask 在终点由已有训练图 DINO fine 缓存、同一步 384→16→1 头、0.25 阈值和 7×7 腐蚀产生，head 前向每图计入预算；原残差历史只用于头训练，P04 不更新它。真实 A 的 8 张 Mask 接受比例 Android 范围 0.769–0.979，Room 范围 0.959–1.000。D 在首张图中央固定矩形的 `A接受且C删除` 像素分别注入 13,857/8,059 像素（全图 1.82%/2.00%），以洋红色替换诊断副本的目标图，原图片不修改。若将来更换视图或 ROI 即不属于本次身份。

## 3. 静态机制核查与反证边界

详见 `source_audit.md`。固定 gsplat 1.5.3 的 `DefaultStrategy._update_state` 对 `means2d.absgrad` 按宽/高、相机数换到屏幕尺度后取二维范数；`count` 只要求两轴 `radii>0`，不要求合成贡献或 Mask 接受。RU 统计从 step 0 开始，Mask 光度从 500 生效，增密 10000–19999 每 100 步、各次事件后清空；首个事件已经累积增密前统计。opacity reset 在 15000/18000 且在同一步增密/清空后，Mask 头更新暂停各后续 300 步。Gaussian 光度 backward → 独立 Mask 监督/头更新 → RU 策略回调 → Gaussian optimizer；Mask 硬化值从 photometric 图分离。L1 的均值分母是全图全通道，DSSIM 先在两幅图乘 Mask 再计算局部 SSIM，不能简单把拒绝像素当作有真实背景梯度。

源代码没有排除投影计数稀释；但 Room A 与 B 实测几乎一致，说明**不是每个普通场景都存在明显真实 Mask 选择性**，且 B 的 `u` 不必对所有投影等于 1。A 下 `A≈0 且 G0>0` 的探针对象未出现（Android/Room 0 个）；不能用 epsilon 虚构机会信号。Pixel-GS 官方项目强调按覆盖像素数加权视图梯度；ReAct-GS §3.2 式 9 使用每视图平均合成贡献。本包只将这些权重适配到 RU AbsGrad，未复现两论文的其它机制或训练结果：[Pixel-GS 项目](https://pixelgs.github.io/)、[ReAct-GS 原论文](https://arxiv.org/html/2510.19653v1)。

## 4. 终点探针集合、评分和风险

每场景 512 探针×8 视图共 4096 对。`z` 为原策略投影计数条件，`b` 为原合成定义的 `ΣTα`，`u=bM/b`（b≤阈值为 0）；`m` 为实际参与像素数。以下全是**探针范围**，非全模型 Gaussian 分布：

| 场景/条件 | 投影 `z=1` | 有贡献 `b>0` | 有接受贡献 `bM>0` | 投影无贡献 | 有贡献但无接受 |
|---|---:|---:|---:|---:|---:|
| Android A | 3503 | 2953 | 2582 | 550 | 371 |
| Android B | 3503 | 2953 | 2953 | 550 | 0 |
| Android C | 3503 | 2953 | 1638 | 550 | 1315 |
| Room A | 1696 | 1613 | 1613 | 83 | 0 |
| Room B | 1696 | 1613 | 1613 | 83 | 0 |
| Room C | 1696 | 1613 | 1020 | 83 | 593 |

Android A 中有贡献的投影之 `u` 均值 0.838、中位 1.0；Room A 为 0.99996、中位 1.0。Room 的 `n` 和 `A` 平均 3.3125/3.1503，差值主要来自投影但无贡献；Android A 为 6.8418/4.8337。B 全接受只令 `b>0` 时 `u=1`，仍保留 550/83 个无贡献投影。对“每个计数投影确有贡献且 u=1”的代数控制，`A=n`，`Gopp=G0`，OAC 返回 `G0`；不能把 B 误当作这个更强前提。

注意 `diagnostic_summary.csv` 的 `mean_u` 是全部 512×8 对（含未投影）上的均值；上段 0.838/0.99996 是 `analysis.json` 明确以 `z=1 且 b>0` 为条件的均值，两者不可混写。

共享有限候选均 512 个，top K=`ceil(10%×512)=52`，ID 固定打破平局，先不按旧 `grow_grad2d` 阈值过滤。无证据权重回退 `G0`，不同相机支持门 3、κ=4、同条件倍率帽 2，均在诊断前冻结。A→C 的自身份 top-52 保留数：

| 场景 | 原 G0 | Pixel 式 | ReAct 式 | 简单 a·u | OAC |
|---|---:|---:|---:|---:|---:|
| Android | 37 | 31 | 36 | 37 | 39 |
| Room | 35 | 37 | 37 | 36 | 37 |

OAC 比简单控制仅多保留 Android 2/52、Room 1/52；这只是固定状态删失后的排序稳定性，不是真值优劣。OAC 与原 G0 的 A 条件 top-52 重合 Android 45、Room 51；Room 未出现普遍 top 改写，但 A 条件个体最大 `OAC/G0=1.571`，表明模型可见性本身会触发机会校正，不能把该比率全归功于 Mask。C 条件 Android 的支持门通过 302/512、倍率帽触发 5；Room 支持门通过 136/512、倍率帽触发 0。`diagnostic_summary.csv`、`analysis.json` 与压缩 `per_view_gaussian.csv.gz` 保留逐视图 b/bM/m/u/z/g 和全部五分数。

C→D 后 Android/Room OAC top-52 均 **52/52** 保持；低证据 `A<1` 探针分别 51/145 个，最大 OAC 绝对增加约 `3.9e-11`/`5.83e-5`，无低证据对象新进入 OAC top-52。重要反例：Room 低证据对象的最大 `OAC(D)/OAC(C)` 约 **11.91 倍**，尽管每一条件内 `OAC≤2×G0`；这是条件 D 同时改变了 G0，说明 2× 帽不能被宣传为防误接受的跨条件保险。这个大倍率对应的绝对增加很小，本探针没有 top 进入；由于 D 只作用一图、且 512 探针中真正得到新增贡献的对象少，不足以确认全面抗错。自然拒绝区也不是有标签的瞬态真值。

## 5. 等价性、开销与预算

64 Gaussian、64×64 的独立参考在两场景与原 gsplat alpha 最大绝对差均 `1.19e-7`；全景首图 Android/Room 最大 `0.001025`/`4.41e-6`。Android 760,285 像素中仅 1 个超过 `2e-4`，其余 99% 差不超过 `5.96e-8`，已作为阈值附近的数值边界保留，不宣称逐位精确。诊断开/关时原渲染及 loss 差均 0，AbsGrad 最大差 Android `1.09e-11`、Room `3.64e-12`。前后 Gaussian tensor SHA 相同；模型、头、残差文件 SHA 也与 preflight 一致（`reference_validation.json`、`state_integrity.json`）。无 optimizer step、clone/split/prune/reset。

成对同步计时固定两场景首张图，1 次 warmup+3 次测量，全部调用扣账。非 warmup 中位原 render Android/Room 2.082/2.416 ms；render+独立探针 3.618/3.965 ms，增量约 1.536/1.549 ms、比值 1.74/1.64。此处只测额外探针前向，不含 Mask 头、原 RU coarse/DINO、backward 或训练调度，不能据此宣称生产训练速度。已有 RU 日志 `DINO_time.json`：Android 30k 渲染特征 212.64 秒/训练 633.99 秒，Room 206.03/576.00 秒；这属**原 RU**成本，不加到探针增量。

探针保留 `int32[N]` 选择映射（全模型每 Gaussian 4 B）和 512×5 个 float32 累加器（每探针 20 B），另有 O(像素) 的 Mask/alpha 临时量；无 N×像素张量，静态结构满足常驻统计≤64 B/Gaussian 目标。`overhead.csv` 原始 CUDA peak 差因先后图保留/allocator 缓存甚至为负，不可解释成节省显存；可靠的**增量峰值**仍为 UNKNOWN。独立 Triton 探针不是原渲染器中的融合生产插桩，其 1.5 ms 不可外推到 P05。预计子代 tile 代理可由 `info['tiles_per_gauss']` 的 tile 交数定义为无量纲/整数工作量，但本包未产生或排序子代，成本代理的预测准确性 UNKNOWN。

## 6. 成本更正、失败与下一决策

P03 成本边界更正见 `p03_cost_accounting_addendum.md` 和 CSV。706/1129/2779 秒不能相除当统一一次成功构建；全尝试总量仍 4976 秒，不重训、不重测 FPS。P04 本包的两次计费技术失败为误选无 GLM 的源码路径与 Triton `break` 语法；另外两次过严断言使核验重启，全部调用仍计 231。`call_budget_ledger.csv` 逐次预扣，`run_ledger.csv` 记录重试。没有静默清零或据分数换图/调 κ、帽、阈值、ROI。

执行终点不等于 OAC 假设成立。本包能够确立真实源码的投影计数与 Android 终点选择性，但**不能**证明 OAC 比简单 a·u 更适合增密、不能说明训练期和最终 PSNR。建议方案助手先审查 `NEXT_DECISION.md`：在没有合法增密期配套 snapshot 和更具辨别力的风险证据时，不冻结 P05 的生产公式，也不以这个终点探针自动开启四场景训练。P04 交付后停止。
