# V3 coverage / local recoverability implementation audit

记录日期：2026-09-09。依据用户本次更正后的覆盖率与局部可恢复性诊断指令实施。运行方式为本地实现、Git GUI 同步、服务器现有环境执行。

## 已实现的边界

- 新入口为 `tools/ru_part_v3_recoverability.py`，独立输出目录，原 V3 目录仅作为输入。未修改原训练器、V3 loss、C 构造和历史筛选判定。
- 原 run 的 checkpoint SHA、step29999、2266597 个 Gaussian、标准六组 float32 参数和 SH3 均核验。图像划分、相机、内参、归一化、训练图文件、SfM 和 C 缓存与源 manifest 对照。
- 只实例化 Parser/训练 Dataset 及独立参数容器，不构建 Runner、DINO 模型、重放控制器或任何训练拓扑循环。test Dataset 只用于核对划分元数据，不调用其 `__getitem__`；Parser 校准读训练图，并拒绝 test 图像解码。
- 从原 trainer AST 提取 renderer 方法和 optimizer 构造尾部，直接使用原逻辑。gsplat 算子仍从现有安装包导入，不把源码仓库根目录加入 Python 路径，不替换现有 wheel 或 CUDA。
- 原 fine 特征经同 run 的最终 head、bilinear 插值、原阈值和腐蚀核得到 M。Head 用 eval/no-grad 冻结；该 head 无训练/评估模式不同的 BN 或 Dropout。不需要 histogram 来推理 M。缺失 head 可接受有源 checkpoint/图像/数组 SHA 和原分辨率证明的完整数值 Mask；PNG 和视觉反推不被入口接受。
- 161 张 C 的统计与可用 M 的统计分开；两个优化图缺任一 M 即停止短优化。每图 M 来源写入 CSV。历史特征 manifest 未保存单文件内容 SHA，本次记录当前实际使用的特征文件 SHA，并在正式诊断前复核；不宣称恢复了历史未记录的哈希。
- 历史进度为稀疏采样。窗口边界可由累计激活数与紧邻行 Q 指示精确恢复时才给出全窗口激活计数；新增 loss 和 mean Q 只列采样均值，无完整逐步记录则不输出全窗口 loss 总和。

## 静态标签与预注册

- 两张优化图固定为 DSC07987.JPG、DSC07989.JPG，各 200 次，固定交替序列。另两张检查图仅从训练相机几何排序及原图同表面核对中选择。检查图参加过原始 V3 训练，不能称为 unseen views。
- Codex 后续按原图逐视图绘制静态多边形，导出 bool S 和四视图边界图。用户确认后才记录静态标签确认；当前没有预填确认、测试视图 ROI 或残差推断静态标签。
- 未确认草案有独立版本目录。确认后禁止修订；预注册采用排他创建及 SHA，不允许覆盖。
- 冻结来源、导出 M/C/初态 RGB、S、原图预览、边界图、ROI 与确认、代码、学习率、Adam 配置、GPU UUID、环境、相机序列、保护区规则和阈值。正式启动前及运行后检查相应 SHA。
- O 的额外支持严格为 `(1-M)(1-C)S`。在优化图集合上新增 Q 权重及残差加权量须均非零；否则记录 `NO_ORACLE_INTERVENTION`，不启动组训练。

## 优化与评价语义

- 独立临时副本先做 1 次 CUDA forward/backward/Adam 更新，验证有限、非零梯度、实际参数变化、N 不变、源文件未改，之后丢弃。此为运行后的真实检查；CPU 测试不代替它。
- Va/Vb/O 每组重新读源参数并生成独立零状态 Adam。源 optimizer factory 的 eps、betas、各参数组、foreach/fused 等实际默认设置均写入预注册，JSON 往返后逐项比较。
- means LR 用源 cfg 的场景缩放后初始 LR，精确递乘 29999 次原 ExponentialLR gamma；其余 LR 按原 cfg。原 scheduler 在 optimizer 之后更新，不能把 step29999 更新后的第 30000 次衰减误用为诊断初值。诊断 400 步保持 LR 常量，不调用 scheduler。
- Va/Vb 复用 `masked_photo_loss` 和 `0.8 * static_rescue_l1`。O 只将 C 换成 `C+(1-C)S`。M/C/S 均 detach，新增项每步启用；训练 loss 保留整图 RGB 全局平均，没有 ROI 面积归一化、alpha gate 或残差 gate。
- 每组 400 次训练 renderer 调用、400 次 Gaussian backward、16 次四图评价；预检、准备和评价分别计数。没有 coarse pass、Mask 更新、匹配、birth、opacity reset 或 topology 事件。
- 每组保存源摘要、独立 Adam 设置、精确已执行更新计数、每 50 步日志、0/100/200/400 指标、0/400 RGB/alpha/ROI 边界图和标准六参数文件。参数文件标为 step399、源 step29999、400 次 diagnostic updates。
- ROI MAE 采用未截断 render RGB 和原 target/255；图像导出允许显示截断，不改变指标。优化图保护原 M=1 且 S 外区域，若为空才回退至 S 外；检查图保护 S 外，另单独保护其 S 内。
- E0 取三组各自初态均值，初态极差参与 n。T/H 为各自两张图的 ROI MAE 等权均值。工程波动 `n=max(|Va400−Vb400|, 初态极差)`，三倍 n 不当作统计标准差。
- 局部与检查 gain 均同时对比初态和更好控制；六个保护区域逐个通过。过程无效优先，保护失败其次，之后才按 gain 分类。alpha 只作辅助。
- 中途异常保留实际已完成更新数，不启动后续组、不重试。历史 NO_GO 和 REPLAY_NOT_EQUIVALENT 均保留；诊断结束后不追加实验。

## 已执行验证

本地 WSL Python 3.10.12、Torch 2.4.0+cu121，以 CPU 运行：

```text
tests/test_coverage_recoverability.py + tests/test_ru_part_v3.py
65 passed in 16.53s
```

验证内容包括 Q 增量及真实 RGB 梯度、全图归一化、M/C/S 无梯度、缺失 M 与零分母、稀疏边界计数、未截断指标、六个保护区与判定优先级、初态/控制组波动、ROI 合法性、真实 ExponentialLR 更新顺序、实际 trainer 的 renderer/Adam 代码段、标准 checkpoint 检查、test 图名访问拒绝、GPU 0 锁定、三组串行模拟流程、失败中止、预注册改动拒绝。

串行模拟使用 2 个 Gaussian 的 CPU renderer 和独立 Adam，确实执行预检 1 次及三组各 400 次更新，并核对 1200 次训练 raster/backward、48 次评价、四个相同源参数摘要、源文件不变和 step399 诊断参数文件。这是软件流程测试，**不是实际 Garden/L20 实验数据**。

Windows 命令行帮助和四个新增 Python 文件 AST 检查通过。本地未安装或升级依赖；仅在 WSL 测试进程中加入已存在的系统 PyYAML 路径。

## 待服务器完成的事实

真实准备、四视图静态 ROI 确认、预注册、CUDA 预检以及 Va/Vb/O 都尚未运行。因此当前 actual updates 为 Va=0、Vb=0、O=0，不能标记为可恢复、不可恢复或诊断通过。

下一步按 `COVERAGE_RECOVERABILITY_RUNBOOK_CN.md` 执行 GPU 0 的 prepare，并回传生成的 review_bundle.zip。获得原图、Mask 对齐和候选视图后才能绘制并确认 ROI。
