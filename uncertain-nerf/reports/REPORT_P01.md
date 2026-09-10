# 工作包 1 报告：冻结资产、数据和共同协议

日期：2026-09-10  
状态：`COMPLETE_VALID`（工作包 1 已到终点；存在已明确登记的服务器侧未知项）

## 1. 结论

工作包 1 已完整执行，未训练任何新模型，未启动 OAC、Garden/PART 修补或工作包 2。

- 当前分支保持 `ru-part`，审计起点代码为 `4b5489990cd515fb4ef3458673f51c03baf46e26`，未切分支、未回滚历史改动。
- Android、Patio-High、Room 的 B1/RU 身份已冻结；Android 的 10k B1 和原 Phase-R 配对已作为历史行单独保留，禁止与当前 30k 配对混用。
- Garden 只登记冻结状态，不进入 P02。
- Android 与 Patio-High 已各生成一套供 RobustSplat、SLS-mlp 共用的 PINHOLE COLMAP 输入。两次转换均只做确定性的相机/图像格式转换，没有重跑 SfM；相机、位姿、稀疏点、投影矩阵及图像内容均已回读校验。
- Room 历史 factor4 结果与未来正式主表 factor2 协议已严格拆开。现有 Room 模型只能作为 factor4 历史结果，不能填入 factor2 主表。
- 四项外部比较均判定为“须训练”：RobustSplat、SLS-mlp 分别在 Android、Patio-High 各一个 seed42 身份。P01 没有安装外部方法、下载 checkpoint/特征或启动训练。

最小结果包位于本目录：`asset_ledger.csv`、`protocol_manifest.json`、`run_plan_P02.csv`、`p01/run_ledger.csv`、`p01/reevaluation_summary.csv`、`p01/reevaluation_per_image.csv`、`p01/status.json` 和 `p01/NEXT_DECISION.md`。

## 2. 代码、环境与入口审计

| 项目 | 冻结结果 |
|---|---|
| Git 分支 | `ru-part` |
| 审计起点 HEAD | `4b5489990cd515fb4ef3458673f51c03baf46e26` |
| 仓库说明 | 未发现额外的 `AGENTS.md` |
| 主训练启动器 | `run_puri_gs.py`，向固定 gsplat 1.5.3 `examples/simple_trainer.py` 应用项目补丁后运行 |
| 内部方法补丁 | `patches/gsplat_v1.5.3_puri_gs_ru.patch` 等已有补丁；本包未改算法 |
| 冻结服务器环境 | NVIDIA L20；Python 3.10.20；PyTorch 2.4.0+cu121；CUDA 12.1；gsplat 1.5.3，commit `937e29912570c372bed6747a5c9bf85fed877bae`；torchmetrics 1.4.3 |
| 本地 GPU | RTX 5070 Ti（sm_120），与项目冻结的 PyTorch 2.4/cu121 环境不兼容，因此没有把本地 GPU 结果冒充服务器复评 |
| GPU 顺序 | 后续固定为 `6 → 7 → 0 → 1 → 2 → 3 → 4 → 5`；P01 未占用 GPU |
| 服务器实查 | 已连接到 `172.16.55.2`，但 SSH 认证不可用；没有声称已刷新服务器文件或执行 GPU 推理 |

审计中修复了一项确定的接口问题：Patio 解析器现在接收训练器已经传入的 `calibration_index`，并校验其范围。另增加两个小型、可复跑的共同输入导出器及测试，没有把罕见兼容分支塞入训练前端。

## 3. 冻结证据与模型身份

本地冻结归档 `PURI-GS-RU/server-artifacts/puri_gs_ru_artifacts_749d584b.tar` 为 3,355,443,200 bytes，重新计算 SHA-256 为 `ac7376ef0dc89df871635ec2ffe1867a7e721b5979cd84c44029cfe793ed51e3`。Room 可视化归档 SHA-256 为 `360ef84c11c62fd55f5ea4f7179274ee955bfb46b768e7ef6f4896a7aaeac465`。完整逐模型路径、checkpoint 哈希、大小、训练 commit 和评测证据见 `asset_ledger.csv`。

冻结锚点如下：

| 场景 | 方法 | 训练 commit | checkpoint SHA-256 | 高斯数 | 训练秒 | 状态 |
|---|---:|---|---|---:|---:|---|
| Android | B1 30k | `9e292309…` | `9ea31297…b7c` | 1,218,407 | 614.480 | 当前配对锚点 |
| Android | RU 30k | `9e292309…` | `11737109…f73a` | 551,297 | 633.989 | 当前配对锚点 |
| Patio-High | B1 30k | `749d584b…` | `31f618c7…3503` | 3,235,690 | 1109.416 | 冻结锚点 |
| Patio-High | RU 30k | `749d584b…` | `b3c9fca6…e1ae` | 435,779 | 551.629 | 冻结锚点 |
| Room factor4 | B1 30k | `da5089cd…` | 未能从本地证据恢复 | 1,301,078 | 未知 | 仅冻结评测证据 |
| Room factor4 | RU 30k | `da5089cd…` | 未能从本地证据恢复 | 708,337 | 未知 | 仅冻结评测证据 |
| Garden | B1/RU 30k | `9e292309…` | 已在台账完整登记 | 3,798,503 / 2,274,197 | 1631.620 / 1347.455 | 仅登记，不进 P02 |

Android 历史身份必须分开解释：

1. `android-b1-10k-historical` 是 commit `32a7d787…`、step 9999、reported PSNR 24.312067 的旧 B1；checkpoint 不在本地，不能替代 30k B1。
2. `android-phase-r-original-pair` 只保留 reported ΔPSNR `+1.206509`；本地缺少完整 checkpoint 身份，不能与重跑 checkpoint 或严格计时混合。
3. P02 唯一建议内部锚点是同 commit、同 30k 预算的 `9e292309…` B1/RU 配对。

## 4. 数据与共同协议

所有本地数据均留在 `E:\7-DataSet`，没有移动或重复下载原始数据。完整文件名、逐图内容哈希、相机参数、稀疏模型哈希、split、归一化及评测定义见 `protocol_manifest.json`。

| 协议 | 图像/划分 | 实际评测分辨率 | 用途与状态 |
|---|---|---:|---|
| Android 原始 factor4 | 263 注册；122 clutter 训练、19 extra 测试、122 clean 排除 | 1007×755 | 内部 B1/RU 历史训练协议 |
| Android common factor4/loader1 | 同一 263/122/19/122 | 1007×755 | 外部方法共同 PINHOLE 输入，已验证 |
| Patio-High 内部 factor4 | 267 帧；221 训练、45 测试、1 排除 | 1007×755 | 内部 B1/RU 协议 |
| Patio-High common factor4/loader1 | 同一 267/221/45/1 | 1007×755 | 外部方法共同 PINHOLE 输入，已验证 |
| Room factor4 | 311；272 训练、39 测试 | 779×519 | 仅历史保护 |
| Room factor2 | 311；同一 every-8 划分 | 1557×1038 | 正式主表数据已就绪，但需要新模型 |
| Garden factor4 | 185；冻结 every-8 划分 | 1297×840 | 只登记 |

共同输入身份：

- Android：`E:\7-DataSet\PURI-GS-derived\robustnerf\android_colmap_common_factor4_v1`；112,790 个初始点；协议 SHA-256 `19196d04de27236cd1566389a2f315d82b1a00be9f0ddc83c401a6fbfae44bec`；验证文件 SHA-256 `5f9bd7b0dfb8571360c19508fcc3fc070eb01b8557d6752c903b2e8959183c15`。
- Patio-High：`E:\7-DataSet\PURI-GS-derived\ontogo\patio_high_colmap_common_v1`；36,922 个初始点；协议 SHA-256 `03eb7e59d9f30501eb5c8b220b4dbe48418869c37bdbf275d9bcfd5857217ebf`；验证文件 SHA-256 `69877908fa71f56ec1011dca28c7b73407e15a6a7bb3730beaa14d7ff556b377`。

两套 common 输入都满足 `conversion_only=true`、`sfm_reestimated=false`。验证读取的是写盘后的 COLMAP 文件和图片，不是只比较内存变量；结果包含全部图片哈希/尺寸、相机内参、位姿、点云和投影矩阵检查。

## 5. 已有模型复评证据

没有因评测器或协议发生变化而盲目重训。P01 汇总了冻结的同一独立评测器结果，共 8 个场景—方法汇总行和 254 个逐图指标行。评测口径为完整冻结 test 列表、渲染值 clamp 到 `[0,1]`、逐图后算术平均；严格速度为 10 次 warmup、关闭存图、完整测试集重复 3 次后取中位 FPS。

| 场景/协议 | 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 高斯数 | 严格中位 FPS | 训练秒 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Android factor4 | B1 | 23.3873 | 0.79437 | 0.17030 | 1,218,407 | 255.97 | 614.48 |
| Android factor4 | RU | 24.6472 | 0.82119 | 0.16470 | 551,297 | 479.11 | 633.99 |
| Patio-High factor4 | B1 | 14.2167 | 0.42500 | 0.62149 | 3,235,690 | 111.05 | 1109.42 |
| Patio-High factor4 | RU | 19.3408 | 0.65173 | 0.30124 | 435,779 | 498.05 | 551.63 |
| Room factor4 历史 | B1 | 31.3070 | 0.93333 | 0.07672 | 1,301,078 | 263.94 | 未知 |
| Room factor4 历史 | RU | 32.0550 | 0.94642 | 0.07006 | 708,337 | 424.24 | 未知 |
| Garden factor4 | B1 | 27.7167 | 0.87396 | 0.06815 | 3,798,503 | 未存档 | 1631.62 |
| Garden factor4 | RU | 26.6506 | 0.85731 | 0.09051 | 2,274,197 | 未存档 | 1347.45 |

Android、Patio-High、Room 的当前配对里 RU 均提高 PSNR 且减少高斯数；Garden 的 RU 更轻、更快训练但质量下降，因此只登记，不把它叙述成一致优势。单 seed 结果只用于定位，不作稳定性结论。

Android 冻结配对的严格 PSNR 是 23.3873/24.6472。历史配对审计里出现的 B1 23.479 属于另一条评测记录，已禁止拼接。Room 缺少 checkpoint 哈希、字节数和绝对训练秒；归档只支持 RU/B1 训练时长比 1.025487，这些空值保持为空，没有填 0。

## 6. 外部方法兼容性与四项运行表

### RobustSplat

- 冻结源码 commit：`a130281d6d0c004032a9a57e8d6a14962d9836d3`。
- 官方已发布 checkpoint，但其 README 明确说明 RobustNeRF/On-the-go 数据使用作者重跑的 SfM；官方数据/模型因此只能列为作者协议参考，不能当成本项目 common-input 的复评结果。
- 官方读取器支持 PINHOLE/SIMPLE_PINHOLE，并按 clutter/extra 关键词划分；本包的两套 common 输入据此消除了 OPENCV/On-the-go 自定义格式差异。
- 共同输入下没有合格 checkpoint，四项中的两个 RobustSplat 身份均为“须训练”。

来源：[RobustSplat 官方仓库](https://github.com/fcyycf/RobustSplat)、[固定版本](https://github.com/fcyycf/RobustSplat/commit/a130281d6d0c004032a9a57e8d6a14962d9836d3)、[官方 checkpoint](https://huggingface.co/fcy99/RobustSplat-checkpoints)、[官方数据](https://huggingface.co/datasets/fcy99/RobustSplat-data)。

### SpotLessSplats SLS-mlp

- 冻结源码 commit：`0caae3cc45bb1fddf86bd47e4a521888f5c49889`。
- 变体锁定为 SLS-mlp、不启用 UBP：`--loss_type robust --semantics --no-cluster`，不得传 `--ubp`。
- 官方公开复现数字、准备数据和特征不是本项目 common 输入；未发现可直接复评的同协议 checkpoint。
- P02 需为确切 common 图片准备 Stable Diffusion features，并将准备时间入账；Android、Patio-High 两个身份均为“须训练”。

来源：[SpotLessSplats 官方仓库](https://github.com/lilygoli/SpotLessSplats)、[On-the-go 数据集官方仓库](https://github.com/cvg/nerf-on-the-go)。

`run_plan_P02.csv` 仅包含以下四个完整训练身份，不隐藏额外内部模型：

| run_id | 场景 | 方法/变体 | common 协议 | seed | 状态 |
|---|---|---|---|---:|---|
| `P02-android-robustsplat` | Android | RobustSplat | `android-colmap-common-factor4-v1` | 42 | 须训练 |
| `P02-android-sls-mlp` | Android | SLS-mlp-no-UBP | `android-colmap-common-factor4-v1` | 42 | 须训练 |
| `P02-patio_high-robustsplat` | Patio-High | RobustSplat | `patio-high-colmap-common-factor4-v1` | 42 | 须训练 |
| `P02-patio_high-sls-mlp` | Patio-High | SLS-mlp-no-UBP | `patio-high-colmap-common-factor4-v1` | 42 | 须训练 |

每个身份最多首次完整运行加一次“已定位技术问题”后的重启；方法原生世界归一化可以保留并记录，但相机、像素、点云和 split 不得改变。作者 checkpoint 只允许做作者协议参考或最小兼容 smoke，不能替代共同输入训练。

## 7. 代码与数据验证

- 转换/解析专项测试：`8 passed`。
- 仓库其余测试在冻结 `.venv-gsplat153` 中得到 `387 passed, 2 skipped`，另有 2 个失败：一个继续来自既有环境缺少 PyYAML；另一个测试把合法分支硬编码为 `dev`，因本任务按要求保留 `ru-part` 而返回分支门禁失败。完整收集时还有 3 个模块同样因 PyYAML 缺失而中止。本包没有为此安装依赖，也没有修改与 P01 无关的 Garden/PART/OAC 代码。
- 交付重建器：`P01_PROTOCOLS=7`、`P01_ASSET_ROWS=10`、`P01_SUMMARY_ROWS=8`、`P01_PER_IMAGE_ROWS=254`、`P01_P02_RUN_ROWS=4`、`P01_DELIVERABLES=PASS`。
- Android 实际导出：`EXPORTED_ANDROID_IMAGES=263/263`、`ANDROID_COMMON_COLMAP=PASS`。
- Patio-High 实际导出：协议与验证文件均存在，`PATIO_HIGH_COMMON_COLMAP_VALID`。
- 本包结束时没有训练进程或其他后台进程需要继续观察。

## 8. 缺口与边界

1. 服务器 SSH 认证不可用，因此 Room checkpoint 文件哈希/大小、服务器现存路径和绝对训练秒没有现场刷新。相关行明确为 `FROZEN_EVAL_ONLY_CHECKPOINT_SERVER_UNVERIFIED`。
2. 本地 RTX 5070 Ti 不能安全运行冻结的 PyTorch 2.4/cu121 栈，所以没有新增 GPU 小推理；已有模型兼容性来自归档内 checkpoint 标准键、已有独立评测及逐图证据。
3. 外部源码、环境、checkpoint 和 SLS 特征按 P01 边界未安装/下载。P02 获得明确授权后才可落地服务器路径并开始四项运行。
4. Room factor2 需要全新的模型身份；P01 不把它悄悄加入四项外部预算。

上述缺口不会破坏已完成的本地冻结身份和共同协议，但必须在 P02 启动前由方案助手确认接受。

## 9. 交回方案助手的三个决定

1. 是否确认 Android 以 `9e292309…` 的 30k B1/RU 配对作为 P02 唯一内部锚点，并只把 10k B1 与原 Phase-R Android 行保留为历史参考？
2. 是否确认 Android 与 Patio-High 两套经回读验证的物理 factor4/loader factor1 PINHOLE COLMAP 为唯一共同输入，作者重跑 SfM 的数字只列为 Reported？
3. 是否授权下一包按 `run_plan_P02.csv` 启动四个单 seed42 身份：RobustSplat 与 SLS-mlp-no-UBP 各跑 Android、Patio-High？

在这三个问题得到新任务包定义前，实施端停止在工作包 1，不自动启动下一包。
