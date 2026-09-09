# V3 单表面诊断实现审计

日期：2026-09-09。协议：`V3_SINGLE_SURFACE_DIAGNOSTIC_V1`。本地分支核对为 `ru-part`，没有切换分支、提交、推送或覆盖原有工作区修改。本轮依据用户新传的 `RU_PART_V3_SINGLE_SURFACE_DIAGNOSTIC_CODEX_INSTRUCTION.md` 实施。

当前结论：本地实现、自检和原图候选标注已完成；静态标签尚未确认，服务器真实 DeltaQ/CUDA 预检/三组诊断均未执行。三个正式组实际更新数都是 0。不能据本地测试或候选图填写科学诊断 PASS。

## 实际修改与复用

| 文件 | 本轮职责 |
| --- | --- |
| `puri_gs/single_surface.py` | 固定 A/B/H 角色、到 A 的几何候选排序、候选形状检查、真实数值干预资格、空 B ROI 指标、六域保护和固定结果分类 |
| `tools/ru_part_v3_single_surface.py` | 新协议后台入口；旧数组复用、标注、确认、两次更新预检、预注册、调用原循环和报告/回读包 |
| `configs/ru_part_v3_single_surface.json` | 新协议 ID、源身份、旧准备路径、精确终态 LR、400 更新及 0/400 评价、GPU 0 |
| `puri_gs/recoverability_runtime.py` | 增加指定训练视图读取模式；该模式跳过 head/FeatureCache/StaticSupport，只使用旧数组；旧入口默认行为保留 |
| `tools/ru_part_v3_recoverability.py` | 将既有后台包装及 400 步循环提取为共享函数；临时预检可保留一次实际 backward 的 RGB 梯度；失败调用计数和组起止时间 |
| `tests/test_single_surface.py` | 数值规则、实际共享循环的 CPU 替身、准备/确认/修订/失败状态与包内状态检查 |

`diagnostic_loss`、`masked_photo_loss`、`static_rescue_l1` 和原 trainer 的渲染/Adam AST 提取逻辑没有重写。原四图配置和历史判定保留；旧入口仍在 0/100/200/400 评价四图，新入口只在 0/400 评价三图。

## 输入身份与有界准备

- 旧准备文件：`logs-puri/ru_part_v3_coverage_recoverability_diag/garden_v3_final_diag_v1/preparation.json`，从已有包核对 SHA-256 为 `437076a392baec665626d558c0668383ce8c763183d87da550b1837adb4fee20`。
- 固定源 Gaussian SHA-256：`14563fbbcfbb722879929be31fd19ad5e6a02939e8d2409e6a6855ce909d417c`；step29999、SH3、2,266,597 个 Gaussian。服务器实际文件再次验证由 worker 执行。
- 原 renderer/Adam 提取代码摘要记录为 `ec3a0cb8956450fa620e0a9a90dad92a407de09fd1b62aadac4a3e91e3c1d168`，新 worker 要求实际值相等；同时核对源 trainer 身份与旧配置记录。
- 复用旧记录声明且摘要相符的 A/B `M.npy`、`C.npy` 和 A `source_rgb.npy`。本地旧 review 包没有这些二进制数组；这不表示服务器没有。缺少实际文件时报告确切路径，不用着色 PNG 反推。
- 新准备不载入 Gaussian、不渲染、不计算 head 或 C、不重放 161 图。读取相机/划分/归一化/SfM；校验 A/B/H 的训练 RGB；原 Parser 保留一个训练图的尺寸校准读取，记录在 opened_training_images 中；拒绝测试图读取。
- 只读取现有 gsplat 包版本，不再次遍历整包计算每个二进制摘要。核对旧准备的 Torch/CUDA/包版本、精确终态 LR，并记录本轮实际 GPU UUID；不声称重做完整归档审计。
- 旧覆盖记录仍为 C 非零 1.608740%、M 拒绝 1.206149%、Q 非零 0.011493%、11/161 图 Q 非零、三个历史窗口激活 518/1012/735 次。本轮没有重算这些数值。

## 候选排序与原图核对

本地材料目录：`E:\7-DataSet\ru_part_v3\single_surface_diag\house_upper_v1`。新指令原文也归档在此。排序先于新候选视觉检查，依据旧准备提供的相机元数据，仅按到 A 的欧氏距离和 train view ID。

| 顺序 | 图像 | train view ID | 到 A 的距离 |
| --- | --- | ---: | ---: |
| 1 | DSC07986.JPG | 26 | 0.17766094245771946 |
| 2 | DSC08022.JPG | 57 | 0.1985989541988751 |
| 3 | DSC08021.JPG | 56 | 0.25856082639953987 |
| 4 | DSC08023.JPG | 58 | 0.30687798810456346 |
| 5 | DSC08053.JPG | 84 | 0.3175638711897199 |
| 6 | DSC07985.JPG | 25 | 0.35398144207843374 |
| 7 | DSC08082.JPG | 110 | 0.38493679622144994 |
| 8 | DSC08051.JPG | 83 | 0.4156263833205857 |

实际只核对第 1 张。H=`DSC07986.JPG` 清楚显示 A 中同一个三角山墙，顶点、两侧屋檐和砖纹提供对应依据；没有因 H 已重建良好而跳过它，也没有继续搜索后续候选。

两个原图均为 1297×840。A 的独立多边形为 `(692,78),(614,117),(773,112)`，面积 2,946；H 为 `(814,37),(727,79),(902,72)`，面积 3,413；B 数组全零。这些是未确认提案。已逐图检查边界放大图，区域在同一砖面内部，避开天空、屋檐边条和树叶。

初版本地几何记录与生产入口在一个距离上出现小于 1e-15 的浮点末位差异。已核对八个名字、ID、顺序完全一致，并统一命令载荷使用生产入口的 `math.sqrt` 序列化。初版锁定文件保留；说明在 `candidate_order_verification.json`。生产候选列表规范摘要为 `9442a7c2bd8bf604b1b285cd544c17e8137f476fd8218922109153954495af96`。修正不改变候选选择或查看顺序。

服务器准备阶段只读计算 `DeltaQ=(1-M)(1-C)S`、非零数量/比例/权重和/两类均值以及 `D_A`。有限且权重和与 D_A 均大于零才进入标签回读；没有额外覆盖率门。数值与原图要一起呈现后再确认，房屋候选文字不作为静态标签授权。

## 优化、评价与有效性

Va/Vb/O 共享固定 `[A,B]×200`，每组从源 Gaussian 独立载入并新建原类型零状态 Adam，seed42，六组终态常量 LR 与旧准备/实际配置严格相等。恢复项保持全图全 RGB mean，O 替换原权重为 `C+(1-C)S`；B S=0 保留原 V3 恢复项，H 从训练字典中排除。

确认后只用 Va/O 各一个独立临时副本执行一更新。复用这两次实际 backward 中的 RGB 梯度，比较差值与 `0.8*DeltaQ*sign(rgb-target)/rgb.numel()`；核对有限非零参数梯度、参数确实改变、Gaussian 数不变、源文件不变，再丢弃两副本。没有额外反传或 VJP。失败时真实临时更新/渲染/backward 计数保留。

在正式第 1 次更新前生成不可覆盖的 `single_surface_preregistration.json` 及 SHA，锁定源、代码、角色、候选、用户确认、面板及数值区域、学习率/优化器、顺序、GPU/环境和规则。三个组串行，各 400 更新；每组只有 0/400 的 A/B/H 评价，总计 1,200 正式 rasterization/backward 与 18 评价，临时两次单列。

定量指标使用未 clamp 的 float RGB 与原始目标/255。A 主指标只看 S，不平均 B；B 的目标 ROI/alpha 为 null；H R 仅评价。保护域为 A 的 S 外、A 的全部 M 接受区、B 全图、B 全部 M 接受区、H 的 R 内、H 的 R 外。必要域为空或指标非法即无效，不使用区域回退。

初态均值与初态极差由三组各自 0 次评价得到；n 为控制终态差与初态极差的较大者。A/H 增益分别取 5%/2% 初态、3n、1e-4 的最大阈值，且同时优于初态及较好控制。保护的初态预算 d 固定为 max(1% E0,1e-4)，只有控制预算 b=max(d,3n) 可随波动放宽。H 提升是额外证据，不是 A 局部受控响应的必要条件。

失败停止后续组，保留原目录与准确完成数，不自动续跑或换参。终态六组 Gaussian 文件通过重新载入验证，诊断元数据 step399/source_step29999/diagnostic_updates400/diagnostic_only=true，不保存 Adam。结果包在最终状态落盘后制作，包含 `status.json`；打包失败另记 `READBACK_BUNDLE_FAILED`，不诱发优化重跑。

## 已执行验证与边界

本地 WSL Ubuntu-22.04 的现有 `.venv-gsplat153` 使用 Torch2.4.0+cu121 执行 CPU 检查。仅为该测试进程添加既有系统 PyYAML 搜索路径，没有安装、升级或改变服务器依赖。

- `tests/test_single_surface.py`、`tests/test_coverage_recoverability.py`、`tests/test_ru_part_v3.py` 最终完整检查合计 **91 passed**（19.38 秒），包括 26 项新增测试和 65 项旧测试。
- 实际三组共享循环的 CPU 可微替身验证五次独立源载入（临时两次 + 正式三次）、1,200 正式更新、18 评价、B 空 ROI、H 无训练输入、失败第 9 次完成数及含失败状态的 zip。
- 准备流程测试验证旧数组复制、零 Gaussian/Mask/特征调用、旧文件不变；另测面板身份、一次确认、锁定后拒改、边界修订保留旧包、最终状态入包。
- 新入口 `--help`、固定配置验证、修改 Python 文件语法检查通过。生产多边形载荷与旧准备元数据实测匹配；本地两个实际边界图已经视觉检查。
- 中文清单中 9 个 Bash 代码块及 `01_prepare_linux.sh` 经 `bash -n` 检查，清单准备载荷与实际生成文件逐字对应。
- 未连接服务器、未实际验证当前 GPU 空闲/UUID/文件状态，未计算服务器真实 DeltaQ，未运行真实 CUDA 更新或 Va/Vb/O。本地可微替身不替代 CUDA renderer/Adam 检查。

历史记录维持 `QUALITY_RECOVERY_FAIL / NO_GO`、`ROI_NOT_READY`、`REPLAY_NOT_EQUIVALENT`，`HISTORICAL_RECORDS_RECLASSIFIED=false`。本轮完成或触及明确停点即停止。完整用户命令与当前停点见同目录 `SINGLE_SURFACE_RUNBOOK_CN.md`。
