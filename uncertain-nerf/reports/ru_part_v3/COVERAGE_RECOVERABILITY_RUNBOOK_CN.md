# V3 覆盖率与局部可恢复性诊断：操作步骤

本次落实用户更正后的《RU_PART_V3_COVERAGE_RECOVERABILITY_CODEX_INSTRUCTION.md》。原 V3 筛选已经结束，结论仍为 `QUALITY_RECOVERY_FAIL / NO_GO`；旧重放仍为 `REPLAY_NOT_EQUIVALENT`。本次使用现有 V3 终态做覆盖审计及有条件的 Va/Vb/O 局部诊断。

当前已完成本地实现、65 项 CPU 检查，并收到服务器准备压缩包：161 张训练图的终态 M 全部可用，来源与图像哈希核对通过。原图与 Mask 复核暂为 **ROI_NOT_READY**，原因是尚未找到四视图共同可见且能产生新增监督的同一静态表面。详见 [ROI 复核记录](E:/6-Project/1-UncertainNerf/uncertain-nerf/reports/ru_part_v3/COVERAGE_RECOVERABILITY_ROI_REVIEW.md)。以下步骤 1–4 已完成对应的代码同步与产物回传，本次不需要重复 prepare；先停在步骤 5，尚未启动 CUDA 更新预检或 Va/Vb/O。

## 1. 用 Git 图形界面同步本次代码

保持 `ru-part` 分支，在本地 Git GUI 提交并推送以下新增文件，然后在服务器 Git GUI 拉取：

- `configs/ru_part_v3_coverage_recoverability.json`
- `puri_gs/coverage_recoverability.py`
- `puri_gs/recoverability_runtime.py`
- `tools/ru_part_v3_recoverability.py`
- `tests/test_coverage_recoverability.py`
- 本操作说明及 `COVERAGE_RECOVERABILITY_IMPLEMENTATION_AUDIT.md`

工作区还包含此前更新的 V3 结案文档，应保留。准备阶段会记录当前 commit 和实际代码文件 SHA；准备之后至诊断结束不要切换代码或修改配置。无需升级 PyTorch/CUDA 或安装新的 gsplat。

## 2. 核对固定配置与 CPU 检查

在服务器执行：

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
.venv-gsplat153/bin/python tools/ru_part_v3_recoverability.py check-config
.venv-gsplat153/bin/python -m pytest tests/test_coverage_recoverability.py tests/test_ru_part_v3.py -q
```

配置应为源 step29999、Gaussian 数 2266597、Va/Vb/O 各 400 步、优化视图 DSC07987.JPG 和 DSC07989.JPG、GPU 首选 0。源 checkpoint SHA 固定为：

```text
14563fbbcfbb722879929be31fd19ad5e6a02939e8d2409e6a6855ce909d417c
```

测试使用 CPU 小张量和模拟 renderer 验证流程，不能代替真实 CUDA 预检。测试中原 renderer/Adam 源码集成项应使用已经准备好的 `external/gsplat-v1.5.3-ru-part-v3`，服务器上不应因缺少该源码而跳过。

## 3. 在 GPU 0 启动只读准备

```bash
.venv-gsplat153/bin/python tools/ru_part_v3_recoverability.py launch prepare --gpu 0
```

命令检查 GPU 空闲后后台启动，打印 PID、日志和状态命令。默认输出目录：

```text
logs-puri/ru_part_v3_coverage_recoverability_diag/garden_v3_final_diag_v1/
```

此阶段读取原始文件并写入新的诊断目录：

1. 校验原训练 161/24 划分、训练图内容、相机、归一化、静态轨迹缓存和 checkpoint。
2. 为 161 张训练图计算原始 C。使用源 `aux/mask_head_step29999.pt` 与缓存的 fine 特征，按原阈值 0.25、腐蚀核 7 恢复终态 M；计算所有 M 可用视图的 Q。
3. 写逐图覆盖 CSV、逐图等权和像素加权汇总、历史三个时间窗口统计。缺失 M 和零分母明确记录为空，不补零；稀疏日志不推算成全窗口均值。
4. 导出两张优化图的原图、终态 render、残差、C/M/Q 对齐图。
5. 按训练相机中心距离排序其他训练图，导出前 8 个候选的原图，供后续同一静态表面的多视图核对。此阶段没有 24 张 test 图像解码或指标评估。
6. 读取原 optimizer 构造代码，计算 step29999 更新时的终态学习率，记录初始参数摘要、源文件和导出数组 SHA、当前 GPU/驱动/Torch/gsplat 环境。

不做 Gaussian 参数更新，不修改原输出，不加载 DINO 模型，不恢复 histogram 或旧 optimizer，不执行旧的 10k 重放流程。

## 4. 查看准备状态并回传图像

```bash
.venv-gsplat153/bin/python tools/ru_part_v3_recoverability.py status
tail -n 80 logs-puri/ru_part_v3_coverage_recoverability_diag/garden_v3_final_diag_v1/prepare.log
```

期望 `PREPARATION_COMPLETE`、`exit_code: 0`。若为 `BLOCKED`，保留日志和已有覆盖统计，停止后续命令。两个优化视图只要有一张无法获得合法终态 M，就不会进入短优化。

准备完成后，用文件传输工具下载并发回：

```text
/home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/ru_part_v3_coverage_recoverability_diag/garden_v3_final_diag_v1/review_bundle.zip
```

包中含原图、对齐图、候选图、覆盖统计、相机候选排序与准备 manifest。图像必须以原文件传回，终端文本或压缩截图不足以确定逐像素 ROI。若需要存放在本地，使用 `E:\7-DataSet\ru_part_v3\coverage_recoverability_diag`。

## 5. Codex 绘制四视图静态 ROI，再由用户确认

收到图像后，Codex 根据原图和多视图一致性，选择两个排序靠前且能确认同一静态表面的训练检查视图。在四张原始 factor4 图上分别画多边形；不从单张残差、Mask 或测试视图推断“静态”。优先排除植被、反射和不确定遮挡，跳过更近候选时记录原因。

Codex 生成 `propose-roi` 的完整数据命令，用户无需编辑 JSON 或手写坐标。该命令输出四视图边界图及其 SHA：

```text
roi/proposal1/four_view_static_roi_review.png
```

此时请查看边界图，再确认四张图框选区域均为同一真实静态表面，且边界合理。确认针对静态标签和边界；短诊断本身已获本次任务授权。未确认前，程序不能锁定和启动 Va/Vb/O。

未确认草案可以用不同 proposal ID 修订，旧文件保留。确认后禁止修改。Codex 在收到明确答复后提供带实际 panel SHA 和确认原文的 `confirm-roi` 命令，不在此处预填确认。

## 6. 锁定预注册

完成上一步真实确认后执行：

```bash
.venv-gsplat153/bin/python tools/ru_part_v3_recoverability.py lock
```

该步骤检查原始与导出文件 SHA、代码 commit、GPU UUID、确认记录、M/C/S 和 ROI 来源。两个优化图合计必须满足 `(1-M)(1-C)S` 权重和及其残差加权量均非零，否则输出 `NO_ORACLE_INTERVENTION` 并结束。

成功后生成不可覆盖的 `diagnostic_preregistration.json` 和 SHA，固定：四张图、400 次交替相机序列、fresh Adam 设置、终态常量 LR、冻结数组和参数来源、环境、评价时点、保护区域规则以及所有阈值。锁定仅产生数据文件，无需另建 Git commit。

## 7. 串行运行 Va、Vb、O

```bash
.venv-gsplat153/bin/python tools/ru_part_v3_recoverability.py launch run
```

运行过程：

1. 在同一已锁定物理 GPU 上，用独立临时参数副本执行 1 次真实 CUDA forward/backward/Adam 更新。验证梯度和参数有限、参数确实变化、N 不变且源 checkpoint 未改，然后丢弃临时副本。
2. Va、Vb、O 分别重新加载同一源参数，建立独立零状态 Adam，每组固定 400 步，交替优化两张图。源拓扑、SH 阶数、M/C/S、学习率均固定。
3. Va/Vb 使用原 base loss 加 `0.8 * mean((1-M)C * |RGB-target|)`。O 将 C 替换为 `C+(1-C)S`，只增加原支持缺失部分；训练损失始终按整图 RGB 全局均值计算。
4. 每组在 0/100/200/400 次更新后对四张图进行诊断评估。每组 400 次训练 raster、400 次 Gaussian backward、16 次评估 raster；每 50 次更新输出一次进度。
5. 任一步异常立即中止，不启动后续组，不自动重试、续跑或扩大预算。

查看状态和日志：

```bash
.venv-gsplat153/bin/python tools/ru_part_v3_recoverability.py status
tail -n 80 logs-puri/ru_part_v3_coverage_recoverability_diag/garden_v3_final_diag_v1/run.log
```

长日志可使用 `tail -n 40 -f`；Ctrl+C 只退出查看，后台工作继续。

## 8. 读取判定并结束本次诊断

```bash
cat logs-puri/ru_part_v3_coverage_recoverability_diag/garden_v3_final_diag_v1/report.md
```

正式成功运行应为 `DIAGNOSTIC_COMPLETE`、三组 `COMPLETE`、每组 400 次实际更新且参数文件可加载。算法诊断标签是单独的字段，可能为：

- `PROMISING_LOCAL_RECOVERABILITY`：优化 ROI、检查 ROI 均改善且六个保护区域均通过。
- `LOCAL_ONLY_RESPONSE`：只达到局部改善门，保护区域通过。
- `NO_CLEAR_RECOVERABILITY_SIGNAL`：无明确局部可恢复信号。
- `COLLATERAL_ERROR_DETECTED`：任一保护区域失败；优先于改善结论。
- `DIAGNOSTIC_INVALID`：过程、来源或数值不合法，不能当作负结果。

判定用三组各自初态的均值及极差，并以 Va/Vb 终态差和初态极差的较大者构成工程波动 n。局部门为 `max(5% E0_T, 3n_T, 1e-4)`，检查门为 `max(2% E0_H, 3n_H, 1e-4)`，各保护区域门为 `max(1% E0_v, 3n_v, 1e-4)`。既比较初态，也比较更好控制组；只增加 alpha 不算恢复。

保存 `diagnostic_result.json`、报告、覆盖统计、预注册及 SHA、ROI 与确认、组内状态/指标/图像/参数文件。参数文件标明 `diagnostic_only=true`、源 step29999、400 次诊断更新，自身 step399，不冒充新的标准 30k checkpoint。

无论结果正负，本次到此停止。结果只约束该源状态、终态小学习率、fresh Adam 和 400 步预算，不证明 Jacobian 为零或必须 birth，也不改变历史 NO_GO，不启动下一算法或额外实验。
