# RU-PART-V3 逐步操作说明

本地代码与CPU检查已完成；还没有CUDA smoke、正式训练或质量结论。请按顺序执行，前一阶段失败就停止后续阶段。以下服务器目录/GPU来自项目已记录的实际设置，本次SSH连接无权限，必须以第2步预检为准；不会把Windows数据路径传给Linux。

## 第1步：一次正常代码提交和同步

| 项目 | 操作 |
|---|---|
| 在哪里执行 | 本地和服务器各自的Git图形界面 |
| 前置条件 | 本地仍为ru-part；查看本次更改列表 |
| 执行什么 | 只提交代码、2个配置、补丁、测试和本目录说明；备注：**实现RU-PART-V3静态监督恢复与从头筛选流程**。本地上传；服务器确认ru-part后下拉同一提交 |
| 查看进度 | 图形界面的提交/上传/下拉状态 |
| 成功标志 | 两端图形界面显示同一commit；没有数据、缓存、checkpoint、训练日志或诊断图进入提交 |
| 失败如何处理 | 停止后续；发回图形界面的报错文本和分支/commit |

本地数据/检查日志保存在`E:\7-DataSet`。服务器运行产物写入被忽略的`logs-puri/ru_part_v3_screening`和既有`data/PURI-GS-derived`。查看日志、评测、生成报告不需要再次Git操作。若发现真实代码错误，修复后才需要一次对应修复提交。

## 第2步：服务器环境、路径和缓存预检

在哪里：服务器终端。前置条件：第1步同步完成；使用服务器已有`.venv-gsplat153`，不要更新PyTorch/gsplat/CUDA。

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
.venv-gsplat153/bin/python tools/ru_part_v3_screen.py preflight --gpu 6
```

脚本核对分支、CUDA设备、gsplat版本、数据/PNG、DINO、feature cache，查找已有静态缓存，并列出历史Parent候选。它在专用`external/gsplat-v1.5.3-ru-part-v3`里准备原版本Python补丁，不动已有RU/PART例程；运行CPU测试和真正的trainer命令帮助。

查看结果：

```bash
cat /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/ru_part_v3_screening/preflight.json
tail -n 80 /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/ru_part_v3_screening/trainer_help.txt
```

成功标志：`PREFLIGHT_READY`；缓存缺失时为`PREFLIGHT_CACHE_REQUIRED`，进入第3步。`environment_resolved.json`保存本机实际解析路径、Python、GPU和commit。

失败：停止；发回完整预检终端输出、`preflight.json`和trainer帮助日志最后80行。若`cd`失败，发回服务器项目当前绝对路径，不能继续照抄后续路径。若缓存多于一个，发回预检候选清单，由实际v2验收记录确定缓存；不要自行选最新文件。若缺feature cache，停止核对现有资产，不构建新matcher或修改证据条件。

**历史Parent核对是正式训练前的必要步骤。** preflight若列出历史候选，发回`preflight.json`，继续核对其原配置、checkpoint、数据与图像内容、软件/设备、原指标和成本记录。脚本会阻止尚未审计时额外启动Parent。当前无法访问服务器历史文件，因此未预先伪造`parent_comparability_audit.json`。可复用时先完成该实际运行的接入和固定checkpoint重评（PSNR复核阈值0.001dB）；只有确认不可比，才记录`NO_COMPARABLE_EXISTING_PARENT`并补一次Parent。缺少同环境时间只影响时间门，不否定质量对照。

## 第3步：仅在静态缓存缺失时构建一次

在哪里：服务器终端。前置条件：第2步明确`PREFLIGHT_CACHE_REQUIRED`，原v2构建器、DINO feature cache和PNG数据均存在。已有合法缓存跳过本步。

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
.venv-gsplat153/bin/python tools/ru_part_v3_screen.py launch cache
```

此命令调用原v2构建器、原固定参数；保留其原有两次相同构建一致性检查，并计入总准备成本。

```bash
tail -n 80 /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/ru_part_v3_screening/cache.log
tail -n 40 -f /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/ru_part_v3_screening/cache.log
```

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
.venv-gsplat153/bin/python tools/ru_part_v3_screen.py status cache
```

成功标志：状态`CACHE_COMPLETE`且exit_code=0，真实缓存和SHA-256已保存。然后重新执行第2步预检。失败发回状态和`cache.log`最近80行；没有合法tracks时停止，不放宽筛选条件。

## 第4步：标准Parent 600步开销基准

在哪里：服务器终端。前置条件：预检通过，没有其他GPU训练竞争；本轮只运行一个任务。

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
.venv-gsplat153/bin/python tools/ru_part_v3_screen.py launch smoke-parent
```

```bash
tail -n 80 /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/ru_part_v3_screening/smoke-parent.log
tail -n 40 -f /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/ru_part_v3_screening/smoke-parent.log
```

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
.venv-gsplat153/bin/python tools/ru_part_v3_screen.py status smoke-parent
```

成功标志：`SMOKE_COMPLETE`、exit_code=0、last_step=599，标准checkpoint可加载。调度始终是30k，只在第600次更新后停止；不会执行10k拓扑，不是恢复实验。step520..539做同口径有界计时。失败发回状态和最近80行日志。

## 第5步：V3 600步smoke和训练证据图

前置条件：第4步`SMOKE_COMPLETE`。在哪里：服务器终端。

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
.venv-gsplat153/bin/python tools/ru_part_v3_screen.py launch smoke-v3
```

```bash
tail -n 80 /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/ru_part_v3_screening/smoke-v3.log
tail -n 40 -f /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/ru_part_v3_screening/smoke-v3.log
```

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
.venv-gsplat153/bin/python tools/ru_part_v3_screen.py status smoke-v3
cat logs-puri/ru_part_v3_screening/smoke-v3/v3_training_checks.json
cat logs-puri/ru_part_v3_screening/smoke-v3/evidence_check/evidence_check.json
```

成功标志：`SMOKE_COMPLETE`，step599、有效checkpoint、正常Gaussian/head更新；实际训练raster=1200、Gaussian backward=600、birth/probe/scoring/cap/lineage均为0。新增监督若在本次采样激活，会记录active_samples；没有采样到Q时做独立固定张量功能检查，并明确记录`NO_Q_SAMPLED`。

产物包括两张固定训练图的GT/render/residual/M/C/Q/overlay和拼图；使用smoke自身同run同step Gaussian/head。这是**早期激活检查**，不是中后期孔洞覆盖结论。C与Q的统计、top20%高残差区域覆盖、36×36节点追溯也在目录中。

完整训练启动前会检查两次smoke的step520..539中位耗时比。若>1.08，停止排查实现开销，不改算法参数。600步不能保证完整30k预算。

失败：停止后续；发回状态、最近80行日志、training_checks与evidence_check。全训练侧合法C全零时停止候选；某个审计状态Q=0不等于整个训练无监督。

## 第6步：必要时补一次标准Parent 30k

**仅在已确认现有Parent不可比时执行。** 用户应先收到明确原因和“将增加一次Parent训练”的说明。若现有Parent能复用，应改用经实际记录核验后的复用路径，本步不执行；当前脚本不会自动把旧lineage parent改名成标准Parent。

前置条件：第4/5步完成，Parent可比性审计结论允许补跑。

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
.venv-gsplat153/bin/python tools/ru_part_v3_screen.py launch parent
```

```bash
tail -n 80 /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/ru_part_v3_screening/parent.log
tail -n 40 -f /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/ru_part_v3_screening/parent.log
```

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
.venv-gsplat153/bin/python tools/ru_part_v3_screen.py status parent
```

成功标志：`TRAIN_COMPLETE`、exit_code=0、last_step=29999、标准checkpoint存在且完整可加载。训练从step0重新初始化，不加载smoke或旧失败快照。失败发回状态和最近80行日志，不自动续训或重跑。

## 第7步：唯一一次Garden V3 30k

前置条件：前述最小验证通过，标准Parent对照已核对。在哪里：服务器终端。

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
.venv-gsplat153/bin/python tools/ru_part_v3_screen.py launch v3
```

```bash
tail -n 80 /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/ru_part_v3_screening/v3.log
tail -n 40 -f /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/ru_part_v3_screening/v3.log
```

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
.venv-gsplat153/bin/python tools/ru_part_v3_screen.py status v3
cat logs-puri/ru_part_v3_screening/v3/v3_progress.json
```

成功标志同第6步，必须四项同时满足：进程exit0、step29999、checkpoint存在、checkpoint可加载。最终标准模型：`logs-puri/ru_part_v3_screening/v3/ckpts/ckpt_29999_rank0.pt`。正式训练从step0新开始，不续接smoke。记录base/additional loss、M/C/Q、alpha摘要、Gaussian数量、原ADC计数、时间和显存；不读test RGB做训练选择。

失败/OOM：停止后续；发回状态和最近80行日志；不延长训练、改阈值、增加birth、截断Gaussian数量或自动续训。

## 第8步：独立评测全部24张test图

分别等待前一个阶段完成再启动下一个；只对最终checkpoint算一次质量。下面第一项先启动Parent评测。

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
.venv-gsplat153/bin/python tools/ru_part_v3_screen.py launch eval-parent
```

```bash
tail -n 80 /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/ru_part_v3_screening/eval-parent.log
tail -n 40 -f /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/ru_part_v3_screening/eval-parent.log
```

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
.venv-gsplat153/bin/python tools/ru_part_v3_screen.py status eval-parent
```

出现`EVAL_COMPLETE`后：

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
.venv-gsplat153/bin/python tools/ru_part_v3_screen.py launch eval-v3
```

```bash
tail -n 80 /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/ru_part_v3_screening/eval-v3.log
tail -n 40 -f /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/ru_part_v3_screening/eval-v3.log
```

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
.venv-gsplat153/bin/python tools/ru_part_v3_screen.py status eval-v3
```

成功标志：两者`EVAL_COMPLETE`、exit0；24图名称完整一致，per-image和mean指标一致，未加载DINO/head，单次标准raster。计时都使用10次warmup及原同步协议。DSC07988的alpha来自这同一次标准渲染。失败发回对应状态和最近80行日志。

## 第9步：报告和停止

在哪里：服务器终端。前置条件：独立评测完成；失败或缺项也可生成不通过/不完整报告。

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
.venv-gsplat153/bin/python tools/ru_part_v3_screen.py report
cat logs-puri/ru_part_v3_screening/RU_PART_V3_SCREENING_REPORT.md
```

结果：`RU_PART_V3_SCREENING_REPORT.md`、`screening_result.json`、`screening_per_view.csv`。

报告自动复用唯一找到的历史B1/RU-TAR保存数组，用原`top_connected_component`和原阈值恢复固定孔洞ROI，报告ROI绝对误差、alpha及低alpha面积。找不到或存在多个历史来源时标`NOT_ASSESSABLE`，发回报告核对具体历史路径；不会从V3测试结果临时选ROI。B1表中当前是附件给定历史参考，不冒充B1重评产物。

固定门：PSNR≥27.566728、SSIM≥.868957、LPIPS≤.073151、DSC07988 PSNR≥19.491301、数量≤2312002、完整训练时间比<1.10、单raster、FPS比≥.95。另检查相对Parent PSNR/SSIM不降、LPIPS不升。缺少可比对照/成本/ROI时不会整体晋级。

发回报告及`screening_result.json`即可。然后停止，本轮不自动开展跨场景、多seed、Oracle/VJP、重放、剪枝或下一算法。`PROMISING_SINGLE_SEED`也只表示值得继续正式验证。

## 所有后台任务的共同说明

- `launch`返回只代表已启动。`status`会显示包装进程PID、child_pid、开始/结束时间、最新step、exit_code和最终状态。
- 包装器使用与nohup相同目的的独立会话（start_new_session），日志输出到固定文件；关闭终端不会靠日志窗口维持训练。
- `tail -f`时按Ctrl+C只停止查看日志，不停止后台训练。
- 查看真实进程和设备负载，可根据`status`打印的PID核查；无需自己猜PID，也可直接使用下面的整机只读命令：

```bash
ps -eo pid,ppid,etime,stat,args | grep -E 'ru_part_v3_screen.py worker|simple_trainer.py' | grep -v grep
nvidia-smi -i 6
```

- PID消失但无完成标志为未完成，不能当成功。已存在输出会被拒绝覆盖；异常后先发回状态和最近80行日志，不自行删除结果目录或重复运行。
