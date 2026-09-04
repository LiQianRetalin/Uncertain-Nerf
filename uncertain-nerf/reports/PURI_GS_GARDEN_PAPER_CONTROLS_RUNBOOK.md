# Garden RU-Align / RU-TAR 论文对照：代码里程碑与停点

本阶段只实施两个固定论文对照，不是最终算法，不实现自适应拓扑或阶段 U。
继续使用 dev；不新建分支或 worktree，不运行 Android、Room、Patio-High、额外 seed 或参数搜索。

## 阶段 A 已复核的证据

2026-09-04 用户提供了完整只读审计输出和原始 2×2 Markdown/JSON。
最终标志为 `GARDEN-PAPER-CONTROL-READONLY-AUDIT-PASS`，服务器 dev commit 为
`6e54008e244722b977a9a2b2c54e10207413b991`。GPU 6 为 NVIDIA L20；
PyTorch=2.4.0+cu121，CUDA=12.1，gsplat=1.5.3+pt24cu121。

服务器已有未跟踪报告和 data-transfer 目录必须保留；本次没有复制同名历史报告进 Git，
避免后续 pull 与服务器原有文件冲突。本地用户的
`reports/GARDEN_2X2_CAUSAL_ATTRIBUTION_ANALYSIS_FOR_GPT.md` 也保留且不纳入本次代码提交。

服务器项目目录：`/home/chenglong/Uncertain-Nerf/uncertain-nerf`。

- Garden：项目内 `data/mipnerf360/360_v2/garden`，161 train / 24 test，factor=4。
- cache：项目内 `data/PURI-GS-derived/semantic_features/garden`；161张图 coarse/fine 均通过加载、形状和有限值校验。
- DINO源码：项目内 `external/dinov2`，commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`。
- DINO权重：项目内 `data/PURI-GS-assets/dinov2/dinov2_vits14_reg4_pretrain.pth`，SHA256 `f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb`。
- 四组质量、数量、训练/评测commit和24图同序性与2×2最终报告一致。

历史 RU 的 Gaussian 数：9999→10000 为 138766→230320；14999→15000 为
2238240→2250200；17999→18000 为 2284941→2288647。不能把同时发生的
refine/reset 对数量的影响混为单独 reset 因果效果。
旧 DG-only 控制台有100条refine及2条reset；Mask-only有144条grow及144条prune。
旧运行没有本阶段所需完整聚合表；不补跑历史实验。

固定代表图按历史 RU−B1 PSNR 排序后取第0、5、9、14、18、23位：
DSC07988.JPG、DSC08020.JPG、DSC08060.JPG、DSC08100.JPG、DSC08036.JPG、DSC08124.JPG。
测试图名称序列 SHA256：`37de270739918344ffcddb55039ee15c4d65246a237b505910d2023de9269454`。

## 阶段 B 实现

| 配置 | Mask switch | refine | reset | stats累计 |
|---|---:|---|---|---|
| 正式RU，未修改 | 20000 | 10000..19900 /100；100次 | 15000、18000 | 0..19999 |
| puri_gs_ru_align_full30k.yaml | 10000 | 与RU相同 | 与RU相同 | 与RU相同 |
| puri_gs_ru_tar_full30k.yaml | 10000 | 早期100次；20000..23800 /200晚期20次 | 与RU相同 | 0..23999 |

`paper_control` 是显式身份/日志标记。机器可读diff将该元数据与算法字段分列。
Align的唯一算法字段差为`bootstrap_switch_step`；TAR唯一额外因素为`refine_windows`，
包括该窗口内继续统计和20次refine。没有更改停止reset的原始边界，因此没有21k reset。

沿用同一DefaultStrategy的grow/prune操作，不复制trainer或策略。统计从0开始，
包括本次梯度，refine后清零；顺序保持Gaussian backward、head独立更新、
post-backward（grow/prune/reset）、Gaussian optimizer。10k本步fine先于refine。

新日志：`resolved_schedule.json`、`resolved_config_diff.json`、`topology_events.csv`、
`mask_aggregates.csv`、`paper_control_validation.json`。
split按一个父Gaussian替换为两个子Gaussian计数，所以 `after=before+split+clone-prune`。
topology记录实际reset执行标志；Mask只在指定11步汇总。
eligible/grad2d/opacity额外聚合统一为`not_available`：现有grow/prune只返回数量，
不为了日志增加逐Gaussian归约或每步设备同步。日志不参与训练决策。

完整渲染次数由训练实际调用记录；每步一次完整渲染，10k前另有原有224×224路径。
独立评测继续原来的standard splats路径，不传DINO/head/cache参数。
标准checkpoint仍只有step/splats；head和histogram仍在aux。

结果目录严格拒绝任何已存在目录（包括空目录）。正式服务器输出保留为：

- `logs-puri/phase_r_paper_controls/garden_ru_align_30k/`
- `logs-puri/phase_r_paper_controls/garden_ru_tar_30k/`
- 控制台日志目录：`logs-puri/phase_r_paper_controls-console/`

`tools/audit_puri_gs_paper_controls.py`支持配置/调度审计和训练/独立评测证据核验。
`tools/summarize_puri_gs_garden_paper_controls.py`支持RU-Align中间停点报告，
以及六组最终报告、paired bootstrap、门禁、单位容量收益和6图轻量拼图。
仅在真实运行完成并核验后使用；本次没有生成实验结果报告。

## 本地验证记录

- 全量CPU pytest：233 passed，2 skipped。
- 与修改前commit代码比较：B1/RU/DG-only/Mask-only的配置、CLI参数和事件合同全部一致。
- 真正的DefaultStrategy在CPU执行clone/split/prune，验证数量恒等式、优化器参数接线及统计清零。
- 两个30k调度逐步模拟，100/120事件、reset/head暂停、24k冻结及渲染次数合同通过。
- 旧梯度隔离、DINO冻结、Mask阈值/7×7、cache映射、checkpoint和独立评测测试通过。
- 新结果验证拒绝错序、NaN、标准splats中混入head、事件不符和覆盖。
- 增量补丁在现有causal+efficiency/On-the-go补丁栈临时副本应用及重复执行通过。
- 当前trainer的Config和CLI参数使用隔离的tyro依赖验证；不会更新项目PyTorch/CUDA。
- Python compileall通过；Shell语法通过（旧Windows脚本按Linux换行审查，新脚本原文件也通过bash -n）。
- 两个真实launcher的30k参数dry-run通过，不创建输出目录。
  本地dry-run只核验参数/路径接线，DINO repo/cache使用已有测试占位目录；
  正式cache有效性来自阶段A服务器证据，不能把dry-run当作CUDA smoke。
- `git diff --check`通过。新实验尚未执行CUDA smoke或30k。

## Git里程碑1：当前用户操作

本步骤目的：一次提交并push已自检的两个论文对照代码。

执行位置：Windows本地，使用图形化Git。

需要打开的目录：`E:\6-Project\1-UncertainNerf`（Git仓库根目录）。

需要检查的文件：只选择下面15个文件。

修改的4个文件：

1. `uncertain-nerf/puri_gs/config.py`
2. `uncertain-nerf/puri_gs/delayed_absgrad.py`
3. `uncertain-nerf/puri_gs/ru_training.py`
4. `uncertain-nerf/run_puri_gs.py`

新增的11个文件：

1. `uncertain-nerf/configs/puri_gs_ru_align_full30k.yaml`
2. `uncertain-nerf/configs/puri_gs_ru_tar_full30k.yaml`
3. `uncertain-nerf/patches/gsplat_v1.5.3_puri_gs_paper_controls.patch`
4. `uncertain-nerf/puri_gs/paper_controls.py`
5. `uncertain-nerf/scripts/prepare_puri_gs_paper_controls.sh`
6. `uncertain-nerf/tests/test_paper_controls.py`
7. `uncertain-nerf/tests/test_paper_control_summary.py`
8. `uncertain-nerf/tools/audit_puri_gs_paper_controls.py`
9. `uncertain-nerf/tools/summarize_puri_gs_garden_paper_controls.py`
10. `uncertain-nerf/reports/PURI_GS_GARDEN_PAPER_CONTROLS_RUNBOOK.md`
11. `uncertain-nerf/reports/phase_r_garden_paper_controls_code_audit.json`

需要执行的命令：无。确认分支dev，图形化Git提交说明填写
`实现Garden的RU-Align与RU-TAR论文对照实验`，提交后push一次。

命令执行完成后应看到：15个小文件已提交，push成功；原有未跟踪归因分析报告仍保留。

日志位置：图形化Git的提交详情和push记录；本地机器审计为本目录code_audit JSON。

如何判断成功：新commit包含上述15文件且远端dev同步；四个正式旧配置不在变更列表。

出现什么情况应立即停止：分支非dev、文件超出清单、出现数据/cache/权重/checkpoint/训练日志/
PNG/tmp、发生冲突或push失败。不要清理、覆盖或重置用户改动。

是否需要Git提交：是，一次。

是否需要push：是，提交后一次。

是否需要服务器pull：当前否。先将新commit发给Codex复核，之后才给服务器一次pull及环境检查步骤。

完成后需要发给Codex的内容：完整新commit、15文件列表截图、push成功状态。

## 后续必须保持的停点

1. 复核push → 服务器一次pull与环境检查 → 独立目录最小CUDA smoke；任何失败均停止。
2. smoke通过后只给RU-Align 30k命令；完成训练审计后只做一次独立评测。
3. RU-Align中间证据报告后暂停。无论质量是否改善，都不改参数；用户确认后才继续RU-TAR。
4. RU-TAR训练/独立评测完成后生成
   `PHASE_R_GARDEN_RU_ALIGN_TAR_PAPER_CONTROL.md`和
   `phase_r_garden_ru_align_tar_paper_control.json`；存在则拒绝覆盖。
5. 最终报告完成后停止。小报告经复核才进入Git里程碑2，不提交大文件，服务器暂不再次pull。

本手册不提供尚未获前一步证据支持的正式训练命令。

## 阶段 D 首次 smoke 失败与序列化修复

阶段 B 的代码已由用户提交至 `e0915cefdd7cee71092388aa5078206b8964817c`，
额外包含原有的小型归因分析报告；该提交已复核，无需移除这份报告。
阶段 C 服务器审计通过：dev/commit、14个文件指纹、L20 GPU 6、固定环境、
161/24划分、全部161张coarse/fine缓存以及两个正式结果目录不存在。

用户随后上传了 SHA256 为
`c8a8e7510ab35cb2ac8e495cf1d98d87de69294a343b7513e4638468b94bfbdd`
的临时阶段 D 脚本。增量补丁和 Align dry-run 通过；DINO 成功加载且
trainable=0，但 `Runner.train()` 在执行第0步之前的 `yaml.dump(vars(cfg), f)`
失败，错误为 `TypeError: cannot pickle '_csv.writer' object`。
因此 Align 100步、fine组件检查、两组独立评测及TAR短跑均不能视为通过。

原因：`cfg.strategy.event_recorder` 是绑定到 `PaperControlRecorder` 的方法，
PyYAML 序列化策略时递归访问其持有的CSV writer。
修复在 `DelayedAbsGradStrategy.__getstate__` 中返回配置状态的浅拷贝，
只将拷贝中的 `event_recorder` 设为 None。运行中的回调不变，策略参数和调度不变。
不修改gsplat补丁、CUDA、损失、正式配置或checkpoint格式。

新增回归测试先在未修复代码上复现 Align/TAR 两例同样错误，再验证
RU/Align/TAR 的 YAML 写出、序列化往返、内存回调保留和后续CSV写入。
两组完整30k调度模拟也在绑定真实CSV回调并写出YAML后执行。
本地仅将系统已有PyYAML 5.4.1复制到被Git忽略的临时测试目录供CPU测试使用，
没有安装或升级训练环境依赖。完整测试结果见机器审计JSON的 `serialization_fix`。

当前只提交以下4个修复文件，建议备注：`修复论文对照训练配置序列化失败`。

1. `puri_gs/delayed_absgrad.py`
2. `tests/test_paper_controls.py`
3. `reports/PURI_GS_GARDEN_PAPER_CONTROLS_RUNBOOK.md`
4. `reports/phase_r_garden_paper_controls_code_audit.json`

提交并push后先复核新commit，再提供一次服务器pull与更新的smoke命令。
旧临时脚本锁定e0915cef及其输出目录，不能直接重跑。
保留 `logs-puri/phase_r_paper_controls-smoke-e0915cef/` 失败证据，
恢复时使用新目录并锁定修复commit；不得删除旧目录或手改服务器受跟踪代码。
