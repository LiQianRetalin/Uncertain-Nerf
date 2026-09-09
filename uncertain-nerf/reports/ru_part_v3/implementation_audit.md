# RU-PART-V3 实施审计

审计日期：2026-09-09。依据用户附件实施；附件中的历史结论不是本次实测结果。

附件SHA-256：`d40bb0ecad7f4280b5477b0ba30b68aa8fece28e6276abb129b6a77a7d2f8908`。

## 1. 仓库与范围

- 当前主项目分支：`ru-part`。
- 修改前 commit：`e30564a2fe864a9b1561801c0fa5b13086d0e7c6`。
- 修改前工作区：干净；本次未覆盖用户未提交修改。
- 未切换主项目分支，未提交、上传、reset、rebase 或操作 dev。
- 仅在忽略目录 `external/gsplat-v1.5.3-v3-verify` 验证独立的、固定版本的 trainer 补丁。
- 仅新增 V3 / 标准 RU Parent 配置、轻量运行包装、报告与必要检查。旧配置默认行为保留。

## 2. 真实代码映射

| 对象 | 实际代码及行为 |
|---|---|
| 标准入口 | `run_puri_gs.py` → 固定 gsplat examples/simple_trainer.py |
| 本轮 Parent | `configs/puri_gs_ru_part_v3_parent_garden30k.yaml`，profile=ru，v3_screening=parent |
| 唯一候选 | `configs/puri_gs_ru_part_v3_v3_garden30k.yaml`，只切换 v3_screening=v3 |
| 基础 Gaussian loss | `puri_gs/semantic_mask.py::masked_photo_loss`；RGB 为 BHWC；Mask 为 B1HW；L1 对 B/H/W/RGB 全部求 mean |
| DSSIM | 同函数：原 M 分别乘 render/target，转 BCHW 后调用原 fused_ssim，padding=valid；V3 不替换这个表达式 |
| 最终 M | `puri_gs/ru_training.py::prepare_photo_mask`；概率先 bilinear/align_corners=False 上采样，再 detach；`hard_static_mask` threshold=.25，7×7 min-pool 腐蚀 |
| 新增项 | `puri_gs/ru_part_v3.py::static_rescue_l1`；输入没有 alpha；step<500 返回零，step≥500 全图 mean((1−M)C·abs(RGB−GT)) |
| 损失接入 | 新补丁在原全部 Gaussian loss 项之后增加 .8×rescue；原 Gaussian backward 仍只有一次；原 strategy 接收完整 loss 的正常梯度 |
| head | 仍为 `semantic_mask.py::StaticResponsibilityHead` 的384→16→1；DINO冻结、histogram、原 head backward/update 不变 |
| 常规 ADC | `delayed_absgrad.py::delayed_strategy_from_default` / `DelayedAbsGradStrategy`；复用原 DefaultStrategy grow/prune，不构造 PART 或 lineage 策略 |
| 独立 evaluator | 原 gsplat RU + efficiency-audit 评测路径；只载入标准 step/splats；额外保存同一次 DSC07988 渲染已有的 alpha 和绝对 RGB 误差 |

V3 补丁叠加顺序：原 robot baseline → 原 RU → 原 causal → 原 efficiency-audit → V3。准备脚本只改 Python examples，不更新任何包或 CUDA 源码。新补丁复用旧 PART 的训练图尺寸校准保护：训练阶段用 train index 1 校准，拒绝在训练过程中全目录生成 PNG 而打开测试图。

## 3. C 的实际定义

- 原构建器：`tools/build_puri_gs_static_tracks.py`，未修改 matcher 或构建参数。
- 上游张量：`track_evidence_binary`，形状 `[161,36,36]`，fine-grid 取值0/1。这是乘 Mask、alpha 或 residual 之前的 C 来源。
- schema：`puri-gs-ru-part-static-tracks-v1`。
- 载入：`static_tracks.py::load_static_track_cache`；校验原 canonical payload SHA-256。
- 上采样：原 `evidence_upsample`，bilinear，align_corners=False，无二次阈值化、归一化或区域扩张。
- 参数：grid=36、pose_neighbors=4、minimum_distinct_cameras=3、epipolar/reprojection tolerance=1 patch、factor=4、near=.01、far=1e10。
- 当前 component 筛选**存在**：互匹配节点做连通分量；同一 component 每个相机只能出现一次；至少3个相机；每节点度≥2才通过循环支持筛选；之后再走原三角化与几何筛选。这个条件比附件旧原型的简述更具体，本轮完整保留。
- 原构建器 main 会做两次同输入构建以检查一致性，其总成本由原 manifest 记录。本轮不删除这一已存在的检验，不搜索参数；已有合格缓存直接复用。
- V3 每步仅上采样当前训练相机的36×36支持图；整个缓存只解析一次。
- 原 builder 的 normalization 使用 float64 点，Parent parser 使用 float32 点。V3 用原 builder loader 重建**完全一致的缓存相机 hash**，另以绝对1e−5核对运行时相机差异；不把两种浮点表示的字节差异当作缓存变更，也不修改几何匹配容差。

**本机未找到 Garden 已验收 feature/static cache；实际缓存 SHA-256、build commit 和原构建成本目前未测。** 服务器 preflight 和运行时 manifest 将记录这些真实值，不能用代码 hash 代替缓存 hash。

旧 v2 schema 没记录源训练图的逐文件内容 hash。它可验证路径、split、相机、特征版本和 payload 完整性，但不能单靠该 schema 证明历史图像字节从未变化。复用前需要结合服务器原验收记录核对。V3 另外记录当前161张训练图和三个 COLMAP 文件的内容 hash，供本轮 Parent/V3 比较；不打开测试 RGB 做训练侧指纹。

## 4. 实际调度与默认值

| 项目 | 本轮固定值 |
|---|---|
| max_steps / 训练更新 | 30000；正式0..29999；smoke只在599提前终止，max_steps仍是30000 |
| Mask / 新 L1 开启 | step500 |
| fine 切换 | step20000；C使用fine缓存不影响head的coarse/fine调度 |
| ADC | step10000..19900，间隔100，共100事件；统计累积沿用原0..19999 |
| opacity reset | step15000、18000 |
| head pause | reset后的300步，原规则原样保留 |
| gsplat / AbsGrad | 1.5.3；保留现有1.5.3+pt24cu121这类构建后缀；AbsGrad=true |
| SH / ssim_lambda | 3 / .2；新增系数 .8 |
| grow_grad2d | .0006 |
| 官方其他策略默认值 | prune_opa=.005，grow_scale3d=.01，grow_scale2d=.05，prune_scale3d=.1，prune_scale2d=.15，refine_scale2d_stop_iter=0，pause_refine_after_reset=0，revised_opacity=false，key_for_gradient=means2d |
| 初始化 | sfm；原 COLMAP points3D；原 normalize_world_space=true；seed42；SH每1000步增加一级 |

上述默认值已从固定1.5.3源文件读取，非依据名称猜测。配置 loader 锁定本轮字段；运行时再核对实际 cfg/strategy。实际 tyro 完整解析帮助在本机被缺失依赖阻断，待服务器 preflight 核验；本地预期配置不可写成“服务器解析已通过”。

## 5. Parent / cap 审计

`ru_part_strategy_from_default` 在旧 current 模式构造 `RUPARTStrategy`，其数量控制使用 `N_MAX`。当前源码的旧 parent/noop 构造 `LineageDelayedAbsGradStrategy`，本身**没有**PART cap，但重写 grow/prune并记录lineage。因此“旧parent配置写了gaussian_hard_cap”不等于该模式实际执行了cap，也不能仅凭parent名称认定其完全等价于本轮标准RU。

本轮使用原 `DelayedAbsGradStrategy`，从结构上绕开：

- `RUPARTController.__init__` 的出生几何准备；
- 旧 `ru_part_rescue_loss` 的 `(1−alpha)` 路径；
- 旧 trainer event中的fixed-opacity额外raster；
- `prepare_event / record_noop_event / lifecycle_counts`；
- `RUPARTStrategy._grow_gs` 的 B/H、donor、birth和cap；
- lineage、prospective scoring及重放保存/恢复。

禁用项的零调用以这些对象不构造、入口不接入为依据；训练另外实际计数标准raster和Gaussian backward。600步期望1200次训练raster（包含Parent原有coarse路径），600次Gaussian backward；30k期望50000次训练raster，30000次Gaussian backward。训练结束后的两张证据图前向单独进行，不计入训练计数或训练时间。

2,312,002只在最终报告中检查，从不进入策略。

## 6. 数据、环境与可比记录

- 本地真实Garden：`E:\7-DataSet\nerf数据集\mipnerf360\360_v2\garden`。
- 原图文件清单185张；按every-eighth预期161/24；DSC07987排序31、DSC07989排序33，为train；DSC07988排序32，为test。
- 本地没有 `images_4_png`。完整 COLMAP parser运行和split/camera映射仍在服务器核对；上述文件名检查不冒充完整parser验收。
- 服务器路径与GPU依据既有runbook：`/home/chenglong/Uncertain-Nerf/uncertain-nerf`，GPU6/L20；本次SSH免密访问返回`Permission denied (publickey,password)`，未实时核实。
- 历史runbook记录PyTorch2.4.0+cu121、gsplat1.5.3+pt24cu121；当前服务器环境由preflight实际记录。
- 本机WSL现有 `.venv-gsplat153` 能运行CPU测试；PyTorch2.4.0+cu121。
- 本机RTX5070Ti为sm_120，实际CUDA张量运算报`no kernel image is available for execution on the device`；训练trainer帮助报缺少`tyro`。未更新PyTorch、gsplat或CUDA。
- 当前没有本地可读取、可核对的标准Parent完整checkpoint及同口径成本。历史B1数值只作附件指定恢复参考，不冒充本次重评。
- preflight列出服务器历史RU/旧parent候选；存在候选时，包装器阻止未经可比性审计就额外跑Parent。仅缺计时不能否定其质量可比性；若能复用，应先完成复用登记和0.001dB重评核对，再继续。**历史Parent自动复用适配尚待服务器实际记录映射，不能把目前未访问的数据编造成已适配。**
- 若证据确认没有算法与数据可比Parent，只补一次标准Parent。先向用户说明原因和额外成本，不自行做多seed。

## 7. 初次本地验证状态（后续服务器结果见第8、9节）

- 固定张量、旧集成兼容、结果判定和checkpoint完成条件测试：27项通过。
- float32 loss最大误差1.65e−8；float64约1.39e−17；固定RGB新增梯度最大误差0。
- Python语法、Bash语法、补丁正向/反向检查及准备脚本幂等检查已通过。
- 项目launcher、阶段包装器、preflight命令帮助已通过。
- 真实trainer完整帮助：本地失败，缺tyro；服务器预检将再次实际执行并阻止失败后的训练。
- E2真实叠图、600步CUDA smoke、profiler、30k训练、独立评测及质量/效率验收：**未执行**。
- 本地检查日志位于Git外：`E:\7-DataSet\PURI-GS-derived\ru_part_v3_local_checks`。

旧状态始终为`REPLAY_NOT_EQUIVALENT`，没有被本次CPU测试重新分类。

## 8. 用户提供的服务器预检补充

后续粘贴输出确认服务器分支`ru-part`、commit `e85dbd413c8a7e890bbc2640f8aa9d4a3f800129`，`PREFLIGHT_READY`，14项V3专项测试通过，实际trainer帮助已通过。以下属于用户提供的服务器结果，不是本机直接登录读取。

- gsplat trainer SHA-256：`51813287fb19d556a06585bb478f05d8c65da56aee5c71ce9146f7a5c05a6675`。
- 已有缓存：`data/PURI-GS-derived/static_tracks/garden_factor4_v1/garden_factor4_static_tracks.pt`。
- cache file SHA-256：`2489df8a6586f95731ac9ee7948bf0c27f9cdf8c62afbe32965bcd4e637ba1a7`。
- payload SHA-256：`327101f76a227e2e0f3129b7fba548a91e38d41c6eef41d672359a0d513f4f94`。
- build commit：`fe12483a7af0ab09b6a5789c4cb2edf0e47a2559`。
- feature manifest SHA-256：`c43f9628d44bf7069781ce4e061d4bd0f3316bf3764878592d602f4cb3137d1b`。
- 参数与本协议固定值一致；161张训练图；fine-grid非零比例`.0042030904442071915`（约0.4203%）。该比例不是上采样后C或Q覆盖率，也不能据此宣称孔洞证据充分。
- 两个运行具有标准step29999 checkpoint：`logs-puri/ru-generalization-rerun-9e292309/garden_ru_30k`、`logs-puri/ru_part_reference_controls/parent_seed42`。其配置、原质量和数据/成本可比性仍待核对；不因旧运行缺少后来新增的`v3_input_manifest.json`就直接否定旧质量对照。
- 本次用户指定优先GPU0。快照中0号L20为71MiB/0%利用率，6/7号各有约30GiB VLLM计算进程。现有服务器命令支持`preflight --gpu 0`，无需为参数选择再次同步代码；后续本轮对照固定同卡。

本节预检阶段尚未收到600步smoke或30k训练完成结果；后续Parent smoke结果见第9节。

## 9. 用户提供的 GPU 0 Parent smoke 结果

用户后续粘贴的服务器状态和日志确认：

- 阶段：`smoke-parent`；实际命令`--gpu 0`，`--v3-stop-step 599`，训练调度仍为`--max_steps 30000`。
- 状态：`SMOKE_COMPLETE`，`exit_code=0`，`last_step=599`；完成step 0..599共600次更新。
- 标准checkpoint存在且可加载；Gaussian数138766；checkpoint SHA-256：`eed4884e48bb4245b991652b3ee0e9c86f91a6e6bb2a429c4d86f4268c7d35a6`。
- Parser实际记录185张图像、161张train/24张test；DINO参数22058112、trainable=0。
- 子进程总墙钟时间20.17361391405575秒；step599训练统计`ellipse_time=11.217679262161255`秒。这两项不能替代step520..539的profiler中位数，也不能用于宣称30k训练成本门通过。
- `599/30000`和2%为保留30k调度的预期提前停止表现。xFormers/torch.load和reconstruction image_path警告没有导致本次运行失败。

上述结果来自用户附件，未直接读取服务器checkpoint；`v3_training_checks.json`原始明细仍待收集。本次完成仅证明Parent smoke阶段完成，不证明V3新增监督已经激活。下一阶段为同一GPU 0上的独立`smoke-v3`。历史完整Parent可比性审计、V3 E2/Q覆盖、配对profiler、正式30k训练与独立评测仍待完成；总体保持`COMPARISON_INCOMPLETE`。

## 10. V3 smoke 启动参数兼容修复

用户提供的首次V3 smoke状态为`FAILED / exit_code=1 / last_step=-1`，包装PID 2289957、child PID 2290031，子进程约1.136秒即退出。错误发生在`run_puri_gs.py`创建结果目录和启动trainer之前：旧`is_ru_part`分支后的参数拒绝检查，将合法V3的`--track-cache`误认作旧PART专属参数。之前测试覆盖了配置、损失和补丁，但未经过包装器到launcher主入口，未发现这个入口错误。

修复将该缓存拒绝条件限定为非`v3_screening=v3`；没有整体跳过旧参数保护，V3的replay/diagnostic等参数仍拒绝，Parent仍拒绝静态缓存。trainer补丁、训练公式、调度和缓存内容未改，成功Parent smoke保留。新增13项真实CLI入口回归检查，本地V3、GPU与旧PART集成合计47项通过（11.95秒）；外部gsplat/数据检查在这些CPU测试中被隔离，CUDA V3 smoke仍待服务器修复重试。

runbook提供仅适用于本次step=-1启动失败的归档流程，保留原错误日志/状态和预检快照；同步修复后重新`preflight --gpu 0`登记当前commit，再重新初始化V3 smoke。没有将失败状态改为成功，也没有续训失败快照。
