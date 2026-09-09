# RU-PART-V3 逐步操作说明

当前进度：优化后的V3短测开销比值1.0634138514755154，通过1.08检查线。历史标准RU检查点已在GPU 0完成重评，exitcode=0，24张测试图匹配，PSNR差为0，标准独立评测检查通过。下一步同步历史Parent复用接入代码，登记后从头启动唯一一次V3 30k。下面旧重评、短测及故障处理流程保留供追溯，已执行步骤不要重复。服务器结果来自用户记录，本机未直接登录。

## 当前操作：登记历史Parent，再启动V3 30k

1. 在本地Git图形界面提交并同步本次代码与报告改动；服务器仍使用`ru-part`分支同步。此次新增`puri_gs/v3_parent_reference.py`和对应测试，更新启动器及报告，使其直接引用旧训练目录和已完成的重评目录。没有训练公式、trainer补丁或缓存变动，不重做两组smoke。

2. 服务器终端更新预检记录并登记。这两项成功后才启动训练；`&&`保证前一项失败时不执行后一项。

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
.venv-gsplat153/bin/python tools/ru_part_v3_screen.py preflight --gpu 0 && \
.venv-gsplat153/bin/python tools/ru_part_v3_screen.py register-parent
```

成功标志为`PREFLIGHT_READY`和`PARENT_REFERENCE_READY`。登记只读取配置、实际`cfg.yml`策略与调度、split、当前训练图/SfM哈希、标准29999检查点、退出码及重评指标，写入独立`parent_reference.json`。不会创建`parent.status.json`或把旧训练冒充本轮新训练；也不会再次评测Parent。若出现配置差异或文件缺失，保留错误并发回，不能手动改状态跳过。

质量复用依据是原标准RU配置/路径/split、实际运行策略、同名检查点复现和当前数据与配对短测一致。历史训练时未记录逐文件图像/SfM哈希和30k相机序列，登记会明确写`NOT_RECORDED`，不会用当前哈希声称证明历史字节身份。历史训练来自GPU 1，完整训练时间门保留`NOT_ASSESSABLE`；本轮GPU 0重评的143.3705007690993 FPS用于后续同设备、同软件、同warmup推理比较。

3. 看到`PARENT_REFERENCE_READY`后，在服务器启动一次完整V3：

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
.venv-gsplat153/bin/python tools/ru_part_v3_screen.py launch v3
```

GPU固定为0；占用时停止并报错，不抢占其他进程。训练从step0初始化，终点29999，不从smoke或历史Parent续训。背景包装器打印PID、日志与状态命令。已有同名输出时拒绝覆盖。

```bash
tail -n 40 -f /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/ru_part_v3_screening/v3.log
```

`Ctrl+C`仅退出日志查看。另一个终端或退出查看后读取状态：

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
.venv-gsplat153/bin/python tools/ru_part_v3_screen.py status v3
```

成功应为`TRAIN_COMPLETE / exit_code=0 / last_step=29999`且标准检查点可加载。失败时只发回状态和`tail -n 80 logs-puri/ru_part_v3_screening/v3.log`，不重跑、不删除目录。

4. 完整训练成功后，仅评测V3，已登记的历史Parent重评会直接复用：

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
.venv-gsplat153/bin/python tools/ru_part_v3_screen.py launch eval-v3
```

```bash
tail -n 40 -f /home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/ru_part_v3_screening/eval-v3.log
```

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
.venv-gsplat153/bin/python tools/ru_part_v3_screen.py status eval-v3
```

5. 看到`EVAL_COMPLETE / exit_code=0`后生成报告并停止：

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
.venv-gsplat153/bin/python tools/ru_part_v3_screen.py report
cat logs-puri/ru_part_v3_screening/RU_PART_V3_SCREENING_REPORT.md
```

报告直接使用`eval-existing-parent`指标及ROI数组，推理口径与历史训练耗时分开判断。缺少可比完整训练计时时不得整体晋级，不会因此自动增加一次Parent训练。

## 已完成留档：历史标准Parent固定checkpoint重评

在服务器终端使用现有`run_puri_gs.py --checkpoint`，直接读取`logs-puri/ru-generalization-rerun-9e292309/garden_ru_30k/ckpts/ckpt_29999_rank0.pt`。独立输出目录为`logs-puri/ru_part_v3_screening/eval-existing-parent`，旁边记录同名`.log`、`.pid`、`.exitcode`，不能覆盖已有结果。保持GPU 0、warmup=10，结果用于历史复现与未来同口径推理比较。

### 启动（服务器终端）

```bash
(
set -e
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
BASE=logs-puri/ru_part_v3_screening/eval-existing-parent
for path in "$BASE" "$BASE.log" "$BASE.pid" "$BASE.exitcode"
do
    if [ -e "$path" ]; then
        echo "Output already exists; stop: $path"
        exit 1
    fi
done
.venv-gsplat153/bin/python -c 'from tools.ru_part_v3_screen import require_complete; from puri_gs.v3_gpu import gpu_inventory, select_gpu; require_complete("smoke-parent", "SMOKE_COMPLETE"); require_complete("smoke-v3", "SMOKE_COMPLETE"); print(select_gpu(gpu_inventory(), pinned=0))'
nohup bash -c '
set +e
export CUDA_VISIBLE_DEVICES=0
.venv-gsplat153/bin/python run_puri_gs.py \
  --config configs/puri_gs_ru_part_v3_parent_garden30k.yaml \
  --gsplat-dir external/gsplat-v1.5.3-ru-part-v3 \
  --data-dir data/mipnerf360/360_v2/garden \
  --result-dir logs-puri/ru_part_v3_screening/eval-existing-parent \
  --gpu 0 \
  --checkpoint logs-puri/ru-generalization-rerun-9e292309/garden_ru_30k/ckpts/ckpt_29999_rank0.pt \
  --eval-warmup-renders 10
rc=$?
printf "%s\n" "$rc" > logs-puri/ru_part_v3_screening/eval-existing-parent.exitcode
exit "$rc"
' > "$BASE.log" 2>&1 < /dev/null &
printf "%s\n" "$!" > "$BASE.pid"
printf "Started evaluator PID=%s\n" "$!"
)
```

包装器先确认两个smoke最终完成及0号空闲；任何已有结果、阶段未完成或GPU占用都会停止启动。前台准备若报错就发回报错；没有出现`Started evaluator`不能当作已启动。

### 监控和验收（服务器终端）

```bash
tail -n 40 -f logs-puri/ru_part_v3_screening/eval-existing-parent.log
```

Ctrl+C仅退出日志查看。随后读取`.exitcode`：文件不存在时任务尚未记录完成，不能判定成功；非0时保留失败目录，发回最近80行日志，不直接覆盖重试。

```bash
cat logs-puri/ru_part_v3_screening/eval-existing-parent.exitcode
tail -n 80 logs-puri/ru_part_v3_screening/eval-existing-parent.log
```

只有exitcode=0后读取下面的摘要。这是质量复现检查，不是最终Parent复用批准，更不是新训练完成标记。成功应为24张视图名单相同、指标有限、PSNR差≤0.001dB、标准checkpoint加载、DINO/head关闭、single-raster ratio=1、warmup=10。最终数据/算法审计和复用登记仍待完成。

```bash
.venv-gsplat153/bin/python - <<'PY'
import csv
import json
import math
from pathlib import Path
r = Path("logs-puri/ru_part_v3_screening")
e = r / "eval-existing-parent"
assert int((r / "eval-existing-parent.exitcode").read_text()) == 0
old = json.loads(Path("logs-puri/ru-generalization-rerun-9e292309/garden_ru_30k/independent_eval/test_metrics.json").read_text())
new = json.loads((e / "test_metrics.json").read_text())
validation = json.loads((e / "ru_validation.json").read_text())
efficiency = json.loads((e / "efficiency_metrics.json").read_text())
with (e / "per_image_metrics.csv").open() as f:
    rows = list(csv.DictReader(f))
identity = json.loads((r / "smoke-parent/v3_input_manifest.json").read_text())
names_match = len(rows) == 24 and sorted(x["image_name"] for x in rows) == sorted(identity["test_basenames"])
finite = all(math.isfinite(float(x[k])) for x in rows for k in ("psnr", "ssim", "lpips"))
delta = abs(new["psnr"] - old["psnr"])
summary_matches = len(rows) == 24 and all(abs(sum(float(x[k]) for x in rows)/24 - new[k]) <= 1e-5 for k in ("psnr", "ssim", "lpips"))
print("view_count:", len(rows), "test_names_match:", names_match)
print("finite:", finite, "summary_matches_csv:", summary_matches)
print("old_PSNR:", old["psnr"], "new_PSNR:", new["psnr"], "abs_difference:", delta)
print("PSNR_reproduction_pass:", math.isfinite(delta) and delta <= .001)
print("new_metrics:", new)
print("DSC07988:", next((x for x in rows if x["image_name"] == "DSC07988.JPG"), None))
print("validation:", validation)
print("efficiency:", efficiency)
print("ROI_arrays_exist:", (e / "DSC07988_alpha.npy").is_file(), (e / "DSC07988_abs_error.npy").is_file())
PY
```

## 已完成：逐步开销修复后的单次V3复测

1. **图形界面：**主项目保持`ru-part`，提交本次代码、测试与说明，建议备注：**消除V3重复Q计算并异步传输当前视图缓存**。本地上传、服务器下拉，确认相同commit。未改trainer补丁、Mask/head参数和训练算法；原Parent smoke无需重跑。同步失败就停止并发回报错。
2. **服务器终端：**仅归档已完成但超限的原V3 smoke，包含checkpoint、原日志/状态和旧预检快照。脚本拒绝活动任务、正式训练目录以及已优化后的重复尝试。看到`ARCHIVED`才继续；报错即停止。

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
.venv-gsplat153/bin/python - <<'PY'
import json
import math
import shutil
from pathlib import Path

base = Path("logs-puri/ru_part_v3_screening").resolve()
def read(path):
    return json.loads(path.read_text())
for path in base.glob("*.status.json"):
    assert read(path)["status"] not in ("STARTING", "RUNNING"), path
assert not (base / "parent").exists() and not (base / "v3").exists()
run = base / "smoke-v3"
state = read(base / "smoke-v3.status.json")
assert state["status"] == "SMOKE_COMPLETE" and state["exit_code"] == 0
assert state["last_step"] == 599 and state["gpu"] == 0
checks = read(run / "v3_training_checks.json")
assert "implementation_revision" not in checks, "Optimized smoke already ran; stop."
p = read(base / "smoke-parent/v3_training_checks.json")["profile_median_ms"]
assert p > 0
ratio = checks["profile_median_ms"] / p
assert math.isfinite(ratio) and ratio > 1.08, ratio
assert (run / "ckpts/ckpt_599_rank0.pt").is_file()
paths = [run, base / "smoke-v3.log", base / "smoke-v3.status.json"]
assert all(path.exists() and not path.is_symlink() and path.resolve().parent == base for path in paths)
archive = base / "measurements/smoke-v3-before-transfer-fix"
assert archive.resolve().is_relative_to(base)
archive.mkdir(parents=True, exist_ok=False)
for name in ("environment_resolved.json", "preflight.json"):
    shutil.copy2(base / name, archive / name)
for path in paths:
    path.rename(archive / path.name)
print("ARCHIVED", archive)
PY
```

3. **服务器终端：**重新预检登记当前commit和GPU 0，预期38项预检测试通过并显示`PREFLIGHT_READY`。0号忙时等待空闲；不更换本轮设备。

```bash
.venv-gsplat153/bin/python tools/ru_part_v3_screen.py preflight --gpu 0
```

4. **服务器终端：**仅READY后执行下列单次V3 smoke，从step0开始、保留30k调度、到599停止。Ctrl+C只退出日志查看。

```bash
.venv-gsplat153/bin/python tools/ru_part_v3_screen.py launch smoke-v3
tail -n 40 -f logs-puri/ru_part_v3_screening/smoke-v3.log
```

5. **服务器终端：**检查状态与最近80行日志。应为`SMOKE_COMPLETE / exit_code=0 / last_step=599`且checkpoint可加载。把状态与下面的短摘要发回；失败发回状态和最近80行日志。若仍超限，停止，不再次归档重跑或修改门槛。

```bash
.venv-gsplat153/bin/python tools/ru_part_v3_screen.py status smoke-v3
tail -n 80 logs-puri/ru_part_v3_screening/smoke-v3.log
.venv-gsplat153/bin/python - <<'PY'
import json
from pathlib import Path
r = Path("logs-puri/ru_part_v3_screening")
p = json.loads((r / "smoke-parent/v3_training_checks.json").read_text())
v = json.loads((r / "smoke-v3/v3_training_checks.json").read_text())
print("revision:", v.get("implementation_revision"))
print("Parent_ms:", p["profile_median_ms"], "V3_ms:", v["profile_median_ms"])
print("ratio:", v["profile_median_ms"] / p["profile_median_ms"])
print("activation:", v["activation_status"], "active_samples:", v["active_samples"])
print("backward/raster:", v["actual_gaussian_backward_count"], v["actual_training_rasterization_count"])
print("disabled:", v["disabled_call_counts"])
print("gradients:", v["parameter_gradient_checks"])
PY
```

成功的实现复测目标为ratio≤1.08、计数600/1200、禁用项全0、梯度检查通过。仍显示`NO_Q_SAMPLED`时按附件保留固定张量检查，不调整Mask阈值、静态证据或额外训练前缀。历史标准Parent训练GPU为1、评测GPU为6，本轮GPU为0；优先继续质量复用登记和重评，同口径成本未接受，不为此自动追加完整Parent。

服务器后续反馈已确认：预检、trainer帮助和14项V3专项测试通过；两端commit仍以preflight.json和图形界面为准。用户指定GPU优先级为 **6→7→0→1→2→3→4→5**。新增GPU选择功能属于后续代码变更，本地含GPU选择的34项相关测试已通过；下次同步可与Parent审计后的必要适配合并，避免仅为读取报告重复Git操作。提交备注建议：**按指定GPU优先级选择空闲设备并固定筛选对照设备**。主项目仍保持ru-part；gsplat子目录的detached HEAD是固定第三方版本的正常状态。

**本次运行覆盖：用户随后指定优先使用GPU 0。** 本轮使用下面的 `--gpu 0` 固定设备；后续任务仍保留上述通用优先级。本参数在服务器现有`e85dbd4`版本中已经支持，不需要为本次改选0号重新Git同步。用户提供的2026-09-09 12:25快照中，0号71MiB、利用率0%，6/7号有VLLM计算进程；这只是当时状态，启动前须再次查看`nvidia-smi -i 0`。

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
nvidia-smi -i 0
.venv-gsplat153/bin/python tools/ru_part_v3_screen.py preflight --gpu 0
```

当前服务器旧版本的显式选卡不自动检查占用；若0号出现新的计算进程，先不要启动smoke。预检成功后按第4步启动smoke-parent，GPU参数已保存到`environment_resolved.json`，启动命令无需再加GPU参数。已有静态缓存已通过预检，可跳过第3步构建。600步smoke、后续正式训练及独立评测保持同一设备。

## 已处理的启动故障：V3缓存参数被旧RU-PART检查误拒绝

用户记录：`smoke-v3`包装PID 2289957、child PID 2290031，`FAILED / exit_code=1 / last_step=-1`，报错`RU-PART-only arguments were supplied to another profile`。V3配置保留`profile=ru`，启动器却将`--track-cache`一律当作旧PART专属参数。修复只允许`v3_screening=v3`通过该缓存参数检查；旧PART replay/diagnostic选项继续拒绝，标准Parent继续不加载静态支持。没有修改trainer补丁、损失或训练调度，已完成的Parent smoke保留。

本地47项相关测试通过，其中新增13项从阶段包装器真实参数进入launcher的回归检查，覆盖Parent/V3短测与完整训练命令、旧参数拒绝、Parent禁用缓存和V3缺缓存失败。测试使用占位资产并隔离外部环境检查，不冒充服务器CUDA验收。

1. **本地/服务器Git图形界面：**在`ru-part`提交本次代码、测试与说明（包含此前未同步的GPU选择文件），备注建议：**修复V3缓存参数校验并完善GPU选择与回归检查**。本地上传，服务器下拉；确认两端相同commit。不要提交运行日志或缓存。图形同步失败时停止并发回报错。
2. **服务器终端：**同步后执行下面的归档。仅接受本次尚未创建训练目录的启动校验失败，保留失败日志、状态和旧预检快照；不删除或移动成功的Parent结果。看到`ARCHIVED`后继续；任何断言失败就停止并发回输出。

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
.venv-gsplat153/bin/python - <<'PY'
import json
import shutil
from pathlib import Path

base = Path("logs-puri/ru_part_v3_screening").resolve()
stage = "smoke-v3"
status = base / f"{stage}.status.json"
log = base / f"{stage}.log"
state = json.loads(status.read_text())
assert state["stage"] == stage and state["status"] == "FAILED", state
assert state["exit_code"] == 1 and state["last_step"] == -1, state
assert "RU-PART-only arguments were supplied to another profile" in log.read_text()
assert not (base / stage).exists(), "Training directory exists; stop and inspect."
archive = base / "failed_attempts" / f"smoke-v3-launcher-{state['pid']}"
archive.mkdir(parents=True, exist_ok=False)
for name in ("environment_resolved.json", "preflight.json"):
    shutil.copy2(base / name, archive / name)
for path in (log, status):
    path.rename(archive / path.name)
print("ARCHIVED", archive)
PY
```

3. **服务器终端：**重新预检以登记修复后的commit，并继续固定GPU 0。预期34项预检测试通过（V3及GPU测试）和`PREFLIGHT_READY`。这一步不训练；原Parent记录仍在。0号忙时先等待0号空闲，不更换本轮对照设备。

```bash
.venv-gsplat153/bin/python tools/ru_part_v3_screen.py preflight --gpu 0
```

4. **服务器终端：**仅在READY后，从step0重新启动V3 smoke，查看后台日志。按Ctrl+C仅退出日志查看。

```bash
.venv-gsplat153/bin/python tools/ru_part_v3_screen.py launch smoke-v3
tail -n 40 -f logs-puri/ru_part_v3_screening/smoke-v3.log
```

5. **服务器终端：**检查`SMOKE_COMPLETE / exit_code=0 / last_step=599`及checkpoint可加载。发回状态与最近80行日志；失败即停止，不重复覆盖。完成后按第5步收集两组`v3_training_checks.json`与V3证据检查，以核对Q激活和配对耗时。历史完整Parent可比性审计仍待完成，不跳到30k Parent。

```bash
.venv-gsplat153/bin/python tools/ru_part_v3_screen.py status smoke-v3
tail -n 80 logs-puri/ru_part_v3_screening/smoke-v3.log
```

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
.venv-gsplat153/bin/python tools/ru_part_v3_screen.py preflight --gpu auto
```

脚本核对分支、CUDA设备、gsplat版本、数据/PNG、DINO、feature cache，查找已有静态缓存，并列出历史Parent候选。它在专用`external/gsplat-v1.5.3-ru-part-v3`里准备原版本Python补丁，不动已有RU/PART例程；运行CPU测试和真正的trainer命令帮助。

`--gpu auto`（也为默认值）按6→7→0→1→2→3→4→5选择首张空闲设备。此处空闲指无compute进程、显存占用≤512MiB、利用率≤5%；只在预检/启动时检查，不做持续轮询。仍支持`--gpu 6`显式固定。第一次smoke启动后，本轮保持同一GPU，重新预检也不自动换卡；忙时不抢占其他进程、不偷偷切卡。空闲检查不是服务器资源锁，其他用户仍可能随后启动任务。

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
nvidia-smi
```

- PID消失但无完成标志为未完成，不能当成功。已存在输出会被拒绝覆盖；异常后先发回状态和最近80行日志，不自行删除结果目录或重复运行。
