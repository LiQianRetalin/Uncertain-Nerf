# V3 单表面诊断操作清单

本轮协议为 `V3_SINGLE_SURFACE_DIAGNOSTIC_V1`。本地实现和 CPU 自检已完成；服务器上的真实 DeltaQ、CUDA 两次更新预检和 Va/Vb/O 尚未执行。**本次先完成步骤 1–4，拿到真实数值和边界图后再确认标签。**

固定 A=`DSC07987.JPG`，仅干预房屋上部三角山墙的砖面；B=`DSC07989.JPG` 的 S 全零，但每组仍参加 200 次优化；H 候选为 `DSC07986.JPG`，仅评价。已按相机到 A 的距离锁定 8 张候选，第一张就清楚显示同一山墙，因此实际只检查了第一张。原图坐标为 factor4 的 1297×840。

| 角色 | 当前未确认的候选区域 | 像素数 |
| --- | --- | ---: |
| A：训练干预 S | (692,78)、(614,117)、(773,112) | 2,946 |
| B：上下文 S | 全零 | 0 |
| H：评价 R | (814,37)、(727,79)、(902,72) | 3,413 |

这些坐标已经由 Codex 生成，用户无需编写 JSON。红色边界位于砖面内部，避开天空、屋檐边条和前景枝叶；尚未把候选当作已确认的静态标签。

## 1. Windows：一次图形化 Git 同步

本地代码目录：`E:\6-Project\1-UncertainNerf\uncertain-nerf`，Git 仓库根目录是上一级。保持 `ru-part` 分支，用现有 Git 图形界面检查、提交并上传以下本轮变更，然后在服务器图形界面下拉同一提交。

- 新增：`configs/ru_part_v3_single_surface.json`、`puri_gs/single_surface.py`、`tools/ru_part_v3_single_surface.py`、`tests/test_single_surface.py`。
- 复用修订：`puri_gs/recoverability_runtime.py`、`tools/ru_part_v3_recoverability.py`。
- 本轮文档：本清单和 `reports/ru_part_v3/SINGLE_SURFACE_IMPLEMENTATION_AUDIT.md`。

提交备注：**修订单静态表面诊断：保留上下文视图并区分检查图保护与增益**。

工作区原有的四视图复核文档修改已保留，按现有工作记录在图形界面核对，不覆盖或撤销它们。ROI、图片、数组、检查点和日志位于数据目录，不加入 Git。后续标签确认和预注册不需要再次提交。

本地候选材料：`E:\7-DataSet\ru_part_v3\single_surface_diag\house_upper_v1`。这里的 `local_roi_proposal_review.png` 是视觉提案，不能代替服务器真实 DeltaQ 统计。

## 2. Linux：核对新入口与环境

前置条件：服务器图形界面已下拉步骤 1 的提交；继续使用原有 `.venv-gsplat153`，无需安装或升级依赖。

在服务器终端逐条执行：

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
.venv-gsplat153/bin/python tools/ru_part_v3_single_surface.py --help
.venv-gsplat153/bin/python tools/ru_part_v3_single_surface.py check-config
.venv-gsplat153/bin/python -m pytest tests/test_single_surface.py tests/test_coverage_recoverability.py tests/test_ru_part_v3.py -q
```

预期：帮助中有 `launch`、`status`、`confirm-roi`、`revise-roi`；配置显示新协议、固定终态学习率、每组 400、评价 `[0,400]`；测试通过。这些测试使用 CPU 数值和小型可微替身，不是 CUDA 实证。若失败，发回报错全文，停在这一步，不自行安装包或启动优化。

后台准备还会核对 `ru-part`、旧准备文件身份、原检查点、原 renderer/Adam 实现、Torch/CUDA/包版本和学习率。服务器的实时文件与 GPU 空闲状态只能在该步骤及后续 worker 实际运行时验证。

## 3. Linux：GPU 0 后台有界准备

前置条件：步骤 2 通过，GPU 0 空闲。下面的新入口复用旧 `garden_v3_final_diag_v1/prepared` 中 A/B 的 M/C 和 A 的 float source RGB；不重跑旧 161 图 prepare，不重新训练或估计 Mask。

输出为独立目录：

`/home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/ru_part_v3_single_surface_diag/house_upper_v1`

完整命令如下，整行复制即可。编码内容就是上表两个多边形和已核对的候选顺序，不含标签确认。

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
.venv-gsplat153/bin/python tools/ru_part_v3_single_surface.py launch prepare --gpu 0 --proposal-base64 eyJwcm90b2NvbCI6IlYzX1NJTkdMRV9TVVJGQUNFX0RJQUdOT1NUSUNfVjEiLCJjYW5kaWRhdGVfc2hhMjU2IjoiOTQ0MmE3YzJiZDhiZjYwNGIxYjI4NWNkNTQ0YzE3ZTgxMzdmNDc2ZmQ4MjE4OTIyMTA5MTUzOTU0NDk1YWY5NiIsImNoZWNrX3ZpZXciOiJEU0MwNzk4Ni5KUEciLCJjYW5kaWRhdGVzX3Jldmlld2VkIjoxLCJlYXJsaWVyX2NhbmRpZGF0ZV9leGNsdXNpb25zIjp7fSwic3VyZmFjZV9kZXNjcmlwdGlvbiI6IlVwcGVyIGludGVyaW9yIGJyaWNrIGZhY2Ugb2YgdGhlIHNhbWUgdHJpYW5ndWxhciBob3VzZSBnYWJsZTsgc2t5LCByb29mIHRyaW0gYW5kIGZvbGlhZ2UgZXhjbHVkZWQuIiwiY29ycmVzcG9uZGVuY2VfYmFzaXMiOiJGaXJzdCBnZW9tZXRyaWMgY2FuZGlkYXRlIHNob3dzIHRoZSBzYW1lIGdhYmxlIGFwZXgsIHJvb2Ygc2xvcGVzIGFuZCBicmljayBwYXR0ZXJuLiBBIGFuZCBIIHBvbHlnb25zIGFyZSBzZXBhcmF0ZWx5IHBsYWNlZCBpbiBvcmlnaW5hbCAxMjk3eDg0MCB0cmFpbmluZyBwaXhlbHMuIiwicG9seWdvbnMiOnsiRFNDMDc5ODcuSlBHIjpbW1s2OTIsNzhdLFs2MTQsMTE3XSxbNzczLDExMl1dXSwiRFNDMDc5ODYuSlBHIjpbW1s4MTQsMzddLFs3MjcsNzldLFs5MDIsNzJdXV19LCJsYWJlbF9zdGF0dXMiOiJVTkNPTkZJUk1FRF9QUk9QT1NBTCJ9
```

同一命令也已存为本地 `E:\7-DataSet\ru_part_v3\single_surface_diag\house_upper_v1\01_prepare_linux.sh`；使用上方复制命令即可，不必额外传脚本。

启动后会打印真实 PID、GPU 编号、UUID 和进程查询命令。立即返回只代表后台进程已启动。查看状态与日志：

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
.venv-gsplat153/bin/python tools/ru_part_v3_single_surface.py status
tail -n 80 logs-puri/ru_part_v3_single_surface_diag/house_upper_v1/prepare.log
tail -n 40 -f logs-puri/ru_part_v3_single_surface_diag/house_upper_v1/prepare.log
```

`Ctrl+C` 只退出持续日志查看。预期最终状态 `ROI_REVIEW_REQUIRED`、`exit_code=0`，实际准备的 Gaussian 更新/渲染均为 0。产物包括 `candidate_views.json`、`preparation_reuse.json`、`roi_manifest.json`、`intervention_qualification.json`、三张原图及边界图、`status.json` 和 `review_bundle.zip`。

若 `BLOCKED / NO_ORACLE_INTERVENTION`，表示候选的真实新增权重或加权残差不满足正值条件，按协议停止，不扩大区域、改变系数或启动三组。其他失败发回上述状态输出和最近 80 行 `prepare.log`。GPU 忙且尚未建立目录时等待空闲；目录已存在时保留现场，不删除目录、不自动重试。

## 4. Windows：回传真实数值和边界图

前置条件：准备进程已经结束。先在 Linux 查看简短数值：

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
cat logs-puri/ru_part_v3_single_surface_diag/house_upper_v1/intervention_qualification.json
cat logs-puri/ru_part_v3_single_surface_diag/house_upper_v1/status.json
```

用现有文件传输界面下载服务器的：

`/home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/ru_part_v3_single_surface_diag/house_upper_v1/review_bundle.zip`

保存到 Windows：`E:\7-DataSet\ru_part_v3\single_surface_diag\house_upper_v1\review_bundle.zip`，然后将这个新包发回当前任务。它包含本轮最终准备状态，不要再次发送旧四视图包。

Codex 将核对实际 DeltaQ 非零像素、占 S 比例、权重和、全图均值、S 内均值和 D_A，并展示 A/H 标注原图及 B 上下文图，再一次性请你确认或纠正静态性和对应关系。这是在补齐具体标签信息，不是重新申请已授权的短诊断运行许可。

**当前执行到这里。下面步骤仅在本轮实际面板和数值回读、并明确确认相同 A/H 区域后执行。** 同一表面的边界修正由 Codex 生成新命令和新提案，不需要你手写坐标；旧提案和包会保留。预注册锁定后不再改区域。

## 5. Linux：记录已完成的标签确认

前置条件：步骤 4 已确认当前面板中的同一静态山墙砖面，真实 `sum(DeltaQ)>0` 且 `D_A>0`。此时执行下列完整命令。它读取当前面板的实际 SHA，无需手工填写散列值；确认记录绑定当前 ROI 数组、原图身份、坐标和面板。

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
.venv-gsplat153/bin/python - <<'PY'
import json
import subprocess
import sys
from pathlib import Path

root = Path('logs-puri/ru_part_v3_single_surface_diag/house_upper_v1')
roi = json.loads((root / 'roi_manifest.json').read_text())
subprocess.run([
    sys.executable, 'tools/ru_part_v3_single_surface.py', 'confirm-roi',
    '--panel-sha256', roi['review_panel_sha256'],
    '--statement', '我已核对本轮A和H的标注原图，确认当前多边形对应同一块静态山墙砖面。',
], check=True)
PY
```

成功标志：`ROI_CONFIRMED`，生成 `roi_confirmation.json`，活动 `roi_manifest.json` 内含确认记录。失败发回完整报错及 `status` 输出。没有确认或数值资格时不执行这一步。

## 6. Linux：GPU 0 串行运行两次预检和三组各 400 步

前置条件：步骤 5 成功，代码、环境、源输入和 GPU UUID 与准备时一致，GPU 0 空闲。

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
.venv-gsplat153/bin/python tools/ru_part_v3_single_surface.py launch run --gpu 0
```

后台顺序固定：新载入 Va 临时副本 1 次更新 → 丢弃 → 新载入 O 临时副本 1 次更新 → 丢弃 → 锁定 `single_surface_preregistration.json` 和 SHA → Va 400 → Vb 400 → O 400 → 评价/报告/停止。

预检复用两次实际 backward 保留的 RGB 梯度核对恢复项差异，不新增 VJP、Jacobian、渲染探针或第二套 smoke。正式三组各自重新载入原 step29999 Gaussian 和新建零状态 Adam。训练序列均为 `[A,B]×200`，H 不进入训练输入。每组只在 0/400 评价 A/B/H；正式计数为 1,200 次训练 rasterization、1,200 次 backward、18 次评价，临时预检另计 2 次。

每 50 次更新输出简短进度。查询当前阶段、当前组、更新数、实际调用计数和退出码：

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
.venv-gsplat153/bin/python tools/ru_part_v3_single_surface.py status
tail -n 80 logs-puri/ru_part_v3_single_surface_diag/house_upper_v1/run.log
tail -n 40 -f logs-puri/ru_part_v3_single_surface_diag/house_upper_v1/run.log
```

若要独立查询实际后台 PID，无需猜 PID，可在准备或运行阶段使用：

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
.venv-gsplat153/bin/python - <<'PY'
import json
import subprocess
from pathlib import Path

root = Path('logs-puri/ru_part_v3_single_surface_diag/house_upper_v1')
for phase in ('prepare', 'run'):
    path = root / (phase + '.status.json')
    if not path.exists():
        continue
    state = json.loads(path.read_text())
    print(phase, {k: state.get(k) for k in ('status', 'pid', 'current_stage', 'current_group', 'updates_completed', 'exit_code')}, flush=True)
    if state.get('pid'):
        subprocess.run(['ps', '-p', str(state['pid']), '-o', 'pid,etime,stat,args'])
PY
```

源身份、NaN/Inf、Gaussian 数量、设置或文件不一致均立即停止后续组。异常记录 `DIAGNOSTIC_INVALID`，保留准确的已完成更新数；不续跑、不重试训练。失败时发回 `status` 输出及最近 80 行 `run.log`。若仅导出包失败，状态明确为 `READBACK_BUNDLE_FAILED`，不要因此重跑优化。

## 7. Linux 完成核对与 Windows 结果回读

前置条件：运行后台已经结束。执行：

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
.venv-gsplat153/bin/python tools/ru_part_v3_single_surface.py status
cat logs-puri/ru_part_v3_single_surface_diag/house_upper_v1/status.json
cat logs-puri/ru_part_v3_single_surface_diag/house_upper_v1/report.md
```

成功执行必须同时满足 `DIAGNOSTIC_COMPLETE`、进程退出码 0、各组 `COMPLETE`/退出码 0/更新 400/末步 399、Gaussian 始终 2,266,597、head/拓扑事件为 0、终态参数可加载、必需评价有效。PID 消失不代表成功。`diagnostic_splats.pt` 仅含六组 Gaussian 和诊断元数据，不保存 Adam。

下载服务器的：

`/home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/ru_part_v3_single_surface_diag/house_upper_v1/result_bundle.zip`

保存到 `E:\7-DataSet\ru_part_v3\single_surface_diag\house_upper_v1\result_bundle.zip` 并发回当前任务。包内含最终 `status.json`、预注册、确认、报告、真实数值、各组状态/进度和少量评价图；不包含大检查点。

报告会逐项给出 A 局部收益、六个保护域、H 额外收益及初态/控制波动。B 的目标 ROI 指标是 null；H 初态良好且未进一步改善不会自动否定 A 的受控局部响应。相对初态的保护预算 `d=max(0.01*E0,1e-4)` 不随控制波动增大。

正常完成可以得到 `PROTECTION_NOT_ESTABLISHED`、`LOCAL_RESPONSE_WITH_CHECK_VIEW_GAIN`、`LOCAL_RESPONSE_WITH_PROTECTION` 或 `NO_CLEAR_LOCAL_RESPONSE`。完成不等于正向结果。历史 V3 的 `QUALITY_RECOVERY_FAIL / NO_GO`、旧四图的 `ROI_NOT_READY` 和重放的 `REPLAY_NOT_EQUIVALENT` 保留；本轮不重评正式质量门，不追加实验。
