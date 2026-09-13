# P04 复跑说明（仅供审核，不要求用户现在执行）

本包结果已完成；不需要重跑。若将来要独立复核，必须在同一 `ru-part`、冻结 L20 环境和原 P01 登记的数据/模型/特征路径下进行，不切分支、不下载或训练。三个脚本分别是 `tools/p04_oac_diagnostic.py`、`tools/p04_postcheck.py`、`tools/p04_analyze.py`。它们不调用任何 optimizer step 或增密拓扑；第一个脚本在每次 F/B 前向持久 CSV 预扣，单个新输出目录最多 300 次。

1. 先只读核验当前分支、HEAD、工作区与 GPU。正确标志：分支仍为 `ru-part`、无其它任务占 GPU 7。不要在原 `/home/chenglong/P04-work` 上执行新的 `prepare`，其 manifest 和累计 ledger 必须保留。
2. 必须明确需要全新复核且获得新预算时，在干净的同路径环境（原 P04 结果先独立归档，不覆盖）运行下述 `prepare`。正确标志：`diagnostic_manifest.json` 为 `PREPARED`，列两场景各 8 个已哈希 TRAIN 图，`call_budget_ledger.csv` 只有表头；这一步没有 F/B。
3. `run` 先产生四条件终点探针，再用 postcheck 检查 64×64 参考与成对开销。正确标志：状态依次为 `DIAGNOSTIC_COMPLETE`、`POSTCHECK_COMPLETE`，预算扣账 ≤300，state_integrity 中全为 unchanged。任何 `FAILED_PARTIAL` 停止，不清空 ledger；人工查明且在剩余预算内才能决定是否继续。

本次实际使用的服务器命令形式（仅供审核；当前目录已有结果，不能重跑 `prepare`）：

```bash
cd /home/chenglong/Uncertain-Nerf/uncertain-nerf
export PYTHONPATH=/home/chenglong/Uncertain-Nerf/uncertain-nerf
export CUDA_VISIBLE_DEVICES=7
PY=.venv-gsplat153/bin/python
$PY /home/chenglong/P04-work/p04_oac_diagnostic.py prepare --views 8 --out /home/chenglong/P04-work
$PY /home/chenglong/P04-work/p04_oac_diagnostic.py run --out /home/chenglong/P04-work
$PY /home/chenglong/P04-work/p04_postcheck.py
$PY /home/chenglong/P04-work/p04_analyze.py
```

本次实际脚本先被复制到 `/home/chenglong/P04-work/`，再以已安装 gsplat 1.5.3 包运行。`--resume-charged` 只用于首次 probe 前已定位技术失败的人工重试，不可重置预算或改变选图/公式。只读查看已完成账本：

```bash
cat /home/chenglong/P04-work/status.json
tail -n 5 /home/chenglong/P04-work/call_budget_ledger.csv
```

本包无新增训练身份；无需 Git 分支变更。此次实施的 Git 同步记录见交付报告；后续是否进入 P05 由方案助手决定，不能根据本文件自行开训。
