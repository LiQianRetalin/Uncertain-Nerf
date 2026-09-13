# P04-T 代码与命令差异

原始服务器 gsplat RU 训练器 SHA-256：`dea00d8a11071789d4afb01c01d5969683b53fc96db20d5582a6c5278bec328c`。P04-T 隔离训练器 `/home/chenglong/P04T-work/examples/p04t_simple_trainer.py` SHA-256：`921c89f59fadc398ce3ccb8baa8cd2c85edfdadd52aa0a443edb7413750f0531`。由仓库内 `tools/p04t_patch_trainer.py` 对固定 SHA 源文件的**三个唯一锚点**生成：

1. 仅当 `P04T_RUN_ID` 存在，保留4 worker 原随机无放回排列但接入 `ReplayableRandomSampler`，记下排列/已送达偏移以便快照恢复；三轮相机顺序与原 `shuffle=True` 经独立验证一致。正常非 P04-T 路径原样。
2. 在原 photometric `loss.backward()` 后、头监督和原策略回调前，调用只读 A 探针及8次克隆 D 影子；不替换原 loss、grad、Mask 或优化器。原 `_grow_gs` 仅被当前实例包装为事件前只读观察，先调用原 `_update_state`，后续原 grow/prune/clear 顺序不变。
3. 在每次原优化器/调度器操作后，记录预算/进度；10999、18999、19999 保存状态，完成19999后 `break`。训练 `max_steps=30000` 未改。`P04T_CONTROLLED_STOP_20000_UPDATES` 为成功停止标志。

原 gsplat checkout、安装包、`puri_gs/delayed_absgrad.py`、旧 P04 代码与数据均未修改。隔离源码生成器仅作用 `/home/chenglong/P04T-work/examples`。本包仓库新增 `tools/p04t_{monitor,patch_trainer,preflight,runtime_preflight,launch,postcheck,analyze,interpret,cost}.py`；其中后检/分析仅读缓存，成本脚本最多13次终点调用。训练实际用的固定脚本与运行参数在 `run_plan.csv`、`frozen_command.json` 和服务器输出身份 `cfg.yml` 中，可据 SHA 核对；不以本地改动补写旧训练身份。

服务器后台进度只读命令（PowerShell）：

```powershell
ssh -i "C:\Users\liqian\.ssh\id_ed25519_p03" chenglong@172.16.55.2 "cat /home/chenglong/P04T-work/status.json"
```

成功标志已记录：服务器训练日志含 `P04T_CONTROLLED_STOP_20000_UPDATES`、`P04T_TRAINING_COMPLETE_STOP`；独立后检打印 `P04T_POSTCHECK_PASS`，纯缓存分析打印 `P04T_CACHE_ANALYSIS_PASS`。训练现已停止，不需要用户再运行该命令。Git 操作仅在现有 `ru-part` 上同步本包小型代码/报告，不含 `states/*.pt`、模型、数据、特征。
