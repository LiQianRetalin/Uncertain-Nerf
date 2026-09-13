# P03 Corner 成本边界更正（不重跑）

本附录只用服务器 `/home/chenglong/P03-work/state/gpu_stage_events.csv`、对应报告目录里的特征 manifest 和原 `method_cost_summary.csv`。原成本表保留不覆盖。全部阶段事件累计 4,976 秒（1.3822 GPU·小时）；其中各方法全尝试消耗 RU 761 秒、RobustSplat 1,239 秒、SLS 2,934 秒，另有未分配给方法的共同 preflight 42 秒。全尝试数字包括失败、重复准备、smoke、正式进程以及后处理/评测，不能解释为“一次成功构建”。

| 方法 | 原表“完整构建” | 全尝试（不含共同 42 秒） | 与正式模型匹配的最后一次成功特征阶段 | 正式进程墙钟 | 统一必要构建 |
|---|---:|---:|---:|---:|---|
| RU | 706 s | 761 s | 8 s（阶段内实际提取 4.001 s，非另加） | 673 s | UNKNOWN |
| RobustSplat | 1129 s | 1239 s | 外部准备 0 s；训练内特征耗时 UNKNOWN | 1129 s | UNKNOWN |
| SLS-mlp-no-UBP | 2779 s | 2934 s | 214 s（manifest 增量 211.908 s，非另加） | 1988 s | UNKNOWN |

阶段证据：RU 四次特征阶段为 8+8+9+8=33 秒；SLS 为 147（失败）+215+215+214=791 秒。最后一次准备紧邻正式 smoke 与训练，且对应最终有效特征目录；不从多次准备中选择最快一次。RobustSplat 没有单列特征准备阶段，不能据此断言整个模型不产生内部特征成本。

正式进程墙钟 RU/Robust/SLS 分别为 673/1129/1988 秒。SLS 正式进程包含原生评测、轨迹/GIF 等附带步骤；RU 命令禁用视频且取消中途评测；Robust 也有自己的内部评测/特征边界。现有事件只记录整进程，不能可靠拆出这些步骤的秒数，故“正式进程内可分离附带秒数”和可直接横比的“统一必要构建成本”均标 `UNKNOWN`。不把 train-step 内部计时当完整成本，也不为补齐边界重新生成特征、训练或测速。原质量和统一 FPS 结论不因本项修订而改变。

机器可读表为 `p03_cost_accounting_corrected.csv`；其中 `matched_success_feature_internal_seconds_nonadditive` 是阶段内部 manifest 计时，不得与阶段墙钟相加。项目总消耗计算：761+1239+2934+42=4976 秒。
