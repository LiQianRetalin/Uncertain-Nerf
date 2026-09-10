# P01 后续决策（交给方案助手）

1. 是否确认 Android 以 `9e292309` 的 30k B1/RU 配对作为 P02 唯一内部锚点，并把 10k B1 与原 Phase-R Android 行仅保留为历史参考？
2. 是否确认共同输入口径为 Android 与 Patio-High 两套经回读验证的物理 factor4/loader factor1 PINHOLE COLMAP；作者重跑 SfM 的 Reported 数字只作参考？
3. 是否按 `run_plan_P02.csv` 开启四个单 seed42 完整训练身份：RobustSplat 与 SLS-mlp（不启用 UBP）各跑 Android、Patio-High？

未得到下一包明确决定前，不自动安装外部方法、不训练、不启动 OAC。
