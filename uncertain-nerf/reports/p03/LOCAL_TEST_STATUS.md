# P03 本地测试状态

日期：2026-09-12。

- P03 Corner、Patio 回归及共同 COLMAP 转换定向测试：`9 passed`。
- `puri_gs`、`tools`、`tests` 与 `run_puri_gs.py` 全量 `compileall`：通过。
- P03 服务器脚本 `bash -n`：通过。
- 四份运行时增量 patch 对各自当前审计源码执行反向 `git apply --check --ignore-whitespace`：通过。
- 其余可收集套件：`399 passed, 2 skipped, 2 failed`。一个失败仍由当前本地 gsplat 环境缺少既有 `PyYAML` 引起；另一个是未修改的 `fit_static_transient_gate.py --dry-run` 既有退出码断言。完整无排除收集另有 3 个既有测试因同一 `PyYAML` 缺失而中止。P03 定向测试没有失败，未为这些历史测试安装或升级环境依赖。

正式 P03 服务器预检仍会独立核验三个已登记环境、CUDA、loader、冻结特征和实际运行提交；本地 GPU 不用于正式结果。
