# P04-T 包末决策交接

建议状态：`INCONCLUSIVE_STOP`。取证执行完整有效，两个真实100步窗与原策略 grad/count 对齐，20k身份受控停止，模型/快照在服务器。OAC相对 `a·u` 有非统一排序差，但更接近原 G0；固定 D 确实碰到68/66个探针，却未碰到任何 `A<1` 低证据对象。OAC与简单控制都无受污染对象进入 top-K，因而不具备任务书要求的相对风险证据。不能把此次缺口当成 OAC 已被实验证伪，也不能按其排序差放行 P05。

请方案助手只据本包报告、逐探针/逐步缓存和总账决定研究方向。按任务书，本次是当前 OAC 唯一最后补充，不再自动重选窗口/ROI、调整 κ/帽/支持门、补 Room/seed、续到30k、启动 P05 或开启 R。原 GS-RU、P01—P03、P04 `COMPLETE_LIMITED` 均保持。

需要审阅的关键证据：`REPORT_P04T.md` §3—4、`analysis.json`、`interpretation.json`、`risk_attribution.csv`、两个 `W*_per_step_probe.npz` 及其 manifest、`postcheck.json`、`state_manifest.csv`、`call_budget_ledger.csv`。三份状态仅服务器保留，不在 ZIP。
