# RU-PART 机制诊断操作手册

仅用于 `ru-part`。诊断默认关闭；普通 Parent/noop/current 路径不变。训练 horizon 始终为 30000，短任务只用 post-update `stop_after_step`，禁止缩短 `max_steps`。

## 固定路径

```bash
ROOT=/home/chenglong/Uncertain-Nerf/uncertain-nerf
GSPLAT="$ROOT/external/gsplat-v1.5.3-ru-part-controls"
GARDEN="$ROOT/data/mipnerf360/360_v2/garden"
DINO_REPO="$ROOT/external/dinov2"
DINO_WEIGHT="$ROOT/data/PURI-GS-assets/dinov2/dinov2_vits14_reg4_pretrain.pth"
FEATURE_CACHE="$ROOT/data/PURI-GS-derived/semantic_features/garden"
PYTHON="$ROOT/.venv-gsplat153/bin/python"
BASE="$ROOT/logs-puri/ru_part_mechanism_diagnostic_20260908"
CONSOLE="$ROOT/logs-puri/ru_part_mechanism_diagnostic_20260908_console"
GPU=6
cd "$ROOT"
```

所有输出独占创建；目录已存在即停止。正式测试 RGB 只允许历史 `audit` 证据核验使用，后续命令只读训练图。

## B：检查与两个 smoke

```bash
bash scripts/prepare_puri_gs_ru_part.sh "$GSPLAT"
"$PYTHON" -m pytest tests/test_mechanism_diagnostic.py \
  tests/test_ru_part_integration.py tests/test_ru_part_replay.py -q
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" tools/ru_part_mechanism_diagnostic.py \
  smoke --device cuda --output-dir "$BASE/synthetic_restore_smoke"
```

完成依据：`PURI-GS-RU-PART-PATCH-READY`、测试全通过、`SMOKE_ACCEPTED`。

真实 trainer 100-update smoke（后台）：

```bash
mkdir -p "$CONSOLE"
nohup "$PYTHON" tools/ru_part_mechanism_diagnostic.py trainer-smoke \
  --gsplat-dir "$GSPLAT" --data-dir "$GARDEN" \
  --output-dir "$BASE/trainer_smoke" --gpu "$GPU" \
  --dino-repo-dir "$DINO_REPO" --dino-weight-path "$DINO_WEIGHT" \
  --feature-cache-dir "$FEATURE_CACHE" \
  >"$CONSOLE/trainer_smoke.log" 2>&1 &
echo $! >"$CONSOLE/trainer_smoke.pid"
```

最简进度检查：

```bash
tail -n 1 "$BASE/trainer_smoke/diagnostic_progress.jsonl" 2>/dev/null
grep -E 'Traceback|CUDA out of memory|RuntimeError' \
  "$CONSOLE/trainer_smoke.log" | tail -n 3
```

完成依据：进度行包含 `completed=100/status=complete`，异常 grep 无输出。

## C：唯一 U，step0–10199 不间断

```bash
nohup "$PYTHON" tools/ru_part_mechanism_diagnostic.py prefix \
  --gsplat-dir "$GSPLAT" --data-dir "$GARDEN" \
  --output-dir "$BASE/U" --gpu "$GPU" \
  --dino-repo-dir "$DINO_REPO" --dino-weight-path "$DINO_WEIGHT" \
  --feature-cache-dir "$FEATURE_CACHE" >"$CONSOLE/U.log" 2>&1 &
echo $! >"$CONSOLE/U.pid"
```

```bash
tail -n 1 "$BASE/U/diagnostic_progress.jsonl" 2>/dev/null
grep -E 'Traceback|CUDA out of memory|RuntimeError' "$CONSOLE/U.log" | tail -n 3
```

每 200 update 一行。完成依据：`completed=10200/status=complete`，并且：

```bash
test -s "$BASE/U/replay_ckpts/replay_step9999.pt"
test -s "$BASE/U/diagnostic_terminal_step10199.pt"
```

## D：R1/R2 各恢复 200 update

R1、R2 串行；都引用 U 的同一 step9999 文件。

```bash
for STAGE in R1 R2; do
  nohup "$PYTHON" tools/ru_part_mechanism_diagnostic.py replay \
    --stage "$STAGE" --checkpoint "$BASE/U/replay_ckpts/replay_step9999.pt" \
    --gsplat-dir "$GSPLAT" --data-dir "$GARDEN" \
    --output-dir "$BASE/$STAGE" --gpu "$GPU" \
    --dino-repo-dir "$DINO_REPO" --dino-weight-path "$DINO_WEIGHT" \
    --feature-cache-dir "$FEATURE_CACHE" >"$CONSOLE/$STAGE.log" 2>&1 &
  echo $! >"$CONSOLE/$STAGE.pid"
  wait "$(cat "$CONSOLE/$STAGE.pid")" || exit 1
  tail -n 1 "$BASE/$STAGE/diagnostic_progress.jsonl"
done
```

完成依据：R1、R2 最后一行均为 `completed=200/status=complete`。随后比较完整状态、200 步轨迹、源图重渲染和固定 Mask：

```bash
"$PYTHON" tools/ru_part_mechanism_diagnostic.py compare \
  --u "$BASE/U/diagnostic_terminal_step10199.pt" \
  --r1 "$BASE/R1/diagnostic_terminal_step10199.pt" \
  --r2 "$BASE/R2/diagnostic_terminal_step10199.pt" \
  --output "$BASE/replay_equivalence.json"
```

只有 `REPLAY_ACCEPTED` 才进入 E；否则报告停止。

## E：ROI、旧 evidence 与人工确认停点

自动解析唯一 Garden track cache：

```bash
mapfile -t TRACKS < <(find "$ROOT/data/PURI-GS-derived" -type f \
  -name 'garden_factor4_static_tracks.pt' -print)
test "${#TRACKS[@]}" -eq 1
TRACK_CACHE="${TRACKS[0]}"
"$PYTHON" tools/ru_part_mechanism_diagnostic.py probe \
  --u-dir "$BASE/U" --track-cache "$TRACK_CACHE" --data-dir "$GARDEN" \
  --output-dir "$BASE/roi"
```

完成依据：`ROI_ESTABLISHED`。若为 `ROI_NOT_ESTABLISHED`，报告停止。输出 `roi_evidence_DSC07987.png`、`roi_evidence_DSC07989.png`、`monitor_views.png` 和 `roi_evidence.json`。

此处必须停下：把三张 PNG 下载给用户确认。未经明确确认，不得运行正式 VJP 或 P1/P2/O。

## F：确认后的 VJP 与固定 400 步分支

只有用户明确确认整块候选是静态背景时才运行：

```bash
"$PYTHON" tools/ru_part_mechanism_diagnostic.py confirm \
  --roi-dir "$BASE/roi" \
  --note '用户已查看叠图并明确确认整块候选为静态背景'
```

正式 VJP 是独立可丢弃进程，不执行 optimizer/topology/head/histogram 更新：

```bash
"$PYTHON" tools/ru_part_mechanism_diagnostic.py vjp --confirmed \
  --checkpoint "$BASE/U/replay_ckpts/replay_step9999.pt" \
  --roi-dir "$BASE/roi" --gsplat-dir "$GSPLAT" --data-dir "$GARDEN" \
  --output-dir "$BASE/VJP" --gpu "$GPU" \
  --dino-repo-dir "$DINO_REPO" --dino-weight-path "$DINO_WEIGHT" \
  --feature-cache-dir "$FEATURE_CACHE"
```

审阅 `VJP/vjp_report.json` 后，仅当结论为 `LOCAL_CONTROLLABILITY_PRESENT` 才记录正门。P1/P2/O 还必须由用户再次明确要求才可启动；每支固定 400 update，冻结 topology、strategy、Mask head 和 histogram，只有 O 增加系数 0.8、全图 HWC mean 的静态解屏蔽 L1。

后台分支的最简进度仍是：

```bash
tail -n 1 "$BASE/P1/diagnostic_progress.jsonl" 2>/dev/null
```

报告生成后强制停止；不自动运行 P/R/C、30k、跨场景或多种子实验。
