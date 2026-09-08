# RU-PART reference/control runbook

This runbook executes only the first causal stage: fresh `parent`, diagnostic
`noop`, and behavior-preserving `current`. Run them serially on one GPU so that
timing and memory contention cannot differ between controls.

## 1. One-time preflight and patch preparation

```bash
set -euo pipefail
ROOT=/home/chenglong/Uncertain-Nerf/uncertain-nerf
GSPLAT="$ROOT/external/gsplat-v1.5.3-ru-part-controls"
GARDEN="$ROOT/data/mipnerf360/360_v2/garden"
DINO_REPO="$ROOT/external/dinov2"
DINO_WEIGHT="$ROOT/data/PURI-GS-assets/dinov2/dinov2_vits14_reg4_pretrain.pth"
FEATURE_CACHE="$ROOT/data/PURI-GS-derived/semantic_features/garden"
TRACK_CACHE=UNUSED_FOR_PARENT
OUT="$ROOT/logs-puri/ru_part_reference_controls"
CONSOLE="$ROOT/logs-puri/ru_part_reference_controls_console"
GPU=6

cd "$ROOT"
test "$(git branch --show-current)" = ru-part
test -d "$GARDEN/images_4_png"
test -f "$DINO_WEIGHT"
test -f "$FEATURE_CACHE/manifest.json"
mkdir -p "$OUT" "$CONSOLE"
bash scripts/prepare_puri_gs_ru_part.sh "$GSPLAT"
```

Completion criteria: the last command prints
`PURI-GS-RU-PART-PATCH-READY`; all `test` commands return zero. A Garden track
cache is not required by the parent mode. It is checked only after
the parent is accepted and before launching noop/current.

## 2. Launch one control at a time

Parent:

```bash
cd "$ROOT"
nohup env PURI_GSPLAT_PYTHON="$ROOT/.venv-gsplat153/bin/python" \
  bash scripts/train_puri_gs_ru_part.sh \
  "$GSPLAT" "$GARDEN" "$OUT/parent_seed42" "$GPU" \
  "$DINO_REPO" "$DINO_WEIGHT" "$FEATURE_CACHE" "$TRACK_CACHE" full parent \
  >"$CONSOLE/parent_seed42.log" 2>&1 &
echo $! >"$CONSOLE/parent_seed42.pid"
```

After parent completes successfully, launch noop:

```bash
cd "$ROOT"
test -f "$OUT/parent_seed42/ckpts/ckpt_29999_rank0.pt"
mapfile -t TRACKS < <(find "$ROOT/data/PURI-GS-derived" -type f -name 'garden_factor4_static_tracks.pt' -print)
test "${#TRACKS[@]}" -eq 1
TRACK_CACHE="${TRACKS[0]}"
nohup env PURI_GSPLAT_PYTHON="$ROOT/.venv-gsplat153/bin/python" \
  bash scripts/train_puri_gs_ru_part.sh \
  "$GSPLAT" "$GARDEN" "$OUT/noop_seed42" "$GPU" \
  "$DINO_REPO" "$DINO_WEIGHT" "$FEATURE_CACHE" "$TRACK_CACHE" full noop \
  >"$CONSOLE/noop_seed42.log" 2>&1 &
echo $! >"$CONSOLE/noop_seed42.pid"
```

After noop completes successfully, launch current:

```bash
cd "$ROOT"
test -f "$OUT/noop_seed42/ckpts/ckpt_29999_rank0.pt"
nohup env PURI_GSPLAT_PYTHON="$ROOT/.venv-gsplat153/bin/python" \
  bash scripts/train_puri_gs_ru_part.sh \
  "$GSPLAT" "$GARDEN" "$OUT/current_seed42" "$GPU" \
  "$DINO_REPO" "$DINO_WEIGHT" "$FEATURE_CACHE" "$TRACK_CACHE" full current \
  >"$CONSOLE/current_seed42.log" 2>&1 &
echo $! >"$CONSOLE/current_seed42.pid"
```

Do not launch the next control when the preceding checkpoint or metrics are
missing. Do not reuse an existing output directory.

## 3. Progress and health checks

Replace `MODE` with `parent`, `noop`, or `current`:

```bash
MODE=parent
tail -n 80 -f "$CONSOLE/${MODE}_seed42.log"
```

In another terminal:

```bash
MODE=parent
PID=$(cat "$CONSOLE/${MODE}_seed42.pid")
ps -p "$PID" -o pid,etime,%cpu,%mem,stat,cmd
nvidia-smi -i "$GPU"
find "$OUT/${MODE}_seed42/replay_ckpts" -maxdepth 1 -type f -printf '%f %s bytes\n' 2>/dev/null | sort
tail -n 5 "$OUT/${MODE}_seed42/Gaussian_count_curve.csv" 2>/dev/null
```

Healthy progress means the process is alive, the log step increases, GPU
utilization is nonzero during training, and replay checkpoints appear at steps
9999, 13299, 16599, and 19899. Training completion requires process exit, no
traceback, `ckpts/ckpt_29999_rank0.pt`, and `train_metrics.json`.

## 4. Independently evaluate each completed checkpoint

Run this after all three trainings, or after each training before launching the
next mode:

```bash
cd "$ROOT"
for MODE in parent noop current; do
  TRAIN_DIR="$OUT/${MODE}_seed42"
  EVAL_DIR="$OUT/${MODE}_seed42_eval"
  if [[ "$MODE" == current ]]; then
    CONFIG="$ROOT/configs/puri_gs_ru_part_garden30k.yaml"
  else
    CONFIG="$ROOT/configs/puri_gs_ru_part_${MODE}_garden30k.yaml"
  fi
  test -f "$TRAIN_DIR/ckpts/ckpt_29999_rank0.pt"
  test ! -e "$EVAL_DIR"
  "$ROOT/.venv-gsplat153/bin/python" run_puri_gs.py \
    --config "$CONFIG" \
    --gsplat-dir "$GSPLAT" \
    --data-dir "$GARDEN" \
    --result-dir "$EVAL_DIR" \
    --gpu "$GPU" \
    --checkpoint "$TRAIN_DIR/ckpts/ckpt_29999_rank0.pt" \
    2>&1 | tee "$CONSOLE/${MODE}_seed42_eval.log"
done
```

Evaluation completion requires `metrics.json`, `per_image_metrics.csv`, and
`ru_validation.json` in each evaluation directory. Compare the three runs only
after checking that `camera_sequence.json` SHA-256 values are identical.

## 5. Replay from the parent step-9999 state

Example for a noop replay branch:

```bash
cd "$ROOT"
nohup env PURI_GSPLAT_PYTHON="$ROOT/.venv-gsplat153/bin/python" \
  PURI_RU_PART_REPLAY_CHECKPOINT="$OUT/parent_seed42/replay_ckpts/replay_step9999.pt" \
  bash scripts/train_puri_gs_ru_part.sh \
  "$GSPLAT" "$GARDEN" "$OUT/noop_from_parent9999_seed42" "$GPU" \
  "$DINO_REPO" "$DINO_WEIGHT" "$FEATURE_CACHE" "$TRACK_CACHE" full noop \
  >"$CONSOLE/noop_from_parent9999_seed42.log" 2>&1 &
echo $! >"$CONSOLE/noop_from_parent9999_seed42.pid"
```

The resumed run must report `start_step: 10000` in `camera_sequence.json`, and
its camera-sequence SHA-256 must equal the parent run. A mismatch is a hard
failure, not a reason to regenerate or relax the sequence.
