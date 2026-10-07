#!/usr/bin/env bash
# All-GPU multi-seed width sweep, run on the IITK A100 box.
#
# Ten independently trained models per width, every model trained *and* measured
# on the same device, so the whole dataset is internally consistent.  The CPU
# sweep on the laptop stays as an independent replication on different hardware.
#
# Restartable: a width already trained to 30 epochs is not retrained, and a
# campaign whose CSV already holds all 3000 rows is not rerun.
# Stop cleanly at any point with:  touch logs/STOP
PY=python3
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"   # the less loaded A100
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
WIDTHS="8 16 24 32 48"
FORMATS="e5m2 e4m3 e3m2"
EPOCHS=30
N=3000
WORKERS=8
DEV=cuda

mkdir -p logs results checkpoints
LOCK=logs/gpu_sweep.lock
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "ABORT: runner PID $(cat "$LOCK") is already active"; exit 1
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

export STOP_FILE=logs/STOP
rm -f "$STOP_FILE"
check_stop() {
  if [ -f "$STOP_FILE" ]; then
    echo "STOP requested -- exiting cleanly $(date '+%m-%d %H:%M')"; exit 0
  fi
}

stem() { if [ "$2" = "0" ]; then echo "$1"; else echo "${1}_s$2"; fi; }
epochs_done() {
  $PY -c "import json;print(len(json.load(open('checkpoints/$1_cifar10.log.json'))))" 2>/dev/null || echo 0
}
rows_done() {
  f="results/e01_$1_$2-K32-ocp-w-n${N}.csv"
  if [ -f "$f" ]; then $PY -c "import pandas as pd;print(len(pd.read_csv('$f')))" 2>/dev/null || echo 0
  else echo 0; fi
}

echo "runner PID $$ | GPU $CUDA_VISIBLE_DEVICES | seeds: $SEEDS"
for S in $SEEDS; do
  echo "########## SEED $S START $(date '+%m-%d %H:%M') ##########"
  for W in $WIDTHS; do
    check_stop
    M="resnet8_w${W}"; T=$(stem "$M" "$S")

    if [ "$(epochs_done "$T")" -ge "$EPOCHS" ]; then
      echo "[s$S w$W] already trained"
    else
      $PY -m mxfi.train --model "$M" --seed "$S" --epochs "$EPOCHS" \
          --workers "$WORKERS" --device "$DEV" 2>&1 | grep -aE "done:|Traceback|Error"
    fi

    check_stop
    got=$(epochs_done "$T")
    if [ "$got" -lt "$EPOCHS" ]; then
      echo "SKIP s$S w$W: training reached only $got/$EPOCHS epochs"; continue
    fi

    for F in $FORMATS; do
      check_stop
      if [ "$(rows_done "$T" "$F")" -ge "$N" ]; then
        echo "[s$S w$W $F] already done"; continue
      fi
      $PY -m experiments.e01_uniform_baseline --model "$M" --train-seed "$S" \
          --fmt "$F" --n "$N" --device "$DEV" 2>&1 \
          | grep -aE "model-wide SDC|Traceback|Error"
      echo "[s$S w$W $F] $(date '+%m-%d %H:%M')"
    done
  done
  $PY -m experiments.width_sweep_multiseed > "results/multiseed_after_seed${S}.txt" 2>&1
  echo "=== SEED $S COMPLETE $(date '+%m-%d %H:%M') ==="
done
echo "=== GPU SWEEP COMPLETE $(date '+%m-%d %H:%M') ==="
