#!/usr/bin/env bash
# Small vision transformer on CIFAR-10: the third architecture family.
# Three seeds, since model-to-model spread is ~10x the injection noise (F24),
# so a single trained model would not support a claim.
PY=python3
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
SEEDS="${SEEDS:-0 1 2}"; EPOCHS=100; N=3000
FORMATS="e5m2 e4m3 e3m2 e2m3 e2m1"

mkdir -p logs results checkpoints
LOCK=logs/vit.lock
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "ABORT: runner PID $(cat "$LOCK") already active"; exit 1; fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT
export STOP_FILE=logs/STOP_VIT; rm -f "$STOP_FILE"
check_stop(){ if [ -f "$STOP_FILE" ]; then echo "STOP requested $(date '+%m-%d %H:%M')"; exit 0; fi; }
stem(){ if [ "$2" = "0" ]; then echo "$1"; else echo "${1}_s$2"; fi; }
epochs_done(){ $PY -c "import json;print(len(json.load(open('checkpoints/$1_cifar10.log.json'))))" 2>/dev/null || echo 0; }
rows_done(){ f="results/e01_$1_$2-K32-ocp-w-n${N}.csv"
  if [ -f "$f" ]; then $PY -c "import pandas as pd;print(len(pd.read_csv('$f')))" 2>/dev/null || echo 0; else echo 0; fi; }

for S in $SEEDS; do
  echo "########## ViT SEED $S START $(date '+%m-%d %H:%M') ##########"
  T=$(stem vit_small "$S"); check_stop
  if [ "$(epochs_done "$T")" -ge "$EPOCHS" ]; then echo "[vit s$S] already trained"
  else
    $PY -m mxfi.train --model vit_small --seed "$S" --epochs "$EPOCHS" \
        --optimizer adamw --lr 1e-3 --weight-decay 0.05 --label-smoothing 0.1 \
        --workers 8 --device cuda 2>&1 | grep -aE "done:|Traceback|Error"
  fi
  check_stop
  if [ "$(epochs_done "$T")" -lt "$EPOCHS" ]; then echo "SKIP vit s$S: incomplete training"; continue; fi

  if [ "$S" = "0" ]; then
    echo "--- quantised accuracy by format ---"
    $PY -m experiments.e00_quantized_accuracy --model vit_small --blocks 32 \
        --modes ocp fit --device cuda 2>&1 | grep -aE "^  e|fp32|Traceback|Error"
  fi

  for F in $FORMATS; do
    check_stop
    if [ "$(rows_done "$T" "$F")" -ge "$N" ]; then echo "[vit s$S $F] already done"; continue; fi
    $PY -m experiments.e01_uniform_baseline --model vit_small --train-seed "$S" \
        --fmt "$F" --n "$N" --device cuda 2>&1 \
      | grep -aE "model-wide SDC|scale-vs-element|non-finite outputs|Traceback|Error"
    echo "[vit s$S $F] $(date '+%m-%d %H:%M')"
  done
  echo "=== ViT SEED $S COMPLETE $(date '+%m-%d %H:%M') ==="
done
echo "=== VIT RUN COMPLETE $(date '+%m-%d %H:%M') ==="
