#!/usr/bin/env bash
# The remaining GPU work: firm up claims that currently rest on one model.
#   1. RepVGG-A0 with three seeds (its 17.8x figure rests on a single model)
#   2. Exhaustive ground truth for e3m2, so the mantissa claim can be exact
#   3. Activation campaigns on RepVGG and the ViT (O4 covers ResNet8 only)
PY=python3
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
mkdir -p logs results checkpoints
LOCK=logs/remaining.lock
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "ABORT: runner PID $(cat "$LOCK") already active"; exit 1; fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT
export STOP_FILE=logs/STOP_REMAINING; rm -f "$STOP_FILE"
check_stop(){ if [ -f "$STOP_FILE" ]; then echo "STOP requested $(date '+%m-%d %H:%M')"; exit 0; fi; }
epochs_done(){ $PY -c "import json;print(len(json.load(open('checkpoints/$1_cifar10.log.json'))))" 2>/dev/null || echo 0; }

echo "########## 1. RepVGG-A0, three seeds $(date '+%m-%d %H:%M') ##########"
for S in 0 1 2; do
  check_stop
  T=$([ "$S" = 0 ] && echo repvgg_a0 || echo repvgg_a0_s$S)
  if [ "$(epochs_done "$T")" -ge 30 ]; then echo "[repvgg s$S] already trained"
  else $PY -m mxfi.train --model repvgg_a0 --seed "$S" --epochs 30 --workers 8 \
         --device cuda 2>&1 | grep -aE "done:|Traceback|Error"; fi
  [ "$(epochs_done "$T")" -lt 30 ] && { echo "SKIP repvgg s$S"; continue; }
  for F in e5m2 e4m3 e3m2; do
    check_stop
    $PY -m experiments.e01_uniform_baseline --model repvgg_a0 --train-seed "$S" \
        --fmt "$F" --n 3000 --device cuda 2>&1 | grep -aE "model-wide SDC|scale-vs-element"
    echo "[repvgg s$S $F] $(date '+%m-%d %H:%M')"
  done
done

echo "########## 2. exhaustive ground truth, e3m2 $(date '+%m-%d %H:%M') ##########"
check_stop
$PY -m experiments.e06_exhaustive --model resnet8_w16 --fmt e3m2 --device cuda 2>&1 \
  | grep -aE "exhaustive fault space|model-wide SDC|truth|estimate|CI covers|inj/s" | tail -12

echo "########## 3. activations on RepVGG and the ViT $(date '+%m-%d %H:%M') ##########"
for M in repvgg_a0 vit_small; do
  check_stop
  echo "--- $M ---"
  $PY -m experiments.e04_activations --model "$M" --n 2000 --images 25 --device cuda 2>&1 \
    | grep -aE "overall|element |scale |WEIGHT vs|^element|^scale|Traceback|Error" | head -14
  echo "[activations $M] $(date '+%m-%d %H:%M')"
done
echo "=== REMAINING GPU WORK COMPLETE $(date '+%m-%d %H:%M') ==="
