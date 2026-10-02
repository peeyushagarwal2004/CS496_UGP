#!/usr/bin/env bash
# CIFAR-100 as the stand-in for ImageNet: train each model, then run the core
# campaigns on it. Two lanes, one per GPU, each restartable: training resumes from
# its last epoch, and a campaign whose output already exists is skipped.
#
#   lane A (GPU 1): the ViT, seeds 0-2      -- the DeiT stand-in
#   lane B (GPU 0): RepVGG-A0 seeds 0-2, then ResNet8 seeds 0-2
#   lane C (GPU 1): activation faults under e5m2 (E04)
#
# Run on the cluster:  (nohup ./c100_run.sh A > logs/c100_A.log 2>&1 &)
#                      (nohup ./c100_run.sh B > logs/c100_B.log 2>&1 &)
# Stop gracefully:     touch logs/STOP_C100   (halts after the current epoch/step)
PY=python3
export PYTHONPATH=.
export STOP_FILE=logs/STOP_C100
N=3000
mkdir -p logs results checkpoints

check_stop(){ if [ -f "$STOP_FILE" ]; then echo "STOP requested $(date '+%m-%d %H:%M')"; exit 0; fi; }
stem(){ if [ "$2" = "0" ]; then echo "$1"; else echo "${1}_s$2"; fi; }
ckpt(){ b=${1%_c100}; if [ "$2" = "0" ]; then echo "${b}_cifar100"; else echo "${b}_s$2_cifar100"; fi; }
epochs_done(){ $PY -c "import json;print(len(json.load(open('checkpoints/$1.log.json'))))" 2>/dev/null || echo 0; }

train(){  # model seed epochs extra-args...
  local m=$1 s=$2 ep=$3; shift 3
  check_stop
  if [ "$(epochs_done "$(ckpt "$m" "$s")")" -ge "$ep" ]; then echo "[$m s$s] already trained"; return; fi
  echo "[$m s$s] training $(date '+%m-%d %H:%M')"
  $PY -m mxfi.train --model "$m" --seed "$s" --epochs "$ep" --device cuda --workers 8 "$@" \
      2>&1 | grep -aE "^epoch +[0-9]*[05] |done:|Traceback|Error"
}

campaigns(){  # model seed formats...
  local m=$1 s=$2; shift 2
  local t; t=$(stem "$m" "$s")
  check_stop
  if [ "$(epochs_done "$(ckpt "$m" "$s")")" -lt 1 ]; then echo "[$m s$s] no checkpoint"; return; fi
  if [ "$s" = "0" ] && [ ! -f "results/e00_${m}_quantized_accuracy.csv" ]; then
    $PY -m experiments.e00_quantized_accuracy --model "$m" --blocks 32 --modes ocp \
        --device cuda 2>&1 | grep -aE "fp32|->|Error"
  fi
  for f in "$@"; do
    check_stop
    [ -f "results/e01_${t}_${f}-K32-ocp-w-n${N}.csv" ] && continue
    echo "[$t $f] e01 $(date '+%m-%d %H:%M')"
    $PY -m experiments.e01_uniform_baseline --model "$m" --train-seed "$s" --fmt "$f" \
        --n "$N" --device cuda 2>&1 | grep -aE "golden|model-wide|scale-vs|non-finite|mean images|Error"
  done
  for f in e4m3 e5m2; do
    check_stop
    [ -f "results/e23_${t}_${f}-K32-ocp-w_nan_guard.csv" ] && continue
    $PY -u -m experiments.e23_nan_guard --model "$m" --train-seed "$s" --fmt "$f" \
        --device cuda 2>&1 | grep -aE "golden|landing|whole element|Error"
  done
  if [ "$s" = "0" ] && [ ! -f "results/e04_${m}_activations_e4m3-K32-ocp-wa.csv" ]; then
    $PY -m experiments.e04_activations --model "$m" --fmt e4m3 --device cuda \
        2>&1 | grep -aE "per inference|weight|activation|Error" | head -20
  fi
}

case "$1" in
  A) export CUDA_VISIBLE_DEVICES=1
     for s in 0 1 2; do
       train vit_small_c100 "$s" 100 --optimizer adamw --lr 1e-3 --weight-decay 0.05 --label-smoothing 0.1
       campaigns vit_small_c100 "$s" e5m2 e4m3 e3m2 e2m3 e2m1
     done ;;
  B) export CUDA_VISIBLE_DEVICES=0
     for s in 0 1 2; do
       train repvgg_a0_c100 "$s" 30 --lr 0.1
       campaigns repvgg_a0_c100 "$s" e5m2 e4m3 e3m2
     done
     for s in 0 1 2; do
       train resnet8_c100 "$s" 60 --lr 0.1
       campaigns resnet8_c100 "$s" e5m2 e4m3 e3m2
     done ;;
  C) # activation faults under the other NaN-coded format, for the models trained here
     export CUDA_VISIBLE_DEVICES=1
     for m in vit_small repvgg_a0 vit_small_c100 repvgg_a0_c100 resnet8_c100; do
       check_stop
       [ -f "results/e04_${m}_activations_e5m2-K32-ocp-wa.csv" ] && continue
       echo "[$m] e04 e5m2 $(date '+%m-%d %H:%M')"
       $PY -m experiments.e04_activations --model "$m" --fmt e5m2 --images 100 --n 4000 \
           --device cuda 2>&1 | grep -aE "overall|element|scale|Error|Trace" | head -6
     done ;;
  *) echo "usage: $0 A|B|C"; exit 1 ;;
esac
echo "LANE_$1_DONE $(date '+%m-%d %H:%M')"
