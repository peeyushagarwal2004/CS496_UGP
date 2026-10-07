#!/usr/bin/env bash
# Multi-seed width sweep: retrain every width under more seeds so F21's ordering
# can be tested against model-to-model variance (F22).
#
# Seed 0 is the original sweep and is reused as-is.  Order is seed-first
# (all widths of seed 1, then all of seed 2, ...) so that stopping at any point
# leaves a *balanced* dataset: after seed 2 there are 3 complete seeds, after
# seed 4 there are 5.
#
# Restartable: a width already trained to 30 epochs is not retrained, and a
# campaign whose CSV already holds all 3000 rows is not rerun.  Re-running this
# script after an interruption resumes where it stopped.
#
# Override seeds with e.g.  SEEDS="1 2" bash scripts/run_width_sweep_seeds.sh
PY=".venv/Scripts/python.exe"
SEEDS="${SEEDS:-1 2 3 4}"
WIDTHS="8 16 24 32 48"
FORMATS="e5m2 e4m3 e3m2"
EPOCHS=30
N=3000

# --- single-runner lock -------------------------------------------------
# Stopping a background task on Windows does not kill its descendants, so a
# stopped runner can keep going and collide with a fresh one.  See
# sweep_lock.sh for why the lock records the Windows PID, not $$.
. scripts/sweep_lock.sh
LOCK="logs/sweep.lock"
mkdir -p logs
acquire_lock "$LOCK" || { echo "ABORT: another runner is already active; not starting a second"; exit 1; }
trap 'rm -f "$LOCK"' EXIT
echo "runner Windows PID $(cat "$LOCK")"

# --- graceful stop -------------------------------------------------------
# Create logs/STOP to end the sweep cleanly: training halts after the epoch in
# progress (its checkpoint is already saved) and the runner exits before its
# next step.  Rerunning this script resumes exactly where it stopped.  A fresh
# launch clears any old STOP, since launching is itself the request to run.
export STOP_FILE="logs/STOP"
rm -f "$STOP_FILE"
check_stop() {
  if [ -f "$STOP_FILE" ]; then echo "STOP requested -- exiting cleanly $(stamp)"; exit 0; fi
}

stem() { if [ "$2" = "0" ]; then echo "$1"; else echo "$1_s$2"; fi; }

epochs_done() {
  "$PY" -c "import json;print(len(json.load(open('checkpoints/$1_cifar10.log.json'))))" 2>/dev/null || echo 0
}

rows_done() {
  local f="results/e01_$1_$2-K32-ocp-w-n${N}.csv"
  if [ -f "$f" ]; then
    "$PY" -c "import pandas as pd;print(len(pd.read_csv(r'$f')))" 2>/dev/null || echo 0
  else
    echo 0
  fi
}

stamp() { date '+%m-%d %H:%M'; }

for S in $SEEDS; do
  echo "########## SEED $S START $(stamp) ##########"
  for W in $WIDTHS; do
    check_stop
    M="resnet8_w${W}"
    T=$(stem "$M" "$S")

    if [ "$(epochs_done "$T")" -ge "$EPOCHS" ]; then
      echo "[seed $S w$W] already trained"
    else
      echo "[seed $S w$W] training $(stamp)"
      "$PY" -m mxfi.train --model "$M" --seed "$S" --epochs "$EPOCHS" 2>&1 \
        | grep -aE "done:|Traceback|Error"
    fi

    check_stop
    got=$(epochs_done "$T")
    if [ "$got" -lt "$EPOCHS" ]; then
      echo "SKIP seed $S w$W: training reached only $got/$EPOCHS epochs"
      continue
    fi

    for F in $FORMATS; do
      check_stop
      if [ "$(rows_done "$T" "$F")" -ge "$N" ]; then
        echo "[seed $S w$W $F] already done"
        continue
      fi
      "$PY" -m experiments.e01_uniform_baseline --model "$M" --train-seed "$S" \
          --fmt "$F" --n "$N" 2>&1 | grep -aE "model-wide SDC|Traceback|Error"
      echo "[seed $S w$W $F] campaign finished $(stamp)"
    done
  done

  "$PY" -m experiments.width_sweep_multiseed \
      > "results/width_sweep_multiseed_after_seed${S}.txt" 2>&1
  echo "=== SEED $S COMPLETE $(stamp) ==="
done

echo "=== MULTISEED SWEEP COMPLETE $(stamp) ==="
