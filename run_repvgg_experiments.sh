#!/usr/bin/env bash
# Wait for RepVGG-A0 training to reach 30 epochs, then run the core experiments.
# Each experiment is independent and writes its own CSV, so a failure in one
# does not stop the rest.
PY=".venv/Scripts/python.exe"
LOG="checkpoints/repvgg_a0_cifar10.log.json"

epochs_done() {
  "$PY" -c "import json,sys;print(len(json.load(open('$LOG'))))" 2>/dev/null || echo 0
}

echo "waiting for repvgg_a0 training (30 epochs)..."
for i in $(seq 1 240); do          # 240 * 5 min = 20 h ceiling
  n=$(epochs_done)
  if [ "$n" -ge 30 ]; then break; fi
  sleep 300
done

n=$(epochs_done)
if [ "$n" -lt 30 ]; then
  echo "ABORT: training only reached epoch $n; not starting experiments"
  exit 1
fi
echo "=== training complete ($n epochs), starting experiments ==="

echo "########## E00 quantised accuracy ##########"
"$PY" -m experiments.e00_quantized_accuracy --model repvgg_a0 2>&1 | grep -v "fi/s\]"

echo "########## E01 uniform FI (matched-accuracy formats) ##########"
for f in e4m3 e5m2 e3m2; do
  echo "----- $f -----"
  "$PY" -m experiments.e01_uniform_baseline --model repvgg_a0 --fmt $f --n 3000 2>&1 | grep -v "fi/s\]"
done

echo "########## E03b block size, matched faults ##########"
for m in ocp fit; do
  echo "----- scale_mode=$m -----"
  "$PY" -m experiments.e03b_block_size_matched --model repvgg_a0 --n 1500 --scale-mode $m 2>&1 | grep -v "fi/s\]"
done

echo "=== ALL REPVGG EXPERIMENTS COMPLETE ==="
