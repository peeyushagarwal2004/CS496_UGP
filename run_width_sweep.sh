#!/usr/bin/env bash
# Width sweep: isolate model capacity as the variable behind F18.
#
# Same topology, same data, same 30-epoch budget -- only ResNet8's base width
# changes, giving a 35x parameter range.  Per width we train then immediately
# campaign, so an interruption leaves complete widths rather than a pile of
# checkpoints with no results.
#
# Formats are chosen to form a gradient in special-code availability:
#   e5m2 = Inf+NaN, e4m3 = NaN only, e3m2 = none.
# F18 predicts vulnerability should grow with capacity for e5m2, less for
# e4m3, and not at all for e3m2.
PY=".venv/Scripts/python.exe"

for W in 8 16 24 32 48; do
  M="resnet8_w${W}"
  echo "########## WIDTH ${W} : train ##########"
  "$PY" -m mxfi.train --model "$M" --epochs 30 2>&1 | tail -5

  N=$("$PY" -c "import json;print(len(json.load(open('checkpoints/${M}_cifar10.log.json'))))" 2>/dev/null || echo 0)
  if [ "$N" -lt 30 ]; then
    echo "SKIP width ${W}: training reached only epoch ${N}"
    continue
  fi

  for F in e5m2 e4m3 e3m2; do
    echo "----- width ${W} / ${F} -----"
    "$PY" -m experiments.e01_uniform_baseline --model "$M" --fmt "$F" --n 3000 2>&1 \
      | grep -aE "golden acc|model-wide SDC|scale-vs-element|non-finite outputs|mean images"
  done
done

echo "########## WIDTH SWEEP COMPLETE ##########"
"$PY" -m experiments.width_sweep_analysis 2>&1
