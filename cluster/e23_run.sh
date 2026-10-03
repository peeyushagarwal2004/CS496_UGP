#!/bin/bash
cd /data/rajeshr/peeyush/bitflip
run(){ PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 python3 -u -m experiments.e23_nan_guard --device cuda "$@" 2>&1 | grep -v "RuntimeWarning\|vals = "; }
for f in e4m3 e5m2; do
  run --model resnet8_w16 --train-seed 0 --fmt $f
  if [ $f = e4m3 ]; then run --model resnet8_w16 --train-seed 9 --fmt $f --exhaustive results/e06_resnet8_w16_e4m3-K32-ocp-w_exhaustive.csv
  else run --model resnet8_w16 --train-seed 9 --fmt $f; fi
  for s in 0 1 2; do run --model repvgg_a0 --train-seed $s --fmt $f; run --model vit_small --train-seed $s --fmt $f; done
done
echo E23_DONE
