#!/bin/bash
cd /data/rajeshr/peeyush/bitflip
for s in 0 9; do for f in e3m2 e4m3 e5m2; do
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 python3 -m experiments.e21_multibit run --fmt $f --train-seed $s --device cuda
done; done
echo E21_DONE
