#!/bin/bash
cd /data/rajeshr/peeyush/bitflip
for L in 6 16; do
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' python3 -u -m experiments.e22_learned_vs_exact --fmt e3m2 --train-seed 0 --leaves $L
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' python3 -u -m experiments.e22_learned_vs_exact --fmt e4m3 --train-seed 9 --leaves $L
done
echo E22_DONE
