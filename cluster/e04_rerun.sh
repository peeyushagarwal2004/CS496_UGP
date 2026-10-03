#!/bin/bash
cd /data/rajeshr/peeyush/bitflip
export PYTHONPATH=.
for m in vit_small repvgg_a0 vit_small_c100 repvgg_a0_c100 resnet8_c100; do
  echo "== $m $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=1 python3 -m experiments.e04_activations --model $m --fmt e4m3 --images 100 --n 4000 --device cuda 2>&1 | grep -aE "overall|element|scale|weight|Error|Trace" | head -12
done
echo E04_DONE
