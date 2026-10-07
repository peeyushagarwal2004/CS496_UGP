#!/usr/bin/env bash
# Start the multi-seed sweep only once the orphaned w8/seed-1 training is out of
# the way, so two processes never write the same checkpoint.  Proceeds when any of:
#   - no python.exe is running at all (the orphan finished or died)
#   - the orphan's run is complete (30 epochs logged; the sweep will then skip it)
#   - its epoch log has not advanced for 30 min (the orphan is dead; ~3-6 min/epoch)
LOG="checkpoints/resnet8_w8_s1_cifar10.log.json"
START=$(date +%s)
mkdir -p logs
while true; do
  if ! tasklist //FI "IMAGENAME eq python.exe" //NH 2>/dev/null | grep -qi python.exe; then
    echo "gate: no python running"; break
  fi
  n=$(.venv/Scripts/python.exe -c "import json;print(len(json.load(open('$LOG'))))" 2>/dev/null || echo 0)
  if [ "$n" -ge 30 ]; then echo "gate: orphan completed 30 epochs"; break; fi
  if [ -f "$LOG" ]; then last=$(date -r "$LOG" +%s); else last=$START; fi
  if [ $(( $(date +%s) - last )) -ge 1800 ]; then echo "gate: orphan stalled 30 min"; break; fi
  sleep 60
done
echo "gate opened $(date '+%m-%d %H:%M')"
bash scripts/run_width_sweep_seeds.sh
