#!/usr/bin/env bash
set -euo pipefail

cd /data/hdd1/yanshuo/2026code/CamylaNet
source scripts/s13_queue/common.sh

GPU=1

msg "===== GPU1 SEED47 CHAIN WAITING ====="

while pgrep -f \
"bash scripts/s13_queue/gpu1_after_seed45.sh" \
>/dev/null 2>&1
do
    sleep 60
done

msg "Seed45 Axis0 chain finished"

run_full_experiment \
    "$GPU" \
    nnUNetTrainer_500epochs_Seed47 \
    seed47_baseline

run_full_experiment \
    "$GPU" \
    nnUNetTrainer_500epochs_NoMirrorAxis0_Seed47 \
    seed47_noaxis0

msg "===== GPU1 SEED47 CHAIN COMPLETE ====="
