#!/usr/bin/env bash
set -euo pipefail

cd /data/hdd1/yanshuo/2026code/CamylaNet
source scripts/s13_queue/common.sh

GPU=1

BASE=nnUNetTrainer_500epochs_Seed45
B5=nnUNetTrainer_500epochs_NoMirrorAxis0_Seed45

msg "===== GPU1 POST-SEED45 QUEUE START ====="

while pgrep -f \
"nnUNetv2_train 302 3d_fullres 4 -tr ${BASE}$" \
>/dev/null 2>&1
do
    sleep 60
done

msg "Seed45 baseline training finished"

validate_existing_run \
    "$GPU" \
    "$BASE" \
    "seed45_baseline"

run_full_experiment \
    "$GPU" \
    "$B5" \
    "seed45_noaxis0"

msg "===== GPU1 B5 COMPLETE ====="
