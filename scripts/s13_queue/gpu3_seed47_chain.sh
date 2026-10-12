#!/usr/bin/env bash
set -euo pipefail

cd /data/hdd1/yanshuo/2026code/CamylaNet
source scripts/s13_queue/common.sh

GPU=3

msg "===== GPU3 SEED47 CHAIN WAITING ====="

while pgrep -f \
"bash scripts/s13_queue/gpu3_after_seed45.sh" \
>/dev/null 2>&1
do
    sleep 60
done

msg "Seed45 Axis1 chain finished"

run_full_experiment \
    "$GPU" \
    nnUNetTrainer_500epochs_NoLRMirror_Seed47 \
    seed47_nolr

run_full_experiment \
    "$GPU" \
    nnUNetTrainer_500epochs_NoMirrorAxis1_Seed47 \
    seed47_noaxis1

msg "===== GPU3 SEED47 CHAIN COMPLETE ====="
