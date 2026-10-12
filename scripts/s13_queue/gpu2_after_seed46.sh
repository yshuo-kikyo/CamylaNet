#!/usr/bin/env bash
set -euo pipefail

cd /data/hdd1/yanshuo/2026code/CamylaNet
source scripts/s13_queue/common.sh

GPU=2

BASE=nnUNetTrainer_500epochs_Seed46
NOLR=nnUNetTrainer_500epochs_NoLRMirror_Seed46

B5A=nnUNetTrainer_500epochs_NoMirrorAxis0_Seed46
B5B=nnUNetTrainer_500epochs_NoMirrorAxis1_Seed46

msg "===== GPU2 POST-SEED46 QUEUE START ====="

while pgrep -f \
"scripts/s13_queue/gpu2_seed46_pair.sh" \
>/dev/null 2>&1
do
    sleep 60
done

msg "Original Seed46 paired training queue finished"

validate_existing_run \
    "$GPU" \
    "$BASE" \
    "seed46_baseline"

validate_existing_run \
    "$GPU" \
    "$NOLR" \
    "seed46_nolr"

run_full_experiment \
    "$GPU" \
    "$B5A" \
    "seed46_noaxis0"

run_full_experiment \
    "$GPU" \
    "$B5B" \
    "seed46_noaxis1"

msg "===== GPU2 B5 COMPLETE ====="
