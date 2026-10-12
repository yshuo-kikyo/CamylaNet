#!/usr/bin/env bash
set -euo pipefail

cd /data/hdd1/yanshuo/2026code/CamylaNet
source scripts/s13_queue/common.sh

GPU=3

NOLR=nnUNetTrainer_500epochs_NoLRMirror_Seed45
B5=nnUNetTrainer_500epochs_NoMirrorAxis1_Seed45

msg "===== GPU3 POST-SEED45 QUEUE START ====="

while pgrep -f \
"nnUNetv2_train 302 3d_fullres 4 -tr ${NOLR}$" \
>/dev/null 2>&1
do
    sleep 60
done

msg "Seed45 NoLR training finished"

validate_existing_run \
    "$GPU" \
    "$NOLR" \
    "seed45_nolr"

run_full_experiment \
    "$GPU" \
    "$B5" \
    "seed45_noaxis1"

msg "===== GPU3 B5 COMPLETE ====="
