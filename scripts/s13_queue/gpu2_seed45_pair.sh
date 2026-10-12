#!/usr/bin/env bash
set -euo pipefail

cd /data/hdd1/yanshuo/2026code/CamylaNet
source scripts/s13_queue/common.sh

msg "===== GPU2 SEED45 PAIRED RUN START ====="

run_full_experiment \
    2 \
    nnUNetTrainer_500epochs_Seed45 \
    seed45_baseline

run_full_experiment \
    2 \
    nnUNetTrainer_500epochs_NoLRMirror_Seed45 \
    seed45_nolr

msg "===== GPU2 SEED45 PAIRED RUN COMPLETE ====="
