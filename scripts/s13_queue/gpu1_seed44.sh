#!/usr/bin/env bash
set -euo pipefail

cd /data/hdd1/yanshuo/2026code/CamylaNet
source scripts/s13_queue/common.sh

msg "===== GPU1 SEED44 BASELINE START ====="

run_full_experiment \
    1 \
    nnUNetTrainer_500epochs_Seed44 \
    seed44_baseline

msg "===== GPU1 SEED44 BASELINE COMPLETE ====="
