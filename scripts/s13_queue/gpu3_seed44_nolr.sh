#!/usr/bin/env bash
set -euo pipefail

cd /data/hdd1/yanshuo/2026code/CamylaNet
source scripts/s13_queue/common.sh

msg "===== GPU3 SEED44 NOLR START ====="

run_full_experiment \
    3 \
    nnUNetTrainer_500epochs_NoLRMirror_Seed44 \
    seed44_nolr

msg "===== GPU3 SEED44 NOLR COMPLETE ====="
