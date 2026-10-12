#!/usr/bin/env bash
set -euo pipefail

cd /data/hdd1/yanshuo/2026code/CamylaNet
source scripts/s13_queue/common.sh

GPU=1
TR=nnUNetTrainer_500epochs_NoMirrorAxis0_Seed46

msg "===== GPU1 SEED46 NOAXIS0 VALIDATION START ====="

run_validation \
    "$GPU" \
    "$TR" \
    "final" \
    "$LOG/seed46_noaxis0_val_final.log"

check_validation "$TR"
archive_validation "$TR" "validation_final_seed46_noaxis0"

run_validation \
    "$GPU" \
    "$TR" \
    "best" \
    "$LOG/seed46_noaxis0_val_best.log"

check_validation "$TR"
archive_validation "$TR" "validation_best_seed46_noaxis0"

msg "===== GPU1 SEED46 NOAXIS0 VALIDATION COMPLETE ====="
