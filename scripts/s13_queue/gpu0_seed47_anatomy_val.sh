#!/usr/bin/env bash
set -euo pipefail

cd /data/hdd1/yanshuo/2026code/CamylaNet
source scripts/s13_queue/common.sh

GPU=0
TR=nnUNetTrainer_500epochs_AnatomyAwareMirror_Seed47

msg "===== GPU0 SEED47 ANATOMYAWARE VALIDATION START ====="

run_validation \
    "$GPU" \
    "$TR" \
    "final" \
    "$LOG/seed47_anatomy_val_final.log"

check_validation "$TR"
archive_validation "$TR" "validation_final_seed47_anatomy"

run_validation \
    "$GPU" \
    "$TR" \
    "best" \
    "$LOG/seed47_anatomy_val_best.log"

check_validation "$TR"
archive_validation "$TR" "validation_best_seed47_anatomy"

msg "===== GPU0 SEED47 ANATOMYAWARE VALIDATION COMPLETE ====="
