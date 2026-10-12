#!/usr/bin/env bash
set -euo pipefail

cd /data/hdd1/yanshuo/2026code/CamylaNet
source scripts/s13_queue/common.sh

GPU=2

NOLR=nnUNetTrainer_500epochs_NoLRMirror_Seed46
A0=nnUNetTrainer_500epochs_NoMirrorAxis0_Seed46
A1=nnUNetTrainer_500epochs_NoMirrorAxis1_Seed46

NOLR_DIR="$ROOT/${NOLR}__nnUNetPlans__3d_fullres/fold_4"

msg "===== GPU2 SEED46 POST-TRAIN CHAIN START ====="

test -f "$NOLR_DIR/checkpoint_final.pth" || {
    msg "ERROR: Seed46 NoLR checkpoint_final missing"
    exit 1
}

msg "Seed46 NoLR training already complete; skipping training"

validate_existing_run \
    "$GPU" \
    "$NOLR" \
    "seed46_nolr"

msg "Seed46 NoLR validation complete"

run_full_experiment \
    "$GPU" \
    "$A0" \
    "seed46_noaxis0"

run_full_experiment \
    "$GPU" \
    "$A1" \
    "seed46_noaxis1"

msg "===== GPU2 SEED46 POST-TRAIN/B5 COMPLETE ====="
