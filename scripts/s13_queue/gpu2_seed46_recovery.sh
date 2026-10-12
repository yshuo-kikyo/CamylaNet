#!/usr/bin/env bash
set -euo pipefail

cd /data/hdd1/yanshuo/2026code/CamylaNet
source scripts/s13_queue/common.sh

GPU=2

NOLR=nnUNetTrainer_500epochs_NoLRMirror_Seed46
A0=nnUNetTrainer_500epochs_NoMirrorAxis0_Seed46
A1=nnUNetTrainer_500epochs_NoMirrorAxis1_Seed46

NOLR_DIR="$ROOT/${NOLR}__nnUNetPlans__3d_fullres/fold_4"

msg "===== GPU2 SEED46 RECOVERY START ====="

# 1. Fresh Seed46 NoLR
if [ ! -f "$NOLR_DIR/checkpoint_final.pth" ]; then
    msg "Seed46 NoLR not started; fresh training"

    CUDA_VISIBLE_DEVICES="$GPU" \
    nnUNet_n_proc_DA=4 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    "$TRAIN" 302 3d_fullres 4 \
    -tr "$NOLR" \
    > "$LOG/seed46_nolr_fresh_$(date +%Y%m%d_%H%M%S).log" 2>&1
fi

test -f "$NOLR_DIR/checkpoint_final.pth" || {
    msg "ERROR: Seed46 NoLR checkpoint_final missing after training"
    exit 1
}

# 2. NoLR final/best
validate_existing_run \
    "$GPU" \
    "$NOLR" \
    "seed46_nolr"

# 3. B5 axis 0
run_full_experiment \
    "$GPU" \
    "$A0" \
    "seed46_noaxis0"

# 4. B5 axis 1
run_full_experiment \
    "$GPU" \
    "$A1" \
    "seed46_noaxis1"

msg "===== GPU2 SEED46 RECOVERY/B5 COMPLETE ====="
