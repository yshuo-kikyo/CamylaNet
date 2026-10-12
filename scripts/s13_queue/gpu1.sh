#!/usr/bin/env bash
set -euo pipefail

cd /data/hdd1/yanshuo/2026code/CamylaNet
source scripts/s13_queue/common.sh

GPU=1

RESUME_TR=nnUNetTrainer_500epochs_NoLRMirror
RESUME_DIR="$ROOT/${RESUME_TR}__nnUNetPlans__3d_fullres/fold_4"

FULL_NOLR_TR=nnUNetTrainer_500epochs_NoLRMirror_Seed43

msg "===== GPU1 QUEUE START ====="

# ------------------------------------------------
# 1. 等当前 100->500 resumed NoLRMirror 完成
# ------------------------------------------------
wait_pattern \
"nnUNetv2_train 302 3d_fullres 4 -tr ${RESUME_TR} --c"

sleep 10

test -f "$RESUME_DIR/checkpoint_final.pth" || {
    msg "ERROR: resumed NoLRMirror checkpoint_final missing"
    exit 1
}

# ------------------------------------------------
# 2. resumed NoLRMirror final validation
# ------------------------------------------------
run_validation \
"$GPU" \
"$RESUME_TR" \
"final" \
"$LOG/no_lr_resume_final_gpu1_$(date +%Y%m%d_%H%M%S).log"

archive_validation \
"$RESUME_DIR" \
"validation_final_resume100_500"

# ------------------------------------------------
# 3. resumed NoLRMirror best validation
# ------------------------------------------------
run_validation \
"$GPU" \
"$RESUME_TR" \
"best" \
"$LOG/no_lr_resume_best_gpu1_$(date +%Y%m%d_%H%M%S).log"

archive_validation \
"$RESUME_DIR" \
"validation_best_resume100_500"

# ------------------------------------------------
# 4. 等 full NoLR Seed43 trainer
# ------------------------------------------------
wait_for_trainer_file "$FULL_NOLR_TR"

FULL_DIR="$ROOT/${FULL_NOLR_TR}__nnUNetPlans__3d_fullres/fold_4"

if [ -e "$FULL_DIR/checkpoint_latest.pth" ] || \
   [ -e "$FULL_DIR/checkpoint_final.pth" ]; then
    msg "ERROR: full NoLR Seed43 result already exists: $FULL_DIR"
    exit 1
fi

# ------------------------------------------------
# 5. Full NoLRMirror Seed43 从头训练
# ------------------------------------------------
run_fresh_training \
"$GPU" \
"$FULL_NOLR_TR" \
"$LOG/no_lr_full_seed43_gpu1_$(date +%Y%m%d_%H%M%S).log"

# ------------------------------------------------
# 6. final validation
# ------------------------------------------------
run_validation \
"$GPU" \
"$FULL_NOLR_TR" \
"final" \
"$LOG/no_lr_full_seed43_final_gpu1_$(date +%Y%m%d_%H%M%S).log"

archive_validation \
"$FULL_DIR" \
"validation_final_seed43"

# ------------------------------------------------
# 7. best validation
# ------------------------------------------------
run_validation \
"$GPU" \
"$FULL_NOLR_TR" \
"best" \
"$LOG/no_lr_full_seed43_best_gpu1_$(date +%Y%m%d_%H%M%S).log"

archive_validation \
"$FULL_DIR" \
"validation_best_seed43"

msg "===== GPU1 QUEUE COMPLETE ====="
