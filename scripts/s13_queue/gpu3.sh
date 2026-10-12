#!/usr/bin/env bash
set -euo pipefail

cd /data/hdd1/yanshuo/2026code/CamylaNet
source scripts/s13_queue/common.sh

GPU=3

REPEAT_TR=nnUNetTrainer_500epochs_Repeat
REPEAT_DIR="$ROOT/${REPEAT_TR}__nnUNetPlans__3d_fullres/fold_4"

BASE_SEED_TR=nnUNetTrainer_500epochs_Seed43

msg "===== GPU3 QUEUE START ====="

# ------------------------------------------------
# 1. 等当前 Repeat final validation 完成
# ------------------------------------------------
wait_pattern \
"nnUNetv2_train 302 3d_fullres 4 -tr ${REPEAT_TR} --val"

sleep 10

archive_validation \
"$REPEAT_DIR" \
"validation_final_repeat"

# ------------------------------------------------
# 2. Repeat best validation
# ------------------------------------------------
run_validation \
"$GPU" \
"$REPEAT_TR" \
"best" \
"$LOG/repeat_best_gpu3_$(date +%Y%m%d_%H%M%S).log"

archive_validation \
"$REPEAT_DIR" \
"validation_best_repeat"

# ------------------------------------------------
# 3. 等 Seed43 baseline trainer 准备好
# ------------------------------------------------
wait_for_trainer_file "$BASE_SEED_TR"

# 防止误续训：新 seed 必须没有旧 fold_4 结果
BASE_DIR="$ROOT/${BASE_SEED_TR}__nnUNetPlans__3d_fullres/fold_4"

if [ -e "$BASE_DIR/checkpoint_latest.pth" ] || \
   [ -e "$BASE_DIR/checkpoint_final.pth" ]; then
    msg "ERROR: Seed43 baseline result already exists: $BASE_DIR"
    exit 1
fi

# ------------------------------------------------
# 4. Baseline Seed43 从头 500ep
# ------------------------------------------------
run_fresh_training \
"$GPU" \
"$BASE_SEED_TR" \
"$LOG/baseline_seed43_gpu3_$(date +%Y%m%d_%H%M%S).log"

# ------------------------------------------------
# 5. final validation + archive
# ------------------------------------------------
run_validation \
"$GPU" \
"$BASE_SEED_TR" \
"final" \
"$LOG/baseline_seed43_final_gpu3_$(date +%Y%m%d_%H%M%S).log"

BASE_DIR="$ROOT/${BASE_SEED_TR}__nnUNetPlans__3d_fullres/fold_4"

archive_validation \
"$BASE_DIR" \
"validation_final_seed43"

# ------------------------------------------------
# 6. best validation + archive
# ------------------------------------------------
run_validation \
"$GPU" \
"$BASE_SEED_TR" \
"best" \
"$LOG/baseline_seed43_best_gpu3_$(date +%Y%m%d_%H%M%S).log"

archive_validation \
"$BASE_DIR" \
"validation_best_seed43"

msg "===== GPU3 QUEUE COMPLETE ====="
