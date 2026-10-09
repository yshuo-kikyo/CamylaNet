#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/hdd1/yanshuo/2026code/CamylaNet
PY=/data/hdd1/yanshuo/conda_envs/camylanet-base/bin/python

CKPT=/data/hdd1/yanshuo/results/CamylaNet/Dataset302_AMOS22CT/nnUNetTrainer_500epochs__nnUNetPlans__3d_fullres/fold_4/checkpoint_best.pth

PP=/data/hdd1/yanshuo/preprocessed/CamylaNet/Dataset302_AMOS22CT/nnUNetPlans_3d_fullres

VAL=/data/hdd1/yanshuo/results/CamylaNet/Dataset302_AMOS22CT/nnUNetTrainer_500epochs__nnUNetPlans__3d_fullres/fold_4/validation

OUT=/data/hdd1/yanshuo/results/CamylaNet/organ_diagnostic

cd "$ROOT"

echo "========================================"
echo "STEP 0: PREFLIGHT"
echo "========================================"

"$PY" tools/check_diagnostic_pipeline.py \
  --json-out "$OUT/preflight_status.json"

echo
echo "========================================"
echo "STEP 1: O1 SANITY, 1 CASE"
echo "========================================"

rm -rf "$OUT/O1_sanity_1case"

CUDA_VISIBLE_DEVICES=0 \
"$PY" tools/organ_gradient_diagnostic.py \
  --checkpoint "$CKPT" \
  --preprocessed-dir "$PP" \
  --case-list-from-dir "$VAL" \
  --num-cases 1 \
  --out-dir "$OUT/O1_sanity_1case"

echo
echo "O1 sanity outputs:"
find "$OUT/O1_sanity_1case" \
  -maxdepth 2 -type f | sort

echo
echo "========================================"
echo "STOP AFTER O1 SANITY"
echo "========================================"

echo "Inspect outputs before running O1_main."
echo
echo "Expected main files:"
echo "  metadata.json"
echo "  organ_gradient_records.csv"
echo "  same_crop_aligned_primary_mean_cosine.csv"
echo "  same_crop_aligned_head_mean_cosine.csv"
echo "  same_crop_dice_primary_mean_cosine.csv"
echo "  same_crop_dice_head_mean_cosine.csv"
