#!/usr/bin/env bash
set -euo pipefail

cd /data/hdd1/yanshuo/2026code/CamylaNet

export camylanet_raw=/data/hdd1/yanshuo/datasets/CamylaNet_raw
export camylanet_preprocessed=/data/hdd1/yanshuo/preprocessed/CamylaNet
export camylanet_results=/data/hdd1/yanshuo/results/CamylaNet

TRAIN=/data/hdd1/yanshuo/conda_envs/camylanet-base/bin/nnUNetv2_train
LOG=/data/hdd1/yanshuo/logs/CamylaNet/long_runs

mkdir -p "$LOG"

echo "[$(date)] Seed46 baseline start"

CUDA_VISIBLE_DEVICES=2 \
nnUNet_n_proc_DA=4 \
OMP_NUM_THREADS=4 \
MKL_NUM_THREADS=4 \
"$TRAIN" 302 3d_fullres 4 \
-tr nnUNetTrainer_500epochs_Seed46 \
> "$LOG/seed46_baseline.log" 2>&1

echo "[$(date)] Seed46 baseline complete"

echo "[$(date)] Seed46 NoLR start"

CUDA_VISIBLE_DEVICES=2 \
nnUNet_n_proc_DA=4 \
OMP_NUM_THREADS=4 \
MKL_NUM_THREADS=4 \
"$TRAIN" 302 3d_fullres 4 \
-tr nnUNetTrainer_500epochs_NoLRMirror_Seed46 \
> "$LOG/seed46_nolr.log" 2>&1

echo "[$(date)] Seed46 NoLR complete"
