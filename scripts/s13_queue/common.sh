#!/usr/bin/env bash
set -euo pipefail

export camylanet_raw=/data/hdd1/yanshuo/datasets/CamylaNet_raw
export camylanet_preprocessed=/data/hdd1/yanshuo/preprocessed/CamylaNet
export camylanet_results=/data/hdd1/yanshuo/results/CamylaNet

TRAIN=/data/hdd1/yanshuo/conda_envs/camylanet-base/bin/nnUNetv2_train
PY=/data/hdd1/yanshuo/conda_envs/camylanet-base/bin/python
ROOT="$camylanet_results/Dataset302_AMOS22CT"
LOG=/data/hdd1/yanshuo/logs/CamylaNet/queue

mkdir -p "$LOG"

ts() {
    date '+%Y-%m-%d %H:%M:%S'
}

msg() {
    echo "[$(ts)] $*"
}

wait_pattern() {
    local pattern="$1"

    msg "Waiting for: $pattern"

    while pgrep -af "$pattern" >/dev/null 2>&1; do
        sleep 60
    done

    msg "Finished: $pattern"
}

check_validation() {
    local dir="$1"

    test -f "$dir/summary.json" || {
        msg "ERROR: summary.json missing: $dir"
        return 1
    }

    local n
    n=$(find "$dir" -maxdepth 1 -name '*.nii.gz' | wc -l)

    msg "Validation predictions: $n"

    test "$n" -eq 40 || {
        msg "ERROR: expected 40 predictions, got $n"
        return 1
    }
}

archive_validation() {
    local run_dir="$1"
    local name="$2"

    local src="$run_dir/validation"
    local dst="$run_dir/$name"

    check_validation "$src"

    test ! -e "$dst" || {
        msg "ERROR: destination already exists: $dst"
        return 1
    }

    mv "$src" "$dst"
    msg "Archived: $dst"
}

run_validation() {
    local gpu="$1"
    local trainer="$2"
    local mode="$3"
    local logfile="$4"

    msg "Starting validation: trainer=$trainer mode=$mode GPU=$gpu"

    if [ "$mode" = "final" ]; then
        CUDA_VISIBLE_DEVICES="$gpu" \
        "$TRAIN" 302 3d_fullres 4 \
        -tr "$trainer" \
        --val \
        > "$logfile" 2>&1
    elif [ "$mode" = "best" ]; then
        CUDA_VISIBLE_DEVICES="$gpu" \
        "$TRAIN" 302 3d_fullres 4 \
        -tr "$trainer" \
        --val --val_best \
        > "$logfile" 2>&1
    else
        msg "ERROR: unknown validation mode $mode"
        return 1
    fi

    msg "Validation completed: $trainer $mode"
}

wait_for_trainer_file() {
    local trainer="$1"

    local file="camylanet/training/nnUNetTrainer/${trainer}.py"

    while [ ! -f "$file" ]; do
        msg "Waiting for trainer file: $file"
        sleep 300
    done

    msg "Trainer ready: $file"
}

run_fresh_training() {
    local gpu="$1"
    local trainer="$2"
    local logfile="$3"

    msg "Starting FRESH training: trainer=$trainer GPU=$gpu"

    CUDA_VISIBLE_DEVICES="$gpu" \
    nnUNet_n_proc_DA=4 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    "$TRAIN" 302 3d_fullres 4 \
    -tr "$trainer" \
    > "$logfile" 2>&1

    msg "Fresh training completed: $trainer"
}

run_full_experiment() {
    local gpu="$1"
    local trainer="$2"
    local tag="$3"

    local run_dir="$ROOT/${trainer}__nnUNetPlans__3d_fullres/fold_4"

    if [ -e "$run_dir/checkpoint_final.pth" ]; then
        msg "ERROR: completed result already exists: $run_dir"
        return 1
    fi

    run_fresh_training \
        "$gpu" \
        "$trainer" \
        "$LOG/${tag}_train_$(date +%Y%m%d_%H%M%S).log"

    run_validation \
        "$gpu" \
        "$trainer" \
        "final" \
        "$LOG/${tag}_final_$(date +%Y%m%d_%H%M%S).log"

    archive_validation \
        "$run_dir" \
        "validation_final_${tag}"

    run_validation \
        "$gpu" \
        "$trainer" \
        "best" \
        "$LOG/${tag}_best_$(date +%Y%m%d_%H%M%S).log"

    archive_validation \
        "$run_dir" \
        "validation_best_${tag}"

    msg "FULL EXPERIMENT COMPLETE: $trainer"
}

validate_existing_run() {
    local gpu="$1"
    local trainer="$2"
    local tag="$3"

    local run_dir="$ROOT/${trainer}__nnUNetPlans__3d_fullres/fold_4"

    test -f "$run_dir/checkpoint_final.pth" || {
        msg "ERROR: checkpoint_final missing: $run_dir"
        return 1
    }

    if [ ! -d "$run_dir/validation_final_${tag}" ]; then
        rm -rf "$run_dir/validation"

        run_validation \
            "$gpu" \
            "$trainer" \
            "final" \
            "$LOG/${tag}_final_$(date +%Y%m%d_%H%M%S).log"

        archive_validation \
            "$run_dir" \
            "validation_final_${tag}"
    else
        msg "SKIP final validation; already archived: $tag"
    fi

    if [ ! -d "$run_dir/validation_best_${tag}" ]; then
        rm -rf "$run_dir/validation"

        run_validation \
            "$gpu" \
            "$trainer" \
            "best" \
            "$LOG/${tag}_best_$(date +%Y%m%d_%H%M%S).log"

        archive_validation \
            "$run_dir" \
            "validation_best_${tag}"
    else
        msg "SKIP best validation; already archived: $tag"
    fi
}
