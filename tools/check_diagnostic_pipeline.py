#!/usr/bin/env python3
"""
Unified preflight checker for the CamylaNet organ-diagnostic pipeline.

Checks
------
1. hostname / current working directory
2. Python / PyTorch / CUDA visibility
3. checkpoint availability
4. preprocessed dataset availability
5. validation prediction availability
6. required diagnostic scripts
7. O1 output status
8. O2 output status
9. O3 builder prerequisites

This script does NOT run training or diagnostics.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
from pathlib import Path

import torch


DEFAULT_REPO = Path(
    "/data/hdd1/yanshuo/2026code/CamylaNet"
)

DEFAULT_CHECKPOINT = Path(
    "/data/hdd1/yanshuo/results/CamylaNet/"
    "Dataset302_AMOS22CT/"
    "nnUNetTrainer_500epochs__nnUNetPlans__3d_fullres/"
    "fold_4/checkpoint_best.pth"
)

DEFAULT_PREPROCESSED = Path(
    "/data/hdd1/yanshuo/preprocessed/CamylaNet/"
    "Dataset302_AMOS22CT/nnUNetPlans_3d_fullres"
)

DEFAULT_VALIDATION = Path(
    "/data/hdd1/yanshuo/results/CamylaNet/"
    "Dataset302_AMOS22CT/"
    "nnUNetTrainer_500epochs__nnUNetPlans__3d_fullres/"
    "fold_4/validation"
)

DEFAULT_DIAG_ROOT = Path(
    "/data/hdd1/yanshuo/results/CamylaNet/"
    "organ_diagnostic"
)


REQUIRED_SCRIPTS = [
    "evaluate_kidney_identity.py",
    "inspect_organ_diag_context.py",
    "organ_gradient_diagnostic.py",
    "analyze_organ_gradients.py",
    "organ_subset_intervention.py",
    "generate_matched_random_triplets.py",
    "analyze_subset_intervention.py",
    "analyze_pairwise_vs_higherorder.py",
    "build_o3_dataset.py",
]


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--repo",
        type=Path,
        default=DEFAULT_REPO,
    )

    p.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )

    p.add_argument(
        "--preprocessed-dir",
        type=Path,
        default=DEFAULT_PREPROCESSED,
    )

    p.add_argument(
        "--validation-dir",
        type=Path,
        default=DEFAULT_VALIDATION,
    )

    p.add_argument(
        "--diag-root",
        type=Path,
        default=DEFAULT_DIAG_ROOT,
    )

    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
    )

    return p.parse_args()


def status(ok, text):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {text}")
    return ok


def warn(text):
    print(f"[WARN] {text}")


def info(text):
    print(f"[INFO] {text}")


def check_file(path, description):
    ok = path.is_file()
    status(
        ok,
        f"{description}: {path}",
    )
    return ok


def check_dir(path, description):
    ok = path.is_dir()
    status(
        ok,
        f"{description}: {path}",
    )
    return ok


def count_files(path, pattern):
    if not path.is_dir():
        return 0

    return len(
        list(
            path.glob(pattern)
        )
    )


def check_scripts(repo):
    tools_dir = repo / "tools"

    result = {}

    print()
    print("=" * 72)
    print("SCRIPT STATUS")
    print("=" * 72)

    for name in REQUIRED_SCRIPTS:
        path = tools_dir / name

        exists = path.is_file()

        result[name] = exists

        status(
            exists,
            str(path),
        )

    return result


def check_cuda():
    print()
    print("=" * 72)
    print("CUDA STATUS")
    print("=" * 72)

    result = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_visible_devices": os.environ.get(
            "CUDA_VISIBLE_DEVICES"
        ),
        "available": False,
        "device_count": 0,
        "devices": [],
        "error": None,
    }

    info(
        f"PyTorch: {torch.__version__}"
    )

    info(
        f"PyTorch CUDA build: {torch.version.cuda}"
    )

    info(
        "CUDA_VISIBLE_DEVICES="
        + str(
            os.environ.get(
                "CUDA_VISIBLE_DEVICES"
            )
        )
    )

    try:
        available = torch.cuda.is_available()

        result["available"] = bool(
            available
        )

        status(
            available,
            "torch.cuda.is_available()",
        )

        try:
            count = (
                torch.cuda.device_count()
            )
        except Exception as e:
            count = 0

            result["error"] = repr(e)

        result[
            "device_count"
        ] = int(count)

        info(
            f"torch.cuda.device_count() = {count}"
        )

        for i in range(count):
            item = {
                "index": i,
                "name": None,
                "tensor_test": False,
                "error": None,
            }

            try:
                name = (
                    torch.cuda.get_device_name(
                        i
                    )
                )

                item[
                    "name"
                ] = name

                x = torch.randn(
                    8,
                    8,
                    device=f"cuda:{i}",
                )

                value = float(
                    x.mean().item()
                )

                item[
                    "tensor_test"
                ] = True

                status(
                    True,
                    f"GPU {i}: {name}, tensor test OK "
                    f"(mean={value:.6f})",
                )

            except Exception as e:
                item[
                    "error"
                ] = repr(e)

                status(
                    False,
                    f"GPU {i}: {repr(e)}",
                )

            result[
                "devices"
            ].append(item)

    except Exception as e:
        result[
            "error"
        ] = repr(e)

        status(
            False,
            f"CUDA initialization error: {repr(e)}",
        )

    return result


def check_checkpoint(path):
    print()
    print("=" * 72)
    print("CHECKPOINT STATUS")
    print("=" * 72)

    result = {
        "exists": False,
        "trainer_name": None,
        "current_epoch": None,
        "num_labels": None,
        "labels": None,
        "error": None,
    }

    if not check_file(
        path,
        "Checkpoint",
    ):
        return result

    result[
        "exists"
    ] = True

    try:
        ckpt = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        result[
            "trainer_name"
        ] = ckpt.get(
            "trainer_name"
        )

        result[
            "current_epoch"
        ] = ckpt.get(
            "current_epoch"
        )

        labels = (
            ckpt.get(
                "init_args",
                {},
            )
            .get(
                "dataset_json",
                {},
            )
            .get(
                "labels",
                {},
            )
        )

        result[
            "labels"
        ] = labels

        result[
            "num_labels"
        ] = len(labels)

        info(
            "trainer_name="
            + str(
                result[
                    "trainer_name"
                ]
            )
        )

        info(
            "current_epoch="
            + str(
                result[
                    "current_epoch"
                ]
            )
        )

        info(
            "num_labels="
            + str(
                result[
                    "num_labels"
                ]
            )
        )

        if labels:
            info(
                "labels="
                + json.dumps(
                    labels,
                    ensure_ascii=False,
                )
            )

    except Exception as e:
        result[
            "error"
        ] = repr(e)

        status(
            False,
            f"Checkpoint load: {repr(e)}",
        )

    return result


def check_data(
    preprocessed_dir,
    validation_dir,
):
    print()
    print("=" * 72)
    print("DATA STATUS")
    print("=" * 72)

    result = {}

    pp_ok = check_dir(
        preprocessed_dir,
        "Preprocessed directory",
    )

    npz_count = count_files(
        preprocessed_dir,
        "*.npz",
    )

    result[
        "preprocessed_exists"
    ] = pp_ok

    result[
        "npz_count"
    ] = npz_count

    info(
        f"Preprocessed .npz count = {npz_count}"
    )

    if pp_ok:
        status(
            npz_count > 0,
            "At least one preprocessed .npz exists",
        )

    val_ok = check_dir(
        validation_dir,
        "Validation directory",
    )

    nii_count = count_files(
        validation_dir,
        "*.nii.gz",
    )

    result[
        "validation_exists"
    ] = val_ok

    result[
        "validation_prediction_count"
    ] = nii_count

    info(
        f"Validation prediction count = {nii_count}"
    )

    if val_ok:
        status(
            nii_count > 0,
            "At least one validation NIfTI exists",
        )

    return result


def check_diag_outputs(
    diag_root,
):
    print()
    print("=" * 72)
    print("DIAGNOSTIC OUTPUT STATUS")
    print("=" * 72)

    o1 = (
        diag_root
        / "O1_main"
    )

    o2 = (
        diag_root
        / "O2_main"
    )

    result = {
        "O1_main": {},
        "O2_main": {},
    }

    o1_files = [
        "metadata.json",
        "organ_gradient_records.csv",
        "same_crop_aligned_primary_mean_cosine.csv",
        "same_crop_aligned_head_mean_cosine.csv",
        "same_crop_dice_primary_mean_cosine.csv",
        "same_crop_dice_head_mean_cosine.csv",
    ]

    print()
    print("O1_main")

    for name in o1_files:
        path = (
            o1
            / name
        )

        exists = path.is_file()

        result[
            "O1_main"
        ][
            name
        ] = exists

        status(
            exists,
            str(path),
        )

    o2_files = [
        "metadata.json",
        "subset_intervention_results.csv",
        "third_order_residuals.csv",
        "repeat_numerical_control.csv",
    ]

    print()
    print("O2_main")

    for name in o2_files:
        path = (
            o2
            / name
        )

        exists = path.is_file()

        result[
            "O2_main"
        ][
            name
        ] = exists

        status(
            exists,
            str(path),
        )

    return result


def summarize(
    scripts,
    cuda,
    checkpoint,
    data,
    diag,
):
    print()
    print("=" * 72)
    print("PIPELINE SUMMARY")
    print("=" * 72)

    scripts_ready = all(
        scripts.values()
    )

    base_ready = (
        checkpoint[
            "exists"
        ]
        and data[
            "preprocessed_exists"
        ]
        and data[
            "npz_count"
        ] > 0
        and data[
            "validation_exists"
        ]
        and data[
            "validation_prediction_count"
        ] > 0
    )

    gpu_ready = (
        cuda[
            "available"
        ]
        and cuda[
            "device_count"
        ] > 0
        and any(
            d[
                "tensor_test"
            ]
            for d in cuda[
                "devices"
            ]
        )
    )

    o1_ready = all(
        diag[
            "O1_main"
        ].values()
    )

    o2_ready = all(
        diag[
            "O2_main"
        ].values()
    )

    status(
        scripts_ready,
        "All diagnostic scripts present",
    )

    status(
        base_ready,
        "Checkpoint/data prerequisites ready",
    )

    status(
        gpu_ready,
        "At least one working CUDA device",
    )

    status(
        o1_ready,
        "O1_main complete",
    )

    status(
        o2_ready,
        "O2_main complete",
    )

    print()

    if (
        scripts_ready
        and base_ready
        and gpu_ready
    ):
        print(
            "NEXT ACTION: run O1 sanity/main diagnostics."
        )

    elif (
        scripts_ready
        and base_ready
        and not gpu_ready
    ):
        print(
            "NEXT ACTION: wait for GPU recovery; "
            "code/data prerequisites are ready."
        )

    elif not scripts_ready:
        print(
            "NEXT ACTION: finish missing scripts."
        )

    else:
        print(
            "NEXT ACTION: repair missing checkpoint/data paths."
        )

    return {
        "scripts_ready": scripts_ready,
        "base_ready": base_ready,
        "gpu_ready": gpu_ready,
        "o1_complete": o1_ready,
        "o2_complete": o2_ready,
    }


def main():
    args = parse_args()

    print("=" * 72)
    print("CAMYLANET ORGAN DIAGNOSTIC PREFLIGHT")
    print("=" * 72)

    print(
        "hostname:",
        socket.gethostname(),
    )

    print(
        "cwd:",
        Path.cwd(),
    )

    print(
        "python:",
        platform.python_version(),
    )

    print(
        "platform:",
        platform.platform(),
    )

    repo_ok = check_dir(
        args.repo,
        "Repository",
    )

    if not repo_ok:
        raise SystemExit(
            2
        )

    scripts = check_scripts(
        args.repo
    )

    cuda = check_cuda()

    checkpoint = check_checkpoint(
        args.checkpoint
    )

    data = check_data(
        args.preprocessed_dir,
        args.validation_dir,
    )

    diag = check_diag_outputs(
        args.diag_root
    )

    summary = summarize(
        scripts,
        cuda,
        checkpoint,
        data,
        diag,
    )

    result = {
        "hostname": (
            socket.gethostname()
        ),

        "cwd": str(
            Path.cwd()
        ),

        "repo": str(
            args.repo
        ),

        "checkpoint": (
            checkpoint
        ),

        "data": data,

        "scripts": scripts,

        "cuda": cuda,

        "diagnostic_outputs": (
            diag
        ),

        "summary": (
            summary
        ),
    }

    if (
        args.json_out
        is not None
    ):
        args.json_out.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with args.json_out.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                result,
                f,
                indent=2,
                ensure_ascii=False,
            )

        print()
        print(
            "JSON:",
            args.json_out,
        )


if __name__ == "__main__":
    main()
