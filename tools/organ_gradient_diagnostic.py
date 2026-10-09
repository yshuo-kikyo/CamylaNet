#!/usr/bin/env python3
"""
O1: Organ Gradient Interaction Diagnostic

Purpose
-------
Characterize pairwise organ optimization interactions on an EXISTING
checkpoint. This script does NOT train or update the model.

Primary observation site:
    decoder.stages.4

Auxiliary control:
    decoder.seg_layers.4

Two diagnostic crop modes:
    same_crop
        One deterministic foreground-centered crop shared by all organs.

    organ_center
        Each organ uses its own GT-centered crop.
        Results are reported separately and MUST NOT be mixed with same_crop.

Important
---------
This is a PAIRWISE diagnostic only.

It does NOT claim evidence for higher-order optimization interaction.
Higher-order evidence must come from the later subset-intervention test.

Per-class diagnostic loss:
    foreground soft-Dice loss for the selected organ class.

Absent organs or near-zero gradient norms are marked invalid (NaN),
not forced to zero.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tools.organ_loss_utils import (
    training_aligned_organ_loss,
    dice_only_organ_loss,
)


EPS = 1e-12


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--preprocessed-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)

    p.add_argument(
        "--case-list-from-dir",
        type=Path,
        default=None,
        help="Optional validation prediction directory. Case IDs are read from *.nii.gz names.",
    )

    p.add_argument(
        "--num-cases",
        type=int,
        default=8,
        help="Number of deterministic cases to diagnose.",
    )

    p.add_argument(
        "--device",
        default="cuda",
    )

    p.add_argument(
        "--primary-prefix",
        default="decoder.stages.4.",
    )

    p.add_argument(
        "--head-prefix",
        default="decoder.seg_layers.4.",
    )

    p.add_argument(
        "--grad-norm-eps",
        type=float,
        default=1e-10,
    )

    return p.parse_args()


def trainer_class_from_name(name: str):
    # Current formal nnU-Net epoch variants live here.
    candidate_modules = [
        "camylanet.training.nnUNetTrainer.nnUNetTrainer_Xepochs",
        "camylanet.training.nnUNetTrainer.nnUNetTrainer",
    ]

    for modname in candidate_modules:
        mod = importlib.import_module(modname)
        if hasattr(mod, name):
            return getattr(mod, name)

    raise RuntimeError(f"Cannot locate trainer class: {name}")


def load_trainer_and_network(checkpoint: Path, device: torch.device):
    ckpt = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    trainer_name = ckpt["trainer_name"]
    init_args = ckpt["init_args"]

    Trainer = trainer_class_from_name(trainer_name)

    trainer = Trainer(
        plans=init_args["plans"],
        configuration=init_args["configuration"],
        fold=init_args["fold"],
        dataset_json=init_args["dataset_json"],
        unpack_dataset=False,
        plans_identifier=init_args.get(
            "plans_identifier",
            "nnUNetPlans",
        ),
        device=device,
    )

    trainer.initialize()

    trainer.network.load_state_dict(
        ckpt["network_weights"],
        strict=True,
    )

    trainer.network.eval()

    return ckpt, trainer


def get_patch_size(ckpt):
    init_args = ckpt["init_args"]
    plans = init_args["plans"]
    configuration = init_args["configuration"]

    patch = plans["configurations"][configuration]["patch_size"]

    return tuple(int(x) for x in patch)


def select_parameters(network, prefix):
    items = [
        (n, p)
        for n, p in network.named_parameters()
        if n.startswith(prefix) and p.requires_grad
    ]

    if not items:
        raise RuntimeError(
            f"No parameters matched prefix: {prefix}"
        )

    return items


def case_id_from_name(name: str):
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    if name.endswith(".npz"):
        return name[:-4]
    return Path(name).stem


def choose_cases(args):
    available = sorted(
        p.stem
        for p in args.preprocessed_dir.glob("*.npz")
    )

    if not available:
        raise RuntimeError(
            f"No .npz cases found in {args.preprocessed_dir}"
        )

    if args.case_list_from_dir is not None:
        requested = sorted(
            case_id_from_name(p.name)
            for p in args.case_list_from_dir.glob("*.nii.gz")
        )

        chosen = [
            x for x in requested
            if x in set(available)
        ]

        if not chosen:
            raise RuntimeError(
                "No validation case IDs matched preprocessed .npz files"
            )
    else:
        chosen = available

    return chosen[: args.num_cases]


def load_case(preprocessed_dir, case_id):
    p = preprocessed_dir / f"{case_id}.npz"

    z = np.load(p)

    if "data" not in z or "seg" not in z:
        raise RuntimeError(
            f"{p} must contain keys 'data' and 'seg'; got {z.files}"
        )

    data = np.asarray(z["data"], dtype=np.float32)
    seg = np.asarray(z["seg"])

    if seg.ndim == data.ndim:
        if seg.shape[0] != 1:
            raise RuntimeError(
                f"Unexpected seg shape: {seg.shape}"
            )
        seg = seg[0]

    seg = np.rint(seg).astype(np.int64)

    return data, seg


def centroid(mask):
    pos = np.argwhere(mask)

    if pos.size == 0:
        return None

    return np.round(
        pos.mean(axis=0)
    ).astype(int)


def crop_with_padding(
    data,
    seg,
    center,
    patch_size,
):
    spatial = np.asarray(seg.shape, dtype=int)
    patch = np.asarray(patch_size, dtype=int)
    center = np.asarray(center, dtype=int)

    start = center - patch // 2
    end = start + patch

    src_start = np.maximum(start, 0)
    src_end = np.minimum(end, spatial)

    dst_start = src_start - start
    dst_end = dst_start + (src_end - src_start)

    out_data = np.zeros(
        (data.shape[0], *patch),
        dtype=np.float32,
    )

    out_seg = np.zeros(
        tuple(patch),
        dtype=np.int64,
    )

    src_slices = tuple(
        slice(int(a), int(b))
        for a, b in zip(src_start, src_end)
    )

    dst_slices = tuple(
        slice(int(a), int(b))
        for a, b in zip(dst_start, dst_end)
    )

    out_data[(slice(None), *dst_slices)] = \
        data[(slice(None), *src_slices)]

    out_seg[dst_slices] = seg[src_slices]

    return out_data, out_seg


def select_highest_resolution_logits(output):
    if torch.is_tensor(output):
        return output

    if not isinstance(output, (list, tuple)):
        raise RuntimeError(
            f"Unexpected network output type: {type(output)}"
        )

    tensors = [
        x for x in output
        if torch.is_tensor(x)
    ]

    if not tensors:
        raise RuntimeError("No tensor logits in network output")

    return max(
        tensors,
        key=lambda x: int(np.prod(x.shape[2:])),
    )


def class_soft_dice_loss(logits, target, class_id):
    probs = F.softmax(logits, dim=1)[:, class_id]

    gt = (
        target == class_id
    ).float()

    intersection = (probs * gt).sum()

    denominator = probs.sum() + gt.sum()

    return 1.0 - (
        2.0 * intersection + 1e-5
    ) / (
        denominator + 1e-5
    )


def flatten_gradients(
    grads,
    params,
):
    pieces = []

    for g, p in zip(grads, params):
        if g is None:
            pieces.append(
                torch.zeros(
                    p.numel(),
                    device=p.device,
                    dtype=p.dtype,
                )
            )
        else:
            pieces.append(
                g.detach().reshape(-1)
            )

    return torch.cat(pieces)


def cosine_matrix(vectors, valid):
    n = len(vectors)

    out = np.full(
        (n, n),
        np.nan,
        dtype=np.float64,
    )

    for i in range(n):
        if not valid[i]:
            continue

        for j in range(n):
            if not valid[j]:
                continue

            a = vectors[i]
            b = vectors[j]

            value = F.cosine_similarity(
                a.unsqueeze(0),
                b.unsqueeze(0),
                dim=1,
                eps=EPS,
            ).item()

            out[i, j] = value

    return out


def write_matrix_csv(path, matrix, labels):
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            ["class"] + labels
        )

        for label, row in zip(labels, matrix):
            writer.writerow(
                [label] + [
                    "" if np.isnan(x)
                    else f"{x:.8f}"
                    for x in row
                ]
            )



def compute_class_gradient_bundle(
    output,
    target,
    class_id,
    primary_params,
    head_params,
    grad_norm_eps,
):
    """
    Compute two organ-gradient definitions on the SAME forward graph.

    Primary:
        training-aligned foreground class contribution
        (Dice contribution + CE contribution + deep supervision)

    Auxiliary:
        highest-resolution class-specific soft-Dice loss

    Returns gradients for:
        aligned_primary
        aligned_head
        dice_primary
        dice_head
    """

    aligned_loss, aligned_details = (
        training_aligned_organ_loss(
            output,
            target,
            class_id,
        )
    )

    dice_loss = (
        dice_only_organ_loss(
            output,
            target,
            class_id,
        )
    )

    primary_only = [
        p for _, p in primary_params
    ]

    head_only = [
        p for _, p in head_params
    ]

    all_params = (
        primary_only
        + head_only
    )

    # First backward: training-aligned loss.
    aligned_grads = torch.autograd.grad(
        aligned_loss,
        all_params,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )

    n_primary = len(
        primary_params
    )

    aligned_gp = flatten_gradients(
        aligned_grads[:n_primary],
        primary_only,
    )

    aligned_gh = flatten_gradients(
        aligned_grads[n_primary:],
        head_only,
    )

    # Second backward: Dice-only auxiliary loss.
    dice_grads = torch.autograd.grad(
        dice_loss,
        all_params,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )

    dice_gp = flatten_gradients(
        dice_grads[:n_primary],
        primary_only,
    )

    dice_gh = flatten_gradients(
        dice_grads[n_primary:],
        head_only,
    )

    aligned_primary_norm = float(
        torch.linalg.vector_norm(
            aligned_gp
        ).item()
    )

    aligned_head_norm = float(
        torch.linalg.vector_norm(
            aligned_gh
        ).item()
    )

    dice_primary_norm = float(
        torch.linalg.vector_norm(
            dice_gp
        ).item()
    )

    dice_head_norm = float(
        torch.linalg.vector_norm(
            dice_gh
        ).item()
    )

    aligned_primary_valid = bool(
        np.isfinite(
            aligned_primary_norm
        )
        and aligned_primary_norm
        > grad_norm_eps
    )

    aligned_head_valid = bool(
        np.isfinite(
            aligned_head_norm
        )
        and aligned_head_norm
        > grad_norm_eps
    )

    dice_primary_valid = bool(
        np.isfinite(
            dice_primary_norm
        )
        and dice_primary_norm
        > grad_norm_eps
    )

    dice_head_valid = bool(
        np.isfinite(
            dice_head_norm
        )
        and dice_head_norm
        > grad_norm_eps
    )

    record = {
        "aligned_loss": float(
            aligned_loss.detach().item()
        ),

        "aligned_dice_contribution": (
            aligned_details[
                "dice_contribution"
            ]
        ),

        "aligned_ce_contribution": (
            aligned_details[
                "ce_contribution"
            ]
        ),

        "dice_only_loss": float(
            dice_loss.detach().item()
        ),

        "aligned_primary_grad_norm": (
            aligned_primary_norm
        ),

        "aligned_head_grad_norm": (
            aligned_head_norm
        ),

        "dice_primary_grad_norm": (
            dice_primary_norm
        ),

        "dice_head_grad_norm": (
            dice_head_norm
        ),

        "aligned_primary_valid": (
            aligned_primary_valid
        ),

        "aligned_head_valid": (
            aligned_head_valid
        ),

        "dice_primary_valid": (
            dice_primary_valid
        ),

        "dice_head_valid": (
            dice_head_valid
        ),
    }

    vectors = {
        "aligned_primary": (
            aligned_gp
            if aligned_primary_valid
            else None
        ),

        "aligned_head": (
            aligned_gh
            if aligned_head_valid
            else None
        ),

        "dice_primary": (
            dice_gp
            if dice_primary_valid
            else None
        ),

        "dice_head": (
            dice_gh
            if dice_head_valid
            else None
        ),
    }

    valid = {
        "aligned_primary": (
            aligned_primary_valid
        ),

        "aligned_head": (
            aligned_head_valid
        ),

        "dice_primary": (
            dice_primary_valid
        ),

        "dice_head": (
            dice_head_valid
        ),
    }

    return (
        record,
        vectors,
        valid,
    )



def diagnose_crop(
    network,
    data_np,
    seg_np,
    class_ids,
    class_names,
    primary_params,
    head_params,
    grad_norm_eps,
    device,
):
    """
    Same-crop organ gradient diagnostic.

    Returns four independent pairwise-gradient views:

        aligned_primary
        aligned_head
        dice_primary
        dice_head

    aligned_*:
        foreground class contribution aligned with actual nnU-Net
        Dice + CE + Deep Supervision training objective.

    dice_*:
        highest-resolution class-specific soft-Dice control.
    """

    x = torch.from_numpy(
        data_np[None]
    ).to(device)

    y = torch.from_numpy(
        seg_np[None]
    ).to(device)

    network.zero_grad(
        set_to_none=True
    )

    # IMPORTANT:
    # Keep the full network output list because training-aligned loss
    # requires all Deep Supervision outputs.
    output = network(x)

    vector_sets = {
        "aligned_primary": [],
        "aligned_head": [],
        "dice_primary": [],
        "dice_head": [],
    }

    valid_sets = {
        "aligned_primary": [],
        "aligned_head": [],
        "dice_primary": [],
        "dice_head": [],
    }

    records = []

    for class_id, class_name in zip(
        class_ids,
        class_names,
    ):
        voxels = int(
            (y == class_id).sum().item()
        )

        present = voxels > 0

        if not present:
            for key in vector_sets:
                vector_sets[key].append(
                    None
                )
                valid_sets[key].append(
                    False
                )

            records.append({
                "class_id": class_id,
                "class_name": class_name,
                "present_crop": False,
                "voxel_count_crop": 0,

                "aligned_loss": float("nan"),
                "aligned_dice_contribution": float("nan"),
                "aligned_ce_contribution": float("nan"),
                "dice_only_loss": float("nan"),

                "aligned_primary_grad_norm": float("nan"),
                "aligned_head_grad_norm": float("nan"),
                "dice_primary_grad_norm": float("nan"),
                "dice_head_grad_norm": float("nan"),

                "aligned_primary_valid": False,
                "aligned_head_valid": False,
                "dice_primary_valid": False,
                "dice_head_valid": False,

                "invalid_reason": (
                    "class_absent_in_crop"
                ),
            })

            continue

        (
            grad_record,
            vectors,
            valid,
        ) = compute_class_gradient_bundle(
            output,
            y,
            class_id,
            primary_params,
            head_params,
            grad_norm_eps,
        )

        for key in vector_sets:
            vector_sets[key].append(
                vectors[key]
            )
            valid_sets[key].append(
                valid[key]
            )

        invalid = [
            key
            for key in valid
            if not valid[key]
        ]

        if invalid:
            reason = (
                "near_zero:"
                + ",".join(invalid)
            )
        else:
            reason = ""

        record = {
            "class_id": class_id,
            "class_name": class_name,
            "present_crop": True,
            "voxel_count_crop": voxels,
            **grad_record,
            "invalid_reason": reason,
        }

        records.append(
            record
        )

    cosine_results = {}

    for key in vector_sets:
        example = next(
            (
                v
                for v in vector_sets[key]
                if v is not None
            ),
            None,
        )

        if example is None:
            cosine_results[key] = np.full(
                (
                    len(class_ids),
                    len(class_ids),
                ),
                np.nan,
                dtype=np.float64,
            )
            continue

        storage_vectors = [
            v
            if v is not None
            else torch.zeros_like(
                example
            )
            for v in vector_sets[key]
        ]

        cosine_results[key] = cosine_matrix(
            storage_vectors,
            valid_sets[key],
        )

    del output, x, y

    return (
        records,
        cosine_results[
            "aligned_primary"
        ],
        cosine_results[
            "aligned_head"
        ],
        cosine_results[
            "dice_primary"
        ],
        cosine_results[
            "dice_head"
        ],
    )


def nanmean_stack(matrices):
    arr = np.stack(matrices, axis=0)

    with np.errstate(
        invalid="ignore",
    ):
        return np.nanmean(
            arr,
            axis=0,
        )


def count_valid_stack(matrices):
    arr = np.stack(matrices, axis=0)

    return np.sum(
        np.isfinite(arr),
        axis=0,
    )


def main():
    args = parse_args()

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(args.device)

    ckpt, trainer = load_trainer_and_network(
        args.checkpoint,
        device,
    )

    network = trainer.network

    patch_size = get_patch_size(ckpt)

    dataset_json = ckpt["init_args"]["dataset_json"]

    labels_dict = dataset_json["labels"]

    # background excluded
    entries = sorted(
        [
            (int(v), str(k))
            for k, v in labels_dict.items()
            if int(v) != 0
        ],
        key=lambda x: x[0],
    )

    class_ids = [
        x[0] for x in entries
    ]

    class_names = [
        x[1] for x in entries
    ]

    primary_params = select_parameters(
        network,
        args.primary_prefix,
    )

    head_params = select_parameters(
        network,
        args.head_prefix,
    )

    cases = choose_cases(args)

    metadata = {
        "checkpoint": str(args.checkpoint),
        "trainer_name": ckpt["trainer_name"],
        "current_epoch": ckpt.get(
            "current_epoch",
            None,
        ),
        "patch_size": patch_size,
        "cases": cases,
        "class_ids": class_ids,
        "class_names": class_names,
        "primary_prefix": args.primary_prefix,
        "head_prefix": args.head_prefix,
        "primary_parameter_names": [
            n for n, _ in primary_params
        ],
        "head_parameter_names": [
            n for n, _ in head_params
        ],
        "diagnostic_loss": (
            "per-class soft Dice loss on highest-resolution logits"
        ),
        "interpretation": (
            "PAIRWISE optimization landscape only; "
            "not evidence of higher-order interaction"
        ),
    }

    with (
        args.out_dir / "metadata.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    all_records = []

    aggregate = {
        "same_crop": {
            "aligned_primary": [],
            "aligned_head": [],
            "dice_primary": [],
            "dice_head": [],
        },

        # organ_center remains the original Dice-only
        # auxiliary diagnostic for now.
        "organ_center": {
            "primary": [],
            "head": [],
        },
    }

    for case_idx, case_id in enumerate(cases):
        print(
            f"[{case_idx+1}/{len(cases)}] {case_id}",
            flush=True,
        )

        data, seg = load_case(
            args.preprocessed_dir,
            case_id,
        )

        foreground = seg > 0
        fg_center = centroid(foreground)

        if fg_center is None:
            fg_center = (
                np.asarray(seg.shape) // 2
            )

        # -------------------------------------------------
        # same_crop
        # -------------------------------------------------
        same_data, same_seg = crop_with_padding(
            data,
            seg,
            fg_center,
            patch_size,
        )

        (
            records,
            aligned_primary_cos,
            aligned_head_cos,
            dice_primary_cos,
            dice_head_cos,
        ) = diagnose_crop(
            network,
            same_data,
            same_seg,
            class_ids,
            class_names,
            primary_params,
            head_params,
            args.grad_norm_eps,
            device,
        )

        full_counts = {
            c: int((seg == c).sum())
            for c in class_ids
        }

        for r in records:
            r.update({
                "case_id": case_id,
                "crop_mode": "same_crop",
                "full_voxel_count": full_counts[
                    r["class_id"]
                ],
            })

        all_records.extend(records)

        aggregate["same_crop"]["aligned_primary"].append(
            aligned_primary_cos
        )

        aggregate["same_crop"]["aligned_head"].append(
            aligned_head_cos
        )

        aggregate["same_crop"]["dice_primary"].append(
            dice_primary_cos
        )

        aggregate["same_crop"]["dice_head"].append(
            dice_head_cos
        )

        case_dir = (
            args.out_dir
            / "per_case"
            / case_id
            / "same_crop"
        )

        case_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.save(
            case_dir / "aligned_primary_cosine.npy",
            aligned_primary_cos,
        )

        np.save(
            case_dir / "aligned_head_cosine.npy",
            aligned_head_cos,
        )

        np.save(
            case_dir / "dice_primary_cosine.npy",
            dice_primary_cos,
        )

        np.save(
            case_dir / "dice_head_cosine.npy",
            dice_head_cos,
        )

        write_matrix_csv(
            case_dir / "aligned_primary_cosine.csv",
            aligned_primary_cos,
            class_names,
        )

        write_matrix_csv(
            case_dir / "aligned_head_cosine.csv",
            aligned_head_cos,
            class_names,
        )

        write_matrix_csv(
            case_dir / "dice_primary_cosine.csv",
            dice_primary_cos,
            class_names,
        )

        write_matrix_csv(
            case_dir / "dice_head_cosine.csv",
            dice_head_cos,
            class_names,
        )

        # -------------------------------------------------
        # organ_center
        #
        # Each class uses a different crop.
        # We compute one organ gradient from its own crop,
        # then compare vectors across organs.
        # -------------------------------------------------
        organ_primary_vectors = []
        organ_head_vectors = []
        organ_primary_valid = []
        organ_head_valid = []
        organ_records = []

        for class_id, class_name in zip(
            class_ids,
            class_names,
        ):
            center = centroid(
                seg == class_id
            )

            if center is None:
                organ_primary_vectors.append(None)
                organ_head_vectors.append(None)
                organ_primary_valid.append(False)
                organ_head_valid.append(False)

                organ_records.append({
                    "case_id": case_id,
                    "crop_mode": "organ_center",
                    "class_id": class_id,
                    "class_name": class_name,
                    "present_crop": False,
                    "voxel_count_crop": 0,
                    "full_voxel_count": 0,
                    "loss": float("nan"),
                    "primary_grad_norm": float("nan"),
                    "head_grad_norm": float("nan"),
                    "primary_valid": False,
                    "head_valid": False,
                    "invalid_reason": "class_absent_full_volume",
                })

                continue

            cdata, cseg = crop_with_padding(
                data,
                seg,
                center,
                patch_size,
            )

            (
                one_records,
                _,
                _,
            ) = diagnose_crop(
                network,
                cdata,
                cseg,
                [class_id],
                [class_name],
                primary_params,
                head_params,
                args.grad_norm_eps,
                device,
            )

            # Need actual vectors; run once more compactly.
            x = torch.from_numpy(
                cdata[None]
            ).to(device)

            y = torch.from_numpy(
                cseg[None]
            ).to(device)

            network.zero_grad(set_to_none=True)

            output = network(x)
            logits = select_highest_resolution_logits(
                output
            )

            if tuple(logits.shape[2:]) != tuple(y.shape[1:]):
                logits = F.interpolate(
                    logits,
                    size=y.shape[1:],
                    mode="trilinear",
                    align_corners=False,
                )

            loss = class_soft_dice_loss(
                logits,
                y,
                class_id,
            )

            all_params = (
                [p for _, p in primary_params]
                +
                [p for _, p in head_params]
            )

            grads = torch.autograd.grad(
                loss,
                all_params,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )

            np0 = len(primary_params)

            gp = flatten_gradients(
                grads[:np0],
                [p for _, p in primary_params],
            )

            gh = flatten_gradients(
                grads[np0:],
                [p for _, p in head_params],
            )

            gp_norm = float(
                torch.linalg.vector_norm(gp).item()
            )

            gh_norm = float(
                torch.linalg.vector_norm(gh).item()
            )

            pv = (
                np.isfinite(gp_norm)
                and gp_norm > args.grad_norm_eps
            )

            hv = (
                np.isfinite(gh_norm)
                and gh_norm > args.grad_norm_eps
            )

            organ_primary_vectors.append(
                gp if pv else None
            )

            organ_head_vectors.append(
                gh if hv else None
            )

            organ_primary_valid.append(pv)
            organ_head_valid.append(hv)

            rr = one_records[0]

            rr.update({
                "case_id": case_id,
                "crop_mode": "organ_center",
                "full_voxel_count": int(
                    (seg == class_id).sum()
                ),
            })

            organ_records.append(rr)

            del (
                x,
                y,
                output,
                logits,
                loss,
                grads,
                gp,
                gh,
            )

        pexample = next(
            (
                x for x in organ_primary_vectors
                if x is not None
            ),
            None,
        )

        hexample = next(
            (
                x for x in organ_head_vectors
                if x is not None
            ),
            None,
        )

        if pexample is not None and hexample is not None:
            pvecs = [
                x if x is not None
                else torch.zeros_like(pexample)
                for x in organ_primary_vectors
            ]

            hvecs = [
                x if x is not None
                else torch.zeros_like(hexample)
                for x in organ_head_vectors
            ]

            organ_primary_cos = cosine_matrix(
                pvecs,
                organ_primary_valid,
            )

            organ_head_cos = cosine_matrix(
                hvecs,
                organ_head_valid,
            )

            aggregate[
                "organ_center"
            ][
                "primary"
            ].append(
                organ_primary_cos
            )

            aggregate[
                "organ_center"
            ][
                "head"
            ].append(
                organ_head_cos
            )

            case_dir = (
                args.out_dir
                / "per_case"
                / case_id
                / "organ_center"
            )

            case_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            np.save(
                case_dir / "primary_cosine.npy",
                organ_primary_cos,
            )

            np.save(
                case_dir / "head_cosine.npy",
                organ_head_cos,
            )

            write_matrix_csv(
                case_dir / "primary_cosine.csv",
                organ_primary_cos,
                class_names,
            )

            write_matrix_csv(
                case_dir / "head_cosine.csv",
                organ_head_cos,
                class_names,
            )

        all_records.extend(
            organ_records
        )

        torch.cuda.empty_cache()

    # -----------------------------------------------------
    # Save scalar records
    # -----------------------------------------------------
    records_path = (
        args.out_dir
        / "organ_gradient_records.csv"
    )

    fields = [
        "case_id",
        "crop_mode",
        "class_id",
        "class_name",
        "full_voxel_count",
        "present_crop",
        "voxel_count_crop",

        # New same-crop training-aligned diagnostics
        "aligned_loss",
        "aligned_dice_contribution",
        "aligned_ce_contribution",
        "dice_only_loss",

        "aligned_primary_grad_norm",
        "aligned_head_grad_norm",
        "dice_primary_grad_norm",
        "dice_head_grad_norm",

        "aligned_primary_valid",
        "aligned_head_valid",
        "dice_primary_valid",
        "dice_head_valid",

        # Legacy organ-center Dice-only diagnostics.
        # Kept temporarily so organ-center can remain unchanged.
        "loss",
        "primary_grad_norm",
        "head_grad_norm",
        "primary_valid",
        "head_valid",

        "invalid_reason",
    ]

    with records_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()

        for r in all_records:
            writer.writerow(
                {k: r.get(k, "") for k in fields}
            )

    # -----------------------------------------------------
    # Aggregate matrices
    # -----------------------------------------------------

    aggregate_layout = {
        "same_crop": [
            "aligned_primary",
            "aligned_head",
            "dice_primary",
            "dice_head",
        ],
        "organ_center": [
            "primary",
            "head",
        ],
    }

    for mode, sites in aggregate_layout.items():
        for site in sites:
            mats = aggregate[mode][site]

            if not mats:
                continue

            mean_matrix = nanmean_stack(
                mats
            )

            count_matrix = count_valid_stack(
                mats
            )

            np.save(
                args.out_dir
                / f"{mode}_{site}_mean_cosine.npy",
                mean_matrix,
            )

            np.save(
                args.out_dir
                / f"{mode}_{site}_valid_counts.npy",
                count_matrix,
            )

            write_matrix_csv(
                args.out_dir
                / f"{mode}_{site}_mean_cosine.csv",
                mean_matrix,
                class_names,
            )

            write_matrix_csv(
                args.out_dir
                / f"{mode}_{site}_valid_counts.csv",
                count_matrix.astype(float),
                class_names,
            )

    print()
    print("O1 ORGAN GRADIENT DIAGNOSTIC COMPLETE")
    print("Cases:", cases)
    print("Patch size:", patch_size)
    print("Primary:", args.primary_prefix)
    print("Head:", args.head_prefix)
    print("Output:", args.out_dir)
    print()
    print(
        "Interpretation reminder: "
        "these are PAIRWISE gradient diagnostics only."
    )


if __name__ == "__main__":
    main()
