#!/usr/bin/env python3
"""
O2: Three-organ subset intervention diagnostic.

Goal
----
Test whether the effect of jointly optimizing three organs contains a
non-additive component that cannot be reconstructed from all singleton
and pairwise interventions.

For triplet {A,B,C}, evaluate all eight subsets:

    ∅
    A
    B
    C
    AB
    AC
    BC
    ABC

For each subset S, perform ONE virtual gradient step on a selected shared
parameter block:

    theta'_S = theta - eta * sum_{c in S} grad_theta L_c

Then immediately evaluate the changed organ metrics and restore theta.

No optimizer.step() is called.
No checkpoint is modified.
No training state is saved.

Third-order residual for an outcome F:

    I_ABC =
        F(ABC)
      - F(AB) - F(AC) - F(BC)
      + F(A) + F(B) + F(C)
      - F(empty)

Here F is defined relative to the unmodified baseline, so F(empty)=0.

Important
---------
1. This script tests LOCAL non-additivity of an optimization intervention.
2. It does not by itself prove a Hypergraph is useful.
3. Raw summed gradients are intentionally used. Averaging by subset size
   would introduce a coalition-size-dependent normalization and contaminate
   the inclusion-exclusion interpretation.
4. Step-size sensitivity MUST be checked.
5. Repeated identical evaluations estimate numerical reproducibility.
6. Only SAME-CROP comparisons are used. Organ-center crops are unsuitable
   for a joint intervention because different subsets would see different
   input contexts.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tools.organ_loss_utils import (
    training_aligned_organ_loss,
)


EPS = 1e-12


# ---------------------------------------------------------
# Arguments
# ---------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--preprocessed-dir",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--case-list-from-dir",
        type=Path,
        default=None,
        help="Usually the fold validation prediction directory.",
    )

    p.add_argument(
        "--num-cases",
        type=int,
        default=4,
    )

    p.add_argument(
        "--triplet",
        nargs=3,
        type=int,
        action="append",
        required=True,
        metavar=("A", "B", "C"),
        help=(
            "Three foreground label IDs. "
            "May be specified multiple times."
        ),
    )

    p.add_argument(
        "--step-sizes",
        nargs="+",
        type=float,
        default=[1e-4, 2.5e-4, 5e-4, 1e-3],
    )

    p.add_argument(
        "--repeats",
        type=int,
        default=3,
    )

    p.add_argument(
        "--parameter-prefix",
        default="decoder.stages.4.",
        help="Shared parameter block to perturb.",
    )

    p.add_argument(
        "--device",
        default="cuda",
    )

    p.add_argument(
        "--grad-norm-eps",
        type=float,
        default=1e-10,
    )

    return p.parse_args()


# ---------------------------------------------------------
# Trainer / checkpoint
# ---------------------------------------------------------

def trainer_class_from_name(name: str):
    modules = [
        "camylanet.training.nnUNetTrainer.nnUNetTrainer_Xepochs",
        "camylanet.training.nnUNetTrainer.nnUNetTrainer",
    ]

    for module_name in modules:
        module = importlib.import_module(module_name)

        if hasattr(module, name):
            return getattr(module, name)

    raise RuntimeError(
        f"Cannot locate trainer class: {name}"
    )


def load_network(checkpoint, device):
    ckpt = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    trainer_name = ckpt["trainer_name"]
    init_args = ckpt["init_args"]

    Trainer = trainer_class_from_name(
        trainer_name
    )

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

    patch_size = plans[
        "configurations"
    ][
        configuration
    ][
        "patch_size"
    ]

    return tuple(
        int(x)
        for x in patch_size
    )


# ---------------------------------------------------------
# Cases
# ---------------------------------------------------------

def case_id_from_name(name):
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
        for p in args.preprocessed_dir.glob(
            "*.npz"
        )
    )

    if not available:
        raise RuntimeError(
            "No preprocessed .npz files found in "
            f"{args.preprocessed_dir}"
        )

    available_set = set(available)

    if args.case_list_from_dir is None:
        candidates = available
    else:
        candidates = sorted(
            case_id_from_name(p.name)
            for p in args.case_list_from_dir.glob(
                "*.nii.gz"
            )
        )

        candidates = [
            x
            for x in candidates
            if x in available_set
        ]

    if not candidates:
        raise RuntimeError(
            "No requested cases matched preprocessed cases."
        )

    return candidates[: args.num_cases]


def load_case(preprocessed_dir, case_id):
    path = (
        preprocessed_dir
        / f"{case_id}.npz"
    )

    z = np.load(path)

    if "data" not in z or "seg" not in z:
        raise RuntimeError(
            f"{path} has keys {z.files}, "
            "expected data and seg"
        )

    data = np.asarray(
        z["data"],
        dtype=np.float32,
    )

    seg = np.asarray(
        z["seg"]
    )

    if seg.ndim == data.ndim:
        if seg.shape[0] != 1:
            raise RuntimeError(
                f"Unexpected seg shape: {seg.shape}"
            )

        seg = seg[0]

    seg = np.rint(
        seg
    ).astype(
        np.int64
    )

    return data, seg


# ---------------------------------------------------------
# Crop
# ---------------------------------------------------------

def centroid(mask):
    coords = np.argwhere(mask)

    if coords.size == 0:
        return None

    return np.round(
        coords.mean(axis=0)
    ).astype(int)


def crop_with_padding(
    data,
    seg,
    center,
    patch_size,
):
    spatial = np.asarray(
        seg.shape,
        dtype=int,
    )

    patch = np.asarray(
        patch_size,
        dtype=int,
    )

    center = np.asarray(
        center,
        dtype=int,
    )

    start = center - patch // 2
    end = start + patch

    src_start = np.maximum(
        start,
        0,
    )

    src_end = np.minimum(
        end,
        spatial,
    )

    dst_start = (
        src_start - start
    )

    dst_end = (
        dst_start
        + src_end
        - src_start
    )

    out_data = np.zeros(
        (
            data.shape[0],
            *patch,
        ),
        dtype=np.float32,
    )

    out_seg = np.zeros(
        tuple(patch),
        dtype=np.int64,
    )

    src = tuple(
        slice(int(a), int(b))
        for a, b in zip(
            src_start,
            src_end,
        )
    )

    dst = tuple(
        slice(int(a), int(b))
        for a, b in zip(
            dst_start,
            dst_end,
        )
    )

    out_data[
        (slice(None), *dst)
    ] = data[
        (slice(None), *src)
    ]

    out_seg[
        dst
    ] = seg[
        src
    ]

    return out_data, out_seg


# ---------------------------------------------------------
# Network output / loss / metric
# ---------------------------------------------------------

def select_highest_resolution_logits(
    output,
):
    if torch.is_tensor(output):
        return output

    if not isinstance(
        output,
        (list, tuple),
    ):
        raise RuntimeError(
            f"Unexpected network output: {type(output)}"
        )

    tensors = [
        x
        for x in output
        if torch.is_tensor(x)
    ]

    if not tensors:
        raise RuntimeError(
            "No tensor logits in output."
        )

    return max(
        tensors,
        key=lambda x: int(
            np.prod(
                x.shape[2:]
            )
        ),
    )


def ensure_logits_size(
    logits,
    target,
):
    if tuple(
        logits.shape[2:]
    ) == tuple(
        target.shape[1:]
    ):
        return logits

    return F.interpolate(
        logits,
        size=target.shape[1:],
        mode="trilinear",
        align_corners=False,
    )


def class_soft_dice(
    logits,
    target,
    class_id,
):
    probs = F.softmax(
        logits,
        dim=1,
    )[:, class_id]

    gt = (
        target == class_id
    ).float()

    intersection = (
        probs * gt
    ).sum()

    denominator = (
        probs.sum()
        + gt.sum()
    )

    return (
        2.0 * intersection + 1e-5
    ) / (
        denominator + 1e-5
    )


def class_soft_dice_loss(
    logits,
    target,
    class_id,
):
    return (
        1.0
        - class_soft_dice(
            logits,
            target,
            class_id,
        )
    )


def class_hard_dice(
    logits,
    target,
    class_id,
):
    pred = torch.argmax(
        logits,
        dim=1,
    )

    p = pred == class_id
    g = target == class_id

    npred = int(
        p.sum().item()
    )

    ngt = int(
        g.sum().item()
    )

    if npred + ngt == 0:
        return float("nan")

    intersection = int(
        torch.logical_and(
            p,
            g,
        ).sum().item()
    )

    return (
        2.0 * intersection
        / (npred + ngt)
    )


# ---------------------------------------------------------
# Parameters
# ---------------------------------------------------------

def select_parameters(
    network,
    prefix,
):
    selected = [
        (name, p)
        for name, p
        in network.named_parameters()
        if (
            name.startswith(prefix)
            and p.requires_grad
        )
    ]

    if not selected:
        raise RuntimeError(
            "No parameters matched "
            f"{prefix}"
        )

    return selected


def clone_parameters(
    selected,
):
    return [
        p.detach().clone()
        for _, p in selected
    ]


@torch.no_grad()
def restore_parameters(
    selected,
    backups,
):
    for (_, p), old in zip(
        selected,
        backups,
    ):
        p.copy_(old)


@torch.no_grad()
def apply_virtual_step(
    selected,
    gradients,
    step_size,
):
    sq = 0.0

    for (_, p), g in zip(
        selected,
        gradients,
    ):
        if g is None:
            continue

        p.add_(
            g,
            alpha=-step_size,
        )

        sq += float(
            torch.sum(
                g.detach().float() ** 2
            ).item()
        )

    grad_norm = sq ** 0.5
    update_norm = (
        step_size
        * grad_norm
    )

    return grad_norm, update_norm


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

@torch.no_grad()
def evaluate_classes(
    network,
    x,
    y,
    class_ids,
):
    output = network(x)

    logits = (
        select_highest_resolution_logits(
            output
        )
    )

    logits = ensure_logits_size(
        logits,
        y,
    )

    results = {}

    for c in class_ids:
        present = bool(
            (y == c).any().item()
        )

        if not present:
            results[c] = {
                "present": False,
                "soft_dice": float("nan"),
                "loss": float("nan"),
                "hard_dice": float("nan"),
            }

            continue

        soft = float(
            class_soft_dice(
                logits,
                y,
                c,
            ).item()
        )

        hard = float(
            class_hard_dice(
                logits,
                y,
                c,
            )
        )

        results[c] = {
            "present": True,
            "soft_dice": soft,
            "loss": 1.0 - soft,
            "hard_dice": hard,
        }

    return results



def subset_gradient(
    network,
    x,
    y,
    subset,
    selected_params,
):
    """
    Coalition gradient aligned with the actual nnU-Net objective.

    For organ set S:

        L_S = sum_{c in S} L_c^aligned

    where each L_c^aligned contains:
        - foreground Dice class contribution
        - multiclass CE class contribution
        - actual Deep Supervision weighting

    SUM is intentional. Do not normalize by |S|.
    """

    network.zero_grad(
        set_to_none=True
    )

    output = network(x)

    losses = []
    details = []

    for c in subset:
        if not bool(
            (y == c).any().item()
        ):
            raise RuntimeError(
                f"class {c} absent in intervention crop"
            )

        loss_c, detail_c = (
            training_aligned_organ_loss(
                output,
                y,
                c,
            )
        )

        losses.append(
            loss_c
        )

        details.append(
            detail_c
        )

    if not losses:
        return (
            [None] * len(selected_params),
            0.0,
            [],
        )

    coalition_loss = torch.stack(
        losses
    ).sum()

    params = [
        p
        for _, p
        in selected_params
    ]

    grads = torch.autograd.grad(
        coalition_loss,
        params,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )

    return (
        grads,
        float(
            coalition_loss.detach().item()
        ),
        details,
    )



# ---------------------------------------------------------
# Eight subsets
# ---------------------------------------------------------

def all_subsets(triplet):
    a, b, c = triplet

    return [
        (),
        (a,),
        (b,),
        (c,),
        (a, b),
        (a, c),
        (b, c),
        (a, b, c),
    ]


def subset_name(subset):
    if not subset:
        return "empty"

    return "-".join(
        str(x)
        for x in subset
    )


# ---------------------------------------------------------
# Third-order residual
# ---------------------------------------------------------

def third_order_residual(values, triplet):
    a, b, c = triplet

    return (
        values[(a, b, c)]
        - values[(a, b)]
        - values[(a, c)]
        - values[(b, c)]
        + values[(a,)]
        + values[(b,)]
        + values[(c,)]
        - values[()]
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    args = parse_args()

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.device.startswith(
        "cuda"
    ):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA unavailable. "
                "Abort before model initialization."
            )

    device = torch.device(
        args.device
    )

    ckpt, trainer = load_network(
        args.checkpoint,
        device,
    )

    network = trainer.network

    patch_size = get_patch_size(
        ckpt
    )

    selected_params = (
        select_parameters(
            network,
            args.parameter_prefix,
        )
    )

    parameter_norm_sq = 0.0

    for _, p in selected_params:
        parameter_norm_sq += float(
            torch.sum(
                p.detach().float() ** 2
            ).item()
        )

    parameter_norm = (
        parameter_norm_sq ** 0.5
    )

    if not np.isfinite(parameter_norm) or parameter_norm <= 0:
        raise RuntimeError(
            f"Invalid selected-parameter norm: {parameter_norm}"
        )

    parameter_backup = (
        clone_parameters(
            selected_params
        )
    )

    labels = ckpt[
        "init_args"
    ][
        "dataset_json"
    ][
        "labels"
    ]

    id_to_name = {
        int(v): str(k)
        for k, v in labels.items()
    }

    cases = choose_cases(
        args
    )

    triplets = [
        tuple(sorted(x))
        for x in args.triplet
    ]

    for t in triplets:
        if len(set(t)) != 3:
            raise RuntimeError(
                f"Triplet has repeated labels: {t}"
            )

        for c in t:
            if c == 0:
                raise RuntimeError(
                    "Background cannot be used "
                    "as an intervention organ."
                )

            if c not in id_to_name:
                raise RuntimeError(
                    f"Unknown class id: {c}"
                )

    metadata = {
        "checkpoint": str(
            args.checkpoint
        ),
        "trainer_name": ckpt[
            "trainer_name"
        ],
        "checkpoint_epoch": ckpt.get(
            "current_epoch"
        ),
        "patch_size": patch_size,
        "parameter_prefix": (
            args.parameter_prefix
        ),
        "parameter_names": [
            n
            for n, _
            in selected_params
        ],
        "parameter_norm": (
            parameter_norm
        ),
        "cases": cases,
        "triplets": [
            {
                "ids": list(t),
                "names": [
                    id_to_name[c]
                    for c in t
                ],
            }
            for t in triplets
        ],
        "step_sizes": (
            args.step_sizes
        ),
        "repeats": args.repeats,
        "intervention_gradient": (
            "sum of per-organ training-aligned "
            "foreground class-contribution gradients"
        ),
        "intervention_optimizer": (
            "single virtual raw gradient step; "
            "no momentum, no weight decay"
        ),
        "crop_mode": (
            "single deterministic foreground-centered "
            "crop shared across all 8 subsets"
        ),
        "warning": (
            "Third-order residual is a local "
            "non-additivity diagnostic, not by itself "
            "evidence that a hypergraph method is useful."
        ),
    }

    with (
        args.out_dir
        / "metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    subset_rows = []
    residual_rows = []

    for case_index, case_id in enumerate(
        cases
    ):
        print(
            f"[CASE {case_index + 1}/{len(cases)}] "
            f"{case_id}",
            flush=True,
        )

        data, seg = load_case(
            args.preprocessed_dir,
            case_id,
        )

        fg_center = centroid(
            seg > 0
        )

        if fg_center is None:
            print(
                "  SKIP: no foreground"
            )
            continue

        crop_data, crop_seg = (
            crop_with_padding(
                data,
                seg,
                fg_center,
                patch_size,
            )
        )

        x = torch.from_numpy(
            crop_data[None]
        ).to(
            device
        )

        y = torch.from_numpy(
            crop_seg[None]
        ).to(
            device
        )

        # baseline evaluation contains all foreground labels
        eval_ids = sorted(
            [
                int(v)
                for k, v
                in labels.items()
                if int(v) != 0
            ]
        )

        full_voxel_counts = {
            c: int(
                (seg == c).sum()
            )
            for c in eval_ids
        }

        crop_voxel_counts = {
            c: int(
                (crop_seg == c).sum()
            )
            for c in eval_ids
        }

        crop_coverage = {
            c: (
                crop_voxel_counts[c]
                / full_voxel_counts[c]
                if full_voxel_counts[c] > 0
                else float("nan")
            )
            for c in eval_ids
        }

        restore_parameters(
            selected_params,
            parameter_backup,
        )

        full_voxel_counts = {
            c: int((seg == c).sum())
            for c in eval_ids
        }

        crop_voxel_counts = {
            c: int((crop_seg == c).sum())
            for c in eval_ids
        }

        crop_coverage = {
            c: (
                crop_voxel_counts[c]
                / full_voxel_counts[c]
                if full_voxel_counts[c] > 0
                else float("nan")
            )
            for c in eval_ids
        }

        baseline = evaluate_classes(
            network,
            x,
            y,
            eval_ids,
        )

        for triplet in triplets:
            triplet_present = all(
                bool(
                    (y == c).any().item()
                )
                for c in triplet
            )

            if not triplet_present:
                missing = [
                    c
                    for c in triplet
                    if not bool(
                        (y == c).any().item()
                    )
                ]

                print(
                    "  SKIP triplet",
                    triplet,
                    "missing in crop:",
                    missing,
                )

                continue

            triplet_name = "__".join(
                id_to_name[c].replace(
                    " ",
                    "_",
                )
                for c in triplet
            )

            print(
                "  triplet:",
                triplet,
                triplet_name,
            )

            subsets = all_subsets(
                triplet
            )

            for step_size in args.step_sizes:
                for repeat in range(
                    args.repeats
                ):
                    # Maps:
                    # outcome organ ->
                    # subset -> delta metric
                    soft_delta = {
                        target: {}
                        for target in eval_ids
                    }

                    loss_delta = {
                        target: {}
                        for target in eval_ids
                    }

                    hard_delta = {
                        target: {}
                        for target in eval_ids
                    }

                    update_norms = {}

                    for subset in subsets:
                        # Always return to exact original params.
                        restore_parameters(
                            selected_params,
                            parameter_backup,
                        )

                        if subset:
                            grads, coalition_loss, coalition_details = (
                                subset_gradient(
                                    network,
                                    x,
                                    y,
                                    subset,
                                    selected_params,
                                )
                            )

                            grad_sq = 0.0

                            for g in grads:
                                if g is None:
                                    continue

                                grad_sq += float(
                                    torch.sum(
                                        g.detach().float()
                                        ** 2
                                    ).item()
                                )

                            grad_norm = (
                                grad_sq ** 0.5
                            )

                            if (
                                not np.isfinite(
                                    grad_norm
                                )
                                or grad_norm
                                <= args.grad_norm_eps
                            ):
                                raise RuntimeError(
                                    "Invalid intervention "
                                    f"gradient norm for "
                                    f"{case_id}, "
                                    f"{triplet}, "
                                    f"{subset}: "
                                    f"{grad_norm}"
                                )

                            _, update_norm = (
                                apply_virtual_step(
                                    selected_params,
                                    grads,
                                    step_size,
                                )
                            )

                        else:
                            coalition_loss = 0.0
                            coalition_details = []
                            grad_norm = 0.0
                            update_norm = 0.0

                        post = evaluate_classes(
                            network,
                            x,
                            y,
                            eval_ids,
                        )

                        update_norms[
                            subset
                        ] = update_norm

                        for target in eval_ids:
                            b = baseline[target]
                            q = post[target]

                            if not (
                                b["present"]
                                and q["present"]
                            ):
                                soft_d = float(
                                    "nan"
                                )
                                loss_d = float(
                                    "nan"
                                )
                                hard_d = float(
                                    "nan"
                                )
                            else:
                                soft_d = (
                                    q["soft_dice"]
                                    - b["soft_dice"]
                                )

                                loss_d = (
                                    q["loss"]
                                    - b["loss"]
                                )

                                hard_d = (
                                    q["hard_dice"]
                                    - b["hard_dice"]
                                )

                            soft_delta[
                                target
                            ][
                                subset
                            ] = soft_d

                            loss_delta[
                                target
                            ][
                                subset
                            ] = loss_d

                            hard_delta[
                                target
                            ][
                                subset
                            ] = hard_d

                            subset_rows.append({
                                "case_id": (
                                    case_id
                                ),
                                "triplet": (
                                    "-".join(
                                        map(
                                            str,
                                            triplet,
                                        )
                                    )
                                ),
                                "triplet_names": (
                                    "|".join(
                                        id_to_name[c]
                                        for c
                                        in triplet
                                    )
                                ),
                                "step_size": (
                                    step_size
                                ),
                                "repeat": (
                                    repeat
                                ),
                                "subset": (
                                    subset_name(
                                        subset
                                    )
                                ),
                                "subset_size": (
                                    len(subset)
                                ),
                                "target_class": (
                                    target
                                ),
                                "target_name": (
                                    id_to_name[
                                        target
                                    ]
                                ),
                                "target_present": (
                                    b[
                                        "present"
                                    ]
                                ),

                                "target_full_voxels": (
                                    full_voxel_counts[target]
                                ),

                                "target_crop_voxels": (
                                    crop_voxel_counts[target]
                                ),

                                "target_crop_coverage": (
                                    crop_coverage[target]
                                ),
                                "baseline_soft_dice": (
                                    b[
                                        "soft_dice"
                                    ]
                                ),
                                "baseline_hard_dice": (
                                    b[
                                        "hard_dice"
                                    ]
                                ),
                                "delta_soft_dice": (
                                    soft_d
                                ),
                                "delta_loss": (
                                    loss_d
                                ),
                                "delta_hard_dice": (
                                    hard_d
                                ),
                                "coalition_loss": (
                                    coalition_loss
                                ),
                                "coalition_grad_norm": (
                                    grad_norm
                                ),
                                "update_norm": (
                                    update_norm
                                ),

                                "parameter_norm": (
                                    parameter_norm
                                ),

                                "relative_update_norm": (
                                    update_norm / parameter_norm
                                ),
                            })

                        # Critical:
                        # restore immediately after evaluation.
                        restore_parameters(
                            selected_params,
                            parameter_backup,
                        )

                    # -------------------------------------
                    # Inclusion-exclusion residuals
                    # -------------------------------------

                    for target in eval_ids:
                        if not baseline[
                            target
                        ][
                            "present"
                        ]:
                            continue

                        values_soft = (
                            soft_delta[
                                target
                            ]
                        )

                        values_loss = (
                            loss_delta[
                                target
                            ]
                        )

                        values_hard = (
                            hard_delta[
                                target
                            ]
                        )

                        if any(
                            not np.isfinite(
                                values_soft[s]
                            )
                            for s in subsets
                        ):
                            continue

                        i3_soft = (
                            third_order_residual(
                                values_soft,
                                triplet,
                            )
                        )

                        i3_loss = (
                            third_order_residual(
                                values_loss,
                                triplet,
                            )
                        )

                        if all(
                            np.isfinite(
                                values_hard[s]
                            )
                            for s in subsets
                        ):
                            i3_hard = (
                                third_order_residual(
                                    values_hard,
                                    triplet,
                                )
                            )
                        else:
                            i3_hard = float(
                                "nan"
                            )

                        residual_rows.append({
                            "case_id": case_id,
                            "triplet": (
                                "-".join(
                                    map(
                                        str,
                                        triplet,
                                    )
                                )
                            ),
                            "triplet_names": (
                                "|".join(
                                    id_to_name[c]
                                    for c
                                    in triplet
                                )
                            ),
                            "step_size": (
                                step_size
                            ),
                            "repeat": repeat,
                            "target_class": (
                                target
                            ),
                            "target_name": (
                                id_to_name[
                                    target
                                ]
                            ),
                            "target_in_triplet": (
                                target
                                in triplet
                            ),
                            "i3_soft_dice": (
                                i3_soft
                            ),
                            "i3_loss": (
                                i3_loss
                            ),
                            "i3_hard_dice": (
                                i3_hard
                            ),
                            "abc_update_norm": (
                                update_norms[
                                    tuple(
                                        triplet
                                    )
                                ]
                            ),
                        })

        restore_parameters(
            selected_params,
            parameter_backup,
        )

        del x, y

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # -----------------------------------------------------
    # Save subset outcomes
    # -----------------------------------------------------

    subset_path = (
        args.out_dir
        / "subset_intervention_results.csv"
    )

    if subset_rows:
        fields = list(
            subset_rows[0].keys()
        )

        with subset_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fields,
            )

            writer.writeheader()
            writer.writerows(
                subset_rows
            )

    # -----------------------------------------------------
    # Save third-order residuals
    # -----------------------------------------------------

    residual_path = (
        args.out_dir
        / "third_order_residuals.csv"
    )

    if residual_rows:
        fields = list(
            residual_rows[0].keys()
        )

        with residual_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fields,
            )

            writer.writeheader()
            writer.writerows(
                residual_rows
            )

    # -----------------------------------------------------
    # Lightweight deterministic-repeat summary
    # -----------------------------------------------------

    repeat_summary = []

    if residual_rows:
        keys = sorted(
            set(
                (
                    r["case_id"],
                    r["triplet"],
                    r["step_size"],
                    r["target_class"],
                )
                for r in residual_rows
            )
        )

        for key in keys:
            values = [
                r["i3_soft_dice"]
                for r in residual_rows
                if (
                    r["case_id"],
                    r["triplet"],
                    r["step_size"],
                    r["target_class"],
                ) == key
            ]

            arr = np.asarray(
                values,
                dtype=float,
            )

            repeat_summary.append({
                "case_id": key[0],
                "triplet": key[1],
                "step_size": key[2],
                "target_class": key[3],
                "num_repeats": (
                    len(arr)
                ),
                "i3_soft_mean": (
                    float(
                        np.nanmean(
                            arr
                        )
                    )
                ),
                "i3_soft_std_repeat": (
                    float(
                        np.nanstd(
                            arr
                        )
                    )
                ),
                "i3_soft_max_minus_min": (
                    float(
                        np.nanmax(arr)
                        - np.nanmin(arr)
                    )
                ),
            })

        repeat_path = (
            args.out_dir
            / "repeat_numerical_control.csv"
        )

        fields = list(
            repeat_summary[0].keys()
        )

        with repeat_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fields,
            )

            writer.writeheader()
            writer.writerows(
                repeat_summary
            )

    restore_parameters(
        selected_params,
        parameter_backup,
    )

    print()
    print(
        "O2 SUBSET INTERVENTION COMPLETE"
    )
    print(
        "subset results:",
        subset_path,
    )
    print(
        "third-order residuals:",
        residual_path,
    )

    print()
    print(
        "Interpretation:"
    )
    print(
        "1. First compare |I3| with repeated-evaluation "
        "numerical variation."
    )
    print(
        "2. Then inspect consistency across cases."
    )
    print(
        "3. Then inspect sensitivity across step sizes."
    )
    print(
        "4. Only after that compare structured triplets "
        "with matched random triplets."
    )


if __name__ == "__main__":
    main()
