#!/usr/bin/env python3
"""
Shared organ-specific diagnostic losses for O1/O2.

Two definitions
---------------

1. training_aligned_organ_loss
   Foreground-class contribution aligned with nnU-Net training objective:

       class Dice contribution
       + class CE contribution
       + actual Deep Supervision weighting

   Dice:
       nnU-Net foreground Dice loss is the mean over foreground classes.
       For C foreground classes, organ c contributes:

           L_dice,c = - Dice_c / C

   CE:
       nn.CrossEntropyLoss uses mean reduction over all voxels.
       Organ c contributes:

           L_ce,c =
               sum_{v: y_v=c} CE_v / N_all_voxels

       Note: background CE is deliberately NOT copied into every organ.

2. dice_only_organ_loss
   Highest-resolution class-specific soft-Dice loss.
   This is retained as an auxiliary diagnostic control.

The "training aligned" loss is a foreground organ contribution to the
training objective, not an independent exact organ loss.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


TensorOrSequence = Union[
    torch.Tensor,
    Sequence[torch.Tensor],
]


def as_output_list(
    output: TensorOrSequence,
) -> List[torch.Tensor]:

    if torch.is_tensor(output):
        return [output]

    if not isinstance(
        output,
        (list, tuple),
    ):
        raise TypeError(
            f"Unexpected network output type: {type(output)}"
        )

    outputs = [
        x
        for x in output
        if torch.is_tensor(x)
    ]

    if not outputs:
        raise RuntimeError(
            "Network output contains no tensors"
        )

    return outputs


def normalize_target_shape(
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Return target as [B, 1, D, H, W].
    """

    if target.ndim == 4:
        target = target[:, None]

    if target.ndim != 5:
        raise RuntimeError(
            f"Expected 3D target [B,1,D,H,W] "
            f"or [B,D,H,W], got {tuple(target.shape)}"
        )

    if target.shape[1] != 1:
        raise RuntimeError(
            f"Expected target channel dimension = 1, "
            f"got {tuple(target.shape)}"
        )

    return target


def target_for_logits(
    target: torch.Tensor,
    logits: torch.Tensor,
) -> torch.Tensor:
    """
    Match target to a deep-supervision output.

    nnU-Net uses segmentation resize with order=0.
    Nearest interpolation reproduces categorical nearest-neighbor
    downsampling for this diagnostic.
    """

    target = normalize_target_shape(
        target
    )

    if tuple(
        target.shape[2:]
    ) == tuple(
        logits.shape[2:]
    ):
        return target

    resized = F.interpolate(
        target.float(),
        size=logits.shape[2:],
        mode="nearest",
    )

    return resized.to(
        dtype=target.dtype
    )


def deep_supervision_weights(
    num_outputs: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Match nnUNetTrainer._build_loss():

        w_i = 1 / 2**i
        last weight = 0
        normalize to sum 1

    For a single output, weight = 1.
    """

    if num_outputs < 1:
        raise ValueError(
            "num_outputs must be >= 1"
        )

    if num_outputs == 1:
        return torch.ones(
            1,
            device=device,
            dtype=dtype,
        )

    w = torch.tensor(
        [
            1.0 / (2.0 ** i)
            for i in range(
                num_outputs
            )
        ],
        device=device,
        dtype=dtype,
    )

    w[-1] = 0.0

    denom = w.sum()

    if denom <= 0:
        raise RuntimeError(
            "Invalid deep-supervision weights"
        )

    return w / denom


def foreground_class_count(
    logits: torch.Tensor,
) -> int:
    """
    nnU-Net has one background channel + foreground classes.
    """

    c = int(
        logits.shape[1]
    )

    if c < 2:
        raise RuntimeError(
            f"Need background + foreground classes, got C={c}"
        )

    return c - 1


def class_soft_dice_value(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_id: int,
    smooth: float = 1e-5,
) -> torch.Tensor:
    """
    Match MemoryEfficientSoftDiceLoss class Dice definition
    for batch size 1 diagnostic usage.
    """

    target = target_for_logits(
        target,
        logits,
    )

    probs = F.softmax(
        logits,
        dim=1,
    )

    p = probs[
        :,
        class_id
    ]

    g = (
        target[:, 0]
        == class_id
    ).to(
        dtype=p.dtype
    )

    axes = tuple(
        range(
            1,
            p.ndim,
        )
    )

    intersect = (
        p * g
    ).sum(
        dim=axes
    )

    sum_pred = p.sum(
        dim=axes
    )

    sum_gt = g.sum(
        dim=axes
    )

    dice = (
        2.0 * intersect
        + smooth
    ) / torch.clamp(
        sum_pred
        + sum_gt
        + smooth,
        min=1e-8,
    )

    # diagnostic batches are normally B=1;
    # mean keeps behavior defined if B>1.
    return dice.mean()


def class_dice_training_contribution(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_id: int,
    smooth: float = 1e-5,
) -> torch.Tensor:
    """
    Exact class contribution to foreground-mean Dice loss:

        DC_loss = - mean_c Dice_c

    therefore:

        contribution_c = -Dice_c / C_fg
    """

    if class_id <= 0:
        raise ValueError(
            "training-aligned organ loss expects a foreground class"
        )

    num_fg = foreground_class_count(
        logits
    )

    dice_c = class_soft_dice_value(
        logits,
        target,
        class_id,
        smooth=smooth,
    )

    return (
        -dice_c
        / float(num_fg)
    )


def class_ce_training_contribution(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_id: int,
) -> torch.Tensor:
    """
    Exact contribution of GT class c voxels to mean multiclass CE:

        total CE =
            (1/N) sum_v CE_v

        class contribution =
            (1/N) sum_{v:y_v=c} CE_v

    Sum over all classes INCLUDING background recovers total CE.
    """

    target = target_for_logits(
        target,
        logits,
    )

    y = target[
        :,
        0
    ].long()

    ce_voxel = F.cross_entropy(
        logits,
        y,
        reduction="none",
    )

    mask = (
        y
        == class_id
    )

    # Keep denominator identical to ordinary CE mean:
    # all voxels, NOT number of class-c voxels.
    return (
        ce_voxel
        * mask.to(
            ce_voxel.dtype
        )
    ).sum() / ce_voxel.numel()


def training_aligned_single_output(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_id: int,
    smooth: float = 1e-5,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Returns:
        total class contribution,
        Dice contribution,
        CE contribution
    """

    dice_part = (
        class_dice_training_contribution(
            logits,
            target,
            class_id,
            smooth=smooth,
        )
    )

    ce_part = (
        class_ce_training_contribution(
            logits,
            target,
            class_id,
        )
    )

    total = (
        dice_part
        + ce_part
    )

    return (
        total,
        dice_part,
        ce_part,
    )


def training_aligned_organ_loss(
    output: TensorOrSequence,
    target: torch.Tensor,
    class_id: int,
    smooth: float = 1e-5,
) -> Tuple[
    torch.Tensor,
    dict,
]:
    """
    Deep-supervision weighted foreground-class contribution.

    Returns
    -------
    loss:
        scalar tensor used for gradient computation

    details:
        detached diagnostic metadata
    """

    outputs = as_output_list(
        output
    )

    weights = (
        deep_supervision_weights(
            len(outputs),
            outputs[0].device,
            outputs[0].dtype,
        )
    )

    total = outputs[0].sum() * 0.0

    dice_total = (
        outputs[0].sum()
        * 0.0
    )

    ce_total = (
        outputs[0].sum()
        * 0.0
    )

    per_output = []

    for i, (
        logits,
        weight,
    ) in enumerate(
        zip(
            outputs,
            weights,
        )
    ):
        if float(
            weight.detach().cpu()
        ) == 0.0:
            per_output.append({
                "output_index": i,
                "weight": 0.0,
                "skipped": True,
            })
            continue

        (
            component,
            dice_part,
            ce_part,
        ) = (
            training_aligned_single_output(
                logits,
                target,
                class_id,
                smooth=smooth,
            )
        )

        total = (
            total
            + weight
            * component
        )

        dice_total = (
            dice_total
            + weight
            * dice_part
        )

        ce_total = (
            ce_total
            + weight
            * ce_part
        )

        per_output.append({
            "output_index": i,
            "weight": float(
                weight.detach().cpu()
            ),
            "shape": list(
                logits.shape
            ),
            "total": float(
                component.detach().cpu()
            ),
            "dice_part": float(
                dice_part.detach().cpu()
            ),
            "ce_part": float(
                ce_part.detach().cpu()
            ),
            "skipped": False,
        })

    details = {
        "class_id": int(
            class_id
        ),
        "loss": float(
            total.detach().cpu()
        ),
        "dice_contribution": float(
            dice_total.detach().cpu()
        ),
        "ce_contribution": float(
            ce_total.detach().cpu()
        ),
        "deep_supervision_weights": [
            float(x)
            for x in weights.detach().cpu()
        ],
        "per_output": (
            per_output
        ),
    }

    return total, details


def highest_resolution_output(
    output: TensorOrSequence,
) -> torch.Tensor:

    outputs = as_output_list(
        output
    )

    return max(
        outputs,
        key=lambda x: int(
            np.prod(
                x.shape[2:]
            )
        ),
    )


def dice_only_organ_loss(
    output: TensorOrSequence,
    target: torch.Tensor,
    class_id: int,
    smooth: float = 1e-5,
) -> torch.Tensor:
    """
    Auxiliary diagnostic:
    highest-resolution class-specific Dice loss.

        1 - Dice_c

    This is NOT the actual nnU-Net training loss.
    """

    logits = (
        highest_resolution_output(
            output
        )
    )

    dice_c = (
        class_soft_dice_value(
            logits,
            target,
            class_id,
            smooth=smooth,
        )
    )

    return (
        1.0
        - dice_c
    )


def class_present(
    target: torch.Tensor,
    class_id: int,
) -> bool:

    target = normalize_target_shape(
        target
    )

    return bool(
        (
            target
            == class_id
        ).any().item()
    )


def validate_decomposition(
    output: TensorOrSequence,
    target: torch.Tensor,
    smooth: float = 1e-5,
) -> dict:
    """
    Sanity check.

    At each DS level:
      sum of foreground Dice class contributions
      must equal nnU-Net foreground Dice loss.

      sum CE contributions over background + all foreground classes
      must equal ordinary mean CE.

    Useful before running O1/O2.
    """

    outputs = as_output_list(
        output
    )

    report = []

    for i, logits in enumerate(
        outputs
    ):
        t = target_for_logits(
            target,
            logits,
        )

        num_classes = int(
            logits.shape[1]
        )

        dice_parts = []

        for c in range(
            1,
            num_classes,
        ):
            dice_parts.append(
                class_dice_training_contribution(
                    logits,
                    t,
                    c,
                    smooth=smooth,
                )
            )

        decomposed_dice = (
            torch.stack(
                dice_parts
            ).sum()
        )

        probs = F.softmax(
            logits,
            dim=1,
        )

        y = t[
            :,
            0
        ].long()

        dice_values = []

        for c in range(
            1,
            num_classes,
        ):
            p = probs[
                :,
                c
            ]

            g = (
                y == c
            ).to(
                p.dtype
            )

            axes = tuple(
                range(
                    1,
                    p.ndim,
                )
            )

            inter = (
                p * g
            ).sum(
                dim=axes
            )

            sp = p.sum(
                dim=axes
            )

            sg = g.sum(
                dim=axes
            )

            dc = (
                2 * inter
                + smooth
            ) / torch.clamp(
                sp + sg + smooth,
                min=1e-8,
            )

            dice_values.append(
                dc.mean()
            )

        reference_dice = (
            -torch.stack(
                dice_values
            ).mean()
        )

        ce_parts = []

        for c in range(
            num_classes
        ):
            ce_parts.append(
                class_ce_training_contribution(
                    logits,
                    t,
                    c,
                )
            )

        decomposed_ce = (
            torch.stack(
                ce_parts
            ).sum()
        )

        reference_ce = (
            F.cross_entropy(
                logits,
                y,
                reduction="mean",
            )
        )

        report.append({
            "output_index": i,
            "shape": list(
                logits.shape
            ),
            "dice_reference": float(
                reference_dice.detach().cpu()
            ),
            "dice_decomposed": float(
                decomposed_dice.detach().cpu()
            ),
            "dice_abs_error": float(
                abs(
                    (
                        reference_dice
                        - decomposed_dice
                    ).detach().cpu()
                )
            ),
            "ce_reference": float(
                reference_ce.detach().cpu()
            ),
            "ce_decomposed": float(
                decomposed_ce.detach().cpu()
            ),
            "ce_abs_error": float(
                abs(
                    (
                        reference_ce
                        - decomposed_ce
                    ).detach().cpu()
                )
            ),
        })

    return {
        "outputs": report
    }
