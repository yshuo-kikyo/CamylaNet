#!/usr/bin/env python3

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import nibabel as nib
import numpy as np


@dataclass
class CaseAudit:
    case_id: str

    gt_left_voxels: int
    gt_right_voxels: int
    pred_left_voxels: int
    pred_right_voxels: int

    dice_predL_gtL: float
    dice_predL_gtR: float
    dice_predR_gtR: float
    dice_predR_gtL: float

    correct_identity_score: float
    swapped_identity_score: float
    swap_advantage: float
    swap_flag: bool

    gt_left_centroid_i: float
    gt_left_centroid_j: float
    gt_left_centroid_k: float
    gt_right_centroid_i: float
    gt_right_centroid_j: float
    gt_right_centroid_k: float

    pred_left_centroid_i: float
    pred_left_centroid_j: float
    pred_left_centroid_k: float
    pred_right_centroid_i: float
    pred_right_centroid_j: float
    pred_right_centroid_k: float


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred-dir", type=Path, required=True)
    p.add_argument("--gt-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--left-label", type=int, default=3)
    p.add_argument("--right-label", type=int, default=2)
    p.add_argument("--swap-margin", type=float, default=0.05)
    return p.parse_args()


def dice(a, b):
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)

    na = int(a.sum())
    nb = int(b.sum())

    if na + nb == 0:
        return float("nan")

    inter = int(np.logical_and(a, b).sum())
    return 2.0 * inter / (na + nb)


def safe_mean(xs: Iterable[float]):
    arr = np.asarray(list(xs), dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def safe_std(xs: Iterable[float]):
    arr = np.asarray(list(xs), dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanstd(arr))


def centroid(mask):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.full(3, np.nan)
    return coords.mean(axis=0)


def load_seg(path):
    img = nib.load(str(path))
    arr = np.asanyarray(img.dataobj)
    arr = np.rint(arr).astype(np.int16, copy=False)
    return arr


def find_gt(pred_path, gt_dir):
    p = gt_dir / pred_path.name
    if p.exists():
        return p

    name = pred_path.name
    if name.endswith(".nii.gz"):
        stem = name[:-7]
    else:
        stem = pred_path.stem

    for ext in [".nii.gz", ".nii"]:
        p = gt_dir / f"{stem}{ext}"
        if p.exists():
            return p

    return None


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for pred_path in sorted(args.pred_dir.glob("*.nii.gz")):
        gt_path = find_gt(pred_path, args.gt_dir)
        if gt_path is None:
            continue

        pred = load_seg(pred_path)
        gt = load_seg(gt_path)

        if pred.shape != gt.shape:
            raise RuntimeError(
                f"Shape mismatch {pred_path.name}: "
                f"{pred.shape} vs {gt.shape}"
            )

        gt_l = gt == args.left_label
        gt_r = gt == args.right_label
        pr_l = pred == args.left_label
        pr_r = pred == args.right_label

        pll = dice(pr_l, gt_l)
        plr = dice(pr_l, gt_r)
        prr = dice(pr_r, gt_r)
        prl = dice(pr_r, gt_l)

        correct = safe_mean([pll, prr])
        swapped = safe_mean([plr, prl])
        advantage = swapped - correct

        glc = centroid(gt_l)
        grc = centroid(gt_r)
        plc = centroid(pr_l)
        prc = centroid(pr_r)

        row = CaseAudit(
            case_id=pred_path.name.replace(".nii.gz", ""),
            gt_left_voxels=int(gt_l.sum()),
            gt_right_voxels=int(gt_r.sum()),
            pred_left_voxels=int(pr_l.sum()),
            pred_right_voxels=int(pr_r.sum()),
            dice_predL_gtL=pll,
            dice_predL_gtR=plr,
            dice_predR_gtR=prr,
            dice_predR_gtL=prl,
            correct_identity_score=correct,
            swapped_identity_score=swapped,
            swap_advantage=advantage,
            swap_flag=bool(
                np.isfinite(advantage)
                and advantage > args.swap_margin
            ),
            gt_left_centroid_i=float(glc[0]),
            gt_left_centroid_j=float(glc[1]),
            gt_left_centroid_k=float(glc[2]),
            gt_right_centroid_i=float(grc[0]),
            gt_right_centroid_j=float(grc[1]),
            gt_right_centroid_k=float(grc[2]),
            pred_left_centroid_i=float(plc[0]),
            pred_left_centroid_j=float(plc[1]),
            pred_left_centroid_k=float(plc[2]),
            pred_right_centroid_i=float(prc[0]),
            pred_right_centroid_j=float(prc[1]),
            pred_right_centroid_k=float(prc[2]),
        )

        rows.append(row)

    if not rows:
        raise RuntimeError("No valid cases found")

    csv_path = args.out_dir / "kidney_identity_cases.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(asdict(rows[0]).keys()),
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(asdict(row))

    swap_count = sum(r.swap_flag for r in rows)

    summary = {
        "num_cases": len(rows),
        "left_label": args.left_label,
        "right_label": args.right_label,
        "swap_margin": args.swap_margin,
        "swap_count": int(swap_count),
        "swap_rate": float(swap_count / len(rows)),
        "correct_identity_score": {
            "mean": safe_mean(
                [r.correct_identity_score for r in rows]
            ),
            "std": safe_std(
                [r.correct_identity_score for r in rows]
            ),
        },
        "swapped_identity_score": {
            "mean": safe_mean(
                [r.swapped_identity_score for r in rows]
            ),
            "std": safe_std(
                [r.swapped_identity_score for r in rows]
            ),
        },
        "swap_advantage": {
            "mean": safe_mean(
                [r.swap_advantage for r in rows]
            ),
            "std": safe_std(
                [r.swap_advantage for r in rows]
            ),
        },
        "swapped_cases": [
            {
                "case_id": r.case_id,
                "correct": r.correct_identity_score,
                "swapped": r.swapped_identity_score,
                "advantage": r.swap_advantage,
            }
            for r in rows
            if r.swap_flag
        ],
    }

    json_path = args.out_dir / "kidney_identity_summary.json"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Kidney identity audit complete")
    print("Cases:", len(rows))
    print("Swap count:", swap_count)
    print("Swap rate:", summary["swap_rate"])
    print(
        "Correct mean:",
        summary["correct_identity_score"]["mean"],
    )
    print(
        "Swapped mean:",
        summary["swapped_identity_score"]["mean"],
    )
    print("CSV :", csv_path)
    print("JSON:", json_path)


if __name__ == "__main__":
    main()
