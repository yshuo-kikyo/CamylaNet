#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--o1-dir",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--top-k",
        type=int,
        default=20,
    )

    return p.parse_args()


def read_matrix_csv(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        rows = list(csv.reader(f))

    labels = rows[0][1:]

    matrix = []

    for row in rows[1:]:
        vals = []

        for x in row[1:]:
            if x == "":
                vals.append(np.nan)
            else:
                vals.append(float(x))

        matrix.append(vals)

    return labels, np.asarray(
        matrix,
        dtype=float,
    )


def read_records(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return list(
            csv.DictReader(f)
        )


def safe_float(x):
    if x in (
        "",
        None,
        "nan",
        "NaN",
    ):
        return np.nan

    try:
        return float(x)
    except Exception:
        return np.nan


def summarize_pairs(
    labels,
    matrix,
):
    rows = []

    n = len(labels)

    for i in range(n):
        for j in range(i + 1, n):
            value = matrix[i, j]

            if not np.isfinite(value):
                continue

            rows.append({
                "class_i": labels[i],
                "class_j": labels[j],
                "cosine": float(value),
                "abs_cosine": float(
                    abs(value)
                ),
                "sign": (
                    "positive"
                    if value > 0
                    else (
                        "negative"
                        if value < 0
                        else "zero"
                    )
                ),
            })

    return rows


def write_csv(path, rows):
    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


def aggregate_class_stats(records):
    grouped = {}

    for r in records:
        key = (
            r["crop_mode"],
            int(r["class_id"]),
            r["class_name"],
        )

        grouped.setdefault(
            key,
            [],
        ).append(r)

    out = []

    for (
        crop_mode,
        class_id,
        class_name,
    ), items in sorted(
        grouped.items()
    ):
        full_voxels = np.asarray(
            [
                safe_float(
                    x[
                        "full_voxel_count"
                    ]
                )
                for x in items
            ],
            dtype=float,
        )

        crop_voxels = np.asarray(
            [
                safe_float(
                    x[
                        "voxel_count_crop"
                    ]
                )
                for x in items
            ],
            dtype=float,
        )

        losses = np.asarray(
            [
                safe_float(
                    x["loss"]
                )
                for x in items
            ],
            dtype=float,
        )

        primary_norm = np.asarray(
            [
                safe_float(
                    x[
                        "aligned_primary_grad_norm"
                    ]
                )
                for x in items
            ],
            dtype=float,
        )

        head_norm = np.asarray(
            [
                safe_float(
                    x[
                        "aligned_head_grad_norm"
                    ]
                )
                for x in items
            ],
            dtype=float,
        )

        def stat(arr, fn):
            valid = arr[
                np.isfinite(arr)
            ]

            if len(valid) == 0:
                return np.nan

            return float(fn(valid))

        out.append({
            "crop_mode": crop_mode,
            "class_id": class_id,
            "class_name": class_name,

            "n_records": len(items),

            "median_full_voxels": stat(
                full_voxels,
                np.median,
            ),

            "median_crop_voxels": stat(
                crop_voxels,
                np.median,
            ),

            "mean_loss": stat(
                losses,
                np.mean,
            ),

            "mean_primary_grad_norm": stat(
                primary_norm,
                np.mean,
            ),

            "median_primary_grad_norm": stat(
                primary_norm,
                np.median,
            ),

            "mean_head_grad_norm": stat(
                head_norm,
                np.mean,
            ),

            "median_head_grad_norm": stat(
                head_norm,
                np.median,
            ),

            "primary_valid_rate": float(
                sum(
                    x[
                        "primary_valid"
                    ].lower()
                    == "true"
                    for x in items
                )
                / len(items)
            ),

            "head_valid_rate": float(
                sum(
                    x[
                        "head_valid"
                    ].lower()
                    == "true"
                    for x in items
                )
                / len(items)
            ),
        })

    return out


def compare_sites(
    labels,
    primary,
    head,
):
    x = []
    y = []

    n = len(labels)

    for i in range(n):
        for j in range(i + 1, n):
            a = primary[i, j]
            b = head[i, j]

            if (
                np.isfinite(a)
                and np.isfinite(b)
            ):
                x.append(a)
                y.append(b)

    if len(x) < 3:
        return {
            "num_pairs": len(x),
            "pearson": None,
        }

    x = np.asarray(x)
    y = np.asarray(y)

    pearson = float(
        np.corrcoef(
            x,
            y,
        )[0, 1]
    )

    return {
        "num_pairs": len(x),
        "pearson": pearson,
    }


def main():
    args = parse_args()

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    records_path = (
        args.o1_dir
        / "organ_gradient_records.csv"
    )

    if not records_path.exists():
        raise FileNotFoundError(
            records_path
        )

    records = read_records(
        records_path
    )

    class_stats = (
        aggregate_class_stats(
            records
        )
    )

    write_csv(
        args.out_dir
        / "class_gradient_summary.csv",
        class_stats,
    )

    global_summary = {}

    # Primary paper-facing O1 analysis uses SAME-CROP only.
    # organ_center remains a Dice-only auxiliary diagnostic and is
    # intentionally excluded from the aligned-loss summary.
    for mode in [
        "same_crop",
    ]:
        primary_path = (
            args.o1_dir
            / f"{mode}_aligned_primary_mean_cosine.csv"
        )

        head_path = (
            args.o1_dir
            / f"{mode}_aligned_head_mean_cosine.csv"
        )

        if not (
            primary_path.exists()
            and head_path.exists()
        ):
            continue

        labels_p, primary = (
            read_matrix_csv(
                primary_path
            )
        )

        labels_h, head = (
            read_matrix_csv(
                head_path
            )
        )

        if labels_p != labels_h:
            raise RuntimeError(
                "Primary/head labels differ"
            )

        pairs_p = summarize_pairs(
            labels_p,
            primary,
        )

        pairs_h = summarize_pairs(
            labels_h,
            head,
        )

        write_csv(
            args.out_dir
            / f"{mode}_primary_pairs.csv",
            pairs_p,
        )

        write_csv(
            args.out_dir
            / f"{mode}_head_pairs.csv",
            pairs_h,
        )

        positive_p = sorted(
            pairs_p,
            key=lambda x: x[
                "cosine"
            ],
            reverse=True,
        )

        negative_p = sorted(
            pairs_p,
            key=lambda x: x[
                "cosine"
            ],
        )

        write_csv(
            args.out_dir
            / f"{mode}_primary_top_positive.csv",
            positive_p[
                : args.top_k
            ],
        )

        write_csv(
            args.out_dir
            / f"{mode}_primary_top_negative.csv",
            negative_p[
                : args.top_k
            ],
        )

        primary_values = np.asarray(
            [
                x["cosine"]
                for x in pairs_p
            ],
            dtype=float,
        )

        head_values = np.asarray(
            [
                x["cosine"]
                for x in pairs_h
            ],
            dtype=float,
        )

        global_summary[
            mode
        ] = {
            "num_primary_pairs": int(
                len(primary_values)
            ),

            "primary_mean_cosine": (
                float(
                    np.mean(
                        primary_values
                    )
                )
                if len(primary_values)
                else None
            ),

            "primary_median_cosine": (
                float(
                    np.median(
                        primary_values
                    )
                )
                if len(primary_values)
                else None
            ),

            "primary_positive_fraction": (
                float(
                    np.mean(
                        primary_values
                        > 0
                    )
                )
                if len(primary_values)
                else None
            ),

            "primary_negative_fraction": (
                float(
                    np.mean(
                        primary_values
                        < 0
                    )
                )
                if len(primary_values)
                else None
            ),

            "head_mean_cosine": (
                float(
                    np.mean(
                        head_values
                    )
                )
                if len(head_values)
                else None
            ),

            "primary_head_pairwise_similarity": (
                compare_sites(
                    labels_p,
                    primary,
                    head,
                )
            ),
        }

    with (
        args.out_dir
        / "o1_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            global_summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "O1 analysis complete"
    )

    print(
        "Output:",
        args.out_dir,
    )


if __name__ == "__main__":
    main()
