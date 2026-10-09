#!/usr/bin/env python3
"""
Build the unified O3/O4 dataset.

Each manifest row represents ONE diagnostic timepoint from ONE independent run.

Required manifest columns
-------------------------
run_id
timepoint
o1_dir
current_dice_csv
future_dice_csv

Optional
--------
o2_dir

Expected current/future Dice CSV format
---------------------------------------
class_id,dice

Example:
1,0.91
2,0.88
...
15,0.72

O1 requirements
---------------
<o1_dir>/organ_gradient_records.csv
<o1_dir>/same_crop_aligned_primary_mean_cosine.csv

O2 optional input
-----------------
<o2_dir>/third_order_residuals.csv

Output
------
One row per:

    run_id × timepoint × target_class

Columns include:

    run_id
    timepoint
    target_class
    current_dice
    future_dice
    future_dice_gain
    class_loss
    grad_norm
    organ_volume

and complete target-organ pairwise features:

    pair_class_01
    pair_class_02
    ...

plus available intervention features:

    higher_i3_<triplet>

Important
---------
This script ONLY builds features.

It does not perform prediction, significance testing, or claim higher-order
value. The downstream O3 script must split by whole run_id.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--out",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--metadata-out",
        type=Path,
        default=None,
    )

    return p.parse_args()


def read_csv(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return list(
            csv.DictReader(f)
        )


def safe_float(x):
    try:
        v = float(x)
    except Exception:
        return np.nan

    return v if np.isfinite(v) else np.nan


def resolve_path(
    value,
    manifest_dir,
):
    if value is None:
        return None

    value = str(value).strip()

    if value == "":
        return None

    p = Path(value)

    if not p.is_absolute():
        p = (
            manifest_dir
            / p
        ).resolve()

    return p


def load_dice_csv(path):
    if not path.exists():
        raise FileNotFoundError(
            path
        )

    rows = read_csv(path)

    out = {}

    for r in rows:
        if (
            "class_id" not in r
            or "dice" not in r
        ):
            raise RuntimeError(
                f"{path} must contain "
                "'class_id' and 'dice' columns"
            )

        c = int(
            r["class_id"]
        )

        d = safe_float(
            r["dice"]
        )

        if np.isfinite(d):
            out[c] = d

    return out


def load_matrix_csv(path):
    if not path.exists():
        raise FileNotFoundError(
            path
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        rows = list(
            csv.reader(f)
        )

    if len(rows) < 2:
        raise RuntimeError(
            f"Empty matrix CSV: {path}"
        )

    labels = rows[0][1:]

    matrix = []

    for row in rows[1:]:
        vals = []

        for x in row[1:]:
            if x == "":
                vals.append(
                    np.nan
                )
            else:
                vals.append(
                    safe_float(x)
                )

        matrix.append(vals)

    return (
        labels,
        np.asarray(
            matrix,
            dtype=float,
        ),
    )


def parse_class_id_from_label(
    label,
    class_name_to_id,
):
    # Matrix headers produced by O1 use organ names.
    if label in class_name_to_id:
        return class_name_to_id[
            label
        ]

    # Fallback if a future script writes integer labels.
    try:
        return int(label)
    except Exception:
        raise RuntimeError(
            f"Cannot map matrix label "
            f"{label!r} to class id"
        )


def load_o1_class_stats(
    o1_dir,
):
    path = (
        o1_dir
        / "organ_gradient_records.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    rows = read_csv(path)

    grouped = defaultdict(
        lambda: {
            "loss": [],
            "grad_norm": [],
            "volume": [],
        }
    )

    for r in rows:
        # O3 primary features use same-crop only.
        if (
            r.get(
                "crop_mode"
            )
            != "same_crop"
        ):
            continue

        c = int(
            r["class_id"]
        )

        loss = safe_float(
            r.get(
                "loss"
            )
        )

        grad = safe_float(
            r.get(
                "aligned_primary_grad_norm"
            )
        )

        volume = safe_float(
            r.get(
                "full_voxel_count"
            )
        )

        if np.isfinite(loss):
            grouped[
                c
            ][
                "loss"
            ].append(loss)

        if np.isfinite(grad):
            grouped[
                c
            ][
                "grad_norm"
            ].append(grad)

        if np.isfinite(volume):
            grouped[
                c
            ][
                "volume"
            ].append(volume)

    out = {}

    for c, values in grouped.items():
        out[c] = {
            "class_loss": (
                float(
                    np.mean(
                        values[
                            "loss"
                        ]
                    )
                )
                if values[
                    "loss"
                ]
                else np.nan
            ),

            "grad_norm": (
                float(
                    np.mean(
                        values[
                            "grad_norm"
                        ]
                    )
                )
                if values[
                    "grad_norm"
                ]
                else np.nan
            ),

            "organ_volume": (
                float(
                    np.median(
                        values[
                            "volume"
                        ]
                    )
                )
                if values[
                    "volume"
                ]
                else np.nan
            ),
        }

    return out


def load_o1_metadata(
    o1_dir,
):
    path = (
        o1_dir
        / "metadata.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def load_pairwise_features(
    o1_dir,
):
    metadata = load_o1_metadata(
        o1_dir
    )

    class_ids = [
        int(x)
        for x in metadata[
            "class_ids"
        ]
    ]

    class_names = [
        str(x)
        for x in metadata[
            "class_names"
        ]
    ]

    class_name_to_id = {
        name: cid
        for cid, name in zip(
            class_ids,
            class_names,
        )
    }

    path = (
        o1_dir
        / "same_crop_aligned_primary_mean_cosine.csv"
    )

    labels, matrix = (
        load_matrix_csv(
            path
        )
    )

    matrix_ids = [
        parse_class_id_from_label(
            x,
            class_name_to_id,
        )
        for x in labels
    ]

    if matrix.shape != (
        len(matrix_ids),
        len(matrix_ids),
    ):
        raise RuntimeError(
            f"Unexpected pairwise matrix shape "
            f"{matrix.shape}"
        )

    # target_class -> other_class -> cosine
    out = {}

    for i, target in enumerate(
        matrix_ids
    ):
        out[target] = {}

        for j, other in enumerate(
            matrix_ids
        ):
            out[
                target
            ][
                other
            ] = float(
                matrix[
                    i,
                    j,
                ]
            )

    return (
        sorted(
            matrix_ids
        ),
        out,
    )


def canonical_triplet_string(
    triplet,
):
    parts = [
        int(x)
        for x in str(
            triplet
        ).split("-")
        if str(x).strip()
    ]

    parts = sorted(parts)

    return "-".join(
        str(x)
        for x in parts
    )


def load_higher_features(
    o2_dir,
):
    """
    Returns:
        target_class -> {
            "higher_i3_<triplet>": mean_i3_soft_dice
        }
    """

    if o2_dir is None:
        return {}

    path = (
        o2_dir
        / "third_order_residuals.csv"
    )

    if not path.exists():
        return {}

    rows = read_csv(path)

    grouped = defaultdict(
        list
    )

    for r in rows:
        value = safe_float(
            r.get(
                "i3_soft_dice"
            )
        )

        if not np.isfinite(value):
            continue

        target = int(
            r[
                "target_class"
            ]
        )

        triplet = (
            canonical_triplet_string(
                r[
                    "triplet"
                ]
            )
        )

        key = (
            target,
            triplet,
        )

        grouped[
            key
        ].append(
            value
        )

    out = defaultdict(
        dict
    )

    for (
        target,
        triplet,
    ), values in grouped.items():

        feature_name = (
            "higher_i3_"
            + triplet.replace(
                "-",
                "_",
            )
        )

        out[
            target
        ][
            feature_name
        ] = float(
            np.mean(
                values
            )
        )

    return dict(out)


def validate_manifest(
    rows,
):
    required = [
        "run_id",
        "timepoint",
        "o1_dir",
        "current_dice_csv",
        "future_dice_csv",
    ]

    if not rows:
        raise RuntimeError(
            "Manifest is empty"
        )

    columns = set(
        rows[0].keys()
    )

    missing = [
        x
        for x in required
        if x not in columns
    ]

    if missing:
        raise RuntimeError(
            "Manifest missing columns: "
            + ", ".join(
                missing
            )
        )


def write_csv(
    path,
    rows,
    fields,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for r in rows:
            writer.writerow(
                {
                    k: r.get(
                        k,
                        ""
                    )
                    for k in fields
                }
            )


def main():
    args = parse_args()

    manifest_rows = read_csv(
        args.manifest
    )

    validate_manifest(
        manifest_rows
    )

    manifest_dir = (
        args.manifest.parent
    )

    output_rows = []

    all_pair_features = set()
    all_higher_features = set()

    source_summary = []

    for index, m in enumerate(
        manifest_rows
    ):
        run_id = str(
            m[
                "run_id"
            ]
        )

        timepoint = str(
            m[
                "timepoint"
            ]
        )

        o1_dir = resolve_path(
            m[
                "o1_dir"
            ],
            manifest_dir,
        )

        o2_dir = resolve_path(
            m.get(
                "o2_dir"
            ),
            manifest_dir,
        )

        current_path = resolve_path(
            m[
                "current_dice_csv"
            ],
            manifest_dir,
        )

        future_path = resolve_path(
            m[
                "future_dice_csv"
            ],
            manifest_dir,
        )

        print(
            f"[{index + 1}/"
            f"{len(manifest_rows)}] "
            f"run={run_id} "
            f"timepoint={timepoint}"
        )

        current_dice = (
            load_dice_csv(
                current_path
            )
        )

        future_dice = (
            load_dice_csv(
                future_path
            )
        )

        class_stats = (
            load_o1_class_stats(
                o1_dir
            )
        )

        (
            class_ids,
            pairwise,
        ) = load_pairwise_features(
            o1_dir
        )

        higher = (
            load_higher_features(
                o2_dir
            )
        )

        source_summary.append({
            "run_id": run_id,
            "timepoint": timepoint,
            "o1_dir": str(
                o1_dir
            ),
            "o2_dir": (
                str(o2_dir)
                if o2_dir is not None
                else None
            ),
            "current_dice_csv": str(
                current_path
            ),
            "future_dice_csv": str(
                future_path
            ),
        })

        for target in class_ids:
            if (
                target not in current_dice
                or target not in future_dice
            ):
                continue

            if target not in class_stats:
                continue

            row = {
                "run_id": run_id,
                "timepoint": timepoint,
                "target_class": target,

                "current_dice": (
                    current_dice[
                        target
                    ]
                ),

                "future_dice": (
                    future_dice[
                        target
                    ]
                ),

                "future_dice_gain": (
                    future_dice[
                        target
                    ]
                    - current_dice[
                        target
                    ]
                ),

                "class_loss": (
                    class_stats[
                        target
                    ][
                        "class_loss"
                    ]
                ),

                "grad_norm": (
                    class_stats[
                        target
                    ][
                        "grad_norm"
                    ]
                ),

                "organ_volume": (
                    class_stats[
                        target
                    ][
                        "organ_volume"
                    ]
                ),
            }

            # Complete target-vs-all pairwise row.
            for other in class_ids:
                if other == target:
                    continue

                feature = (
                    f"pair_class_"
                    f"{other:02d}"
                )

                value = (
                    pairwise[
                        target
                    ].get(
                        other,
                        np.nan,
                    )
                )

                row[
                    feature
                ] = value

                all_pair_features.add(
                    feature
                )

            # Add all available intervention features
            # for this target.
            for feature, value in (
                higher.get(
                    target,
                    {}
                ).items()
            ):
                row[
                    feature
                ] = value

                all_higher_features.add(
                    feature
                )

            output_rows.append(
                row
            )

    if not output_rows:
        raise RuntimeError(
            "No O3 rows were constructed."
        )

    base_fields = [
        "run_id",
        "timepoint",
        "target_class",
        "current_dice",
        "future_dice",
        "future_dice_gain",
        "class_loss",
        "grad_norm",
        "organ_volume",
    ]

    pair_fields = sorted(
        all_pair_features
    )

    higher_fields = sorted(
        all_higher_features
    )

    fields = (
        base_fields
        + pair_fields
        + higher_fields
    )

    write_csv(
        args.out,
        output_rows,
        fields,
    )

    metadata_path = (
        args.metadata_out
        if args.metadata_out
        is not None
        else args.out.with_suffix(
            ".metadata.json"
        )
    )

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = {
        "num_rows": len(
            output_rows
        ),

        "num_manifest_entries": len(
            manifest_rows
        ),

        "num_runs": len(
            set(
                r[
                    "run_id"
                ]
                for r
                in output_rows
            )
        ),

        "pair_features": (
            pair_fields
        ),

        "higher_features": (
            higher_fields
        ),

        "confounders": [
            "current_dice",
            "class_loss",
            "grad_norm",
            "organ_volume",
        ],

        "outcome": (
            "future_dice_gain"
        ),

        "pairwise_definition": (
            "target-organ cosine values from "
            "same_crop_aligned_primary_mean_cosine.csv"
        ),

        "higher_definition": (
            "mean i3_soft_dice for each available "
            "three-organ intervention triplet"
        ),

        "sources": (
            source_summary
        ),

        "important_rule": (
            "Downstream prediction must split by whole run_id."
        ),
    }

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        "O3 DATASET BUILD COMPLETE"
    )

    print(
        "Rows:",
        len(
            output_rows
        ),
    )

    print(
        "Runs:",
        metadata[
            "num_runs"
        ],
    )

    print(
        "Pairwise features:",
        len(
            pair_fields
        ),
    )

    print(
        "Higher features:",
        len(
            higher_fields
        ),
    )

    print(
        "CSV:",
        args.out,
    )

    print(
        "Metadata:",
        metadata_path,
    )


if __name__ == "__main__":
    main()
