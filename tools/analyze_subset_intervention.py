#!/usr/bin/env python3
"""
Analyze O2 three-organ subset intervention results.

Inputs
------
third_order_residuals.csv
repeat_numerical_control.csv

Optional
--------
matched_random_triplets.json
random O2 result directories

Primary questions
-----------------
1. Is |I3| larger than repeated numerical variation?
2. Is the sign/magnitude stable across step sizes?
3. Is the effect reproducible across cases?
4. If random controls are available, does the structured triplet exceed them?

Important
---------
This script does NOT declare a Hypergraph GO merely because I3 != 0.

The intended interpretation is:

Gate 1:
    reproducible non-additive interaction above numerical error

Gate 2:
    survives later pairwise/confound controls

Gate 3:
    higher-order method later beats individual/pairwise baselines
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--o2-dir",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--metric",
        choices=[
            "i3_soft_dice",
            "i3_loss",
            "i3_hard_dice",
        ],
        default="i3_soft_dice",
    )

    p.add_argument(
        "--noise-multiplier",
        type=float,
        default=5.0,
        help=(
            "Signal must exceed this multiple of repeat std "
            "to be marked above numerical noise."
        ),
    )

    p.add_argument(
        "--min-abs-effect",
        type=float,
        default=1e-6,
        help="Absolute floor for meaningful I3 magnitude.",
    )

    p.add_argument(
        "--min-case-consistency",
        type=float,
        default=0.70,
        help=(
            "Minimum fraction of nonzero case means sharing "
            "the dominant sign."
        ),
    )

    p.add_argument(
        "--min-step-consistency",
        type=float,
        default=0.70,
        help=(
            "Minimum fraction of step-size means sharing "
            "the dominant sign."
        ),
    )

    p.add_argument(
        "--random-root",
        type=Path,
        default=None,
        help=(
            "Optional directory containing random-control O2 runs. "
            "Each child directory should contain "
            "third_order_residuals.csv."
        ),
    )

    return p.parse_args()


def read_csv(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return list(csv.DictReader(f))


def safe_float(x):
    try:
        v = float(x)
    except Exception:
        return np.nan

    return v if np.isfinite(v) else np.nan


def mean_valid(xs):
    arr = np.asarray(xs, dtype=float)
    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        return np.nan

    return float(np.mean(arr))


def std_valid(xs):
    arr = np.asarray(xs, dtype=float)
    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        return np.nan

    return float(np.std(arr))


def median_valid(xs):
    arr = np.asarray(xs, dtype=float)
    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        return np.nan

    return float(np.median(arr))


def dominant_sign_consistency(values, eps=0.0):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        return {
            "dominant_sign": 0,
            "consistency": np.nan,
            "positive_fraction": np.nan,
            "negative_fraction": np.nan,
            "nonzero_count": 0,
        }

    arr = arr[np.abs(arr) > eps]

    if len(arr) == 0:
        return {
            "dominant_sign": 0,
            "consistency": 0.0,
            "positive_fraction": 0.0,
            "negative_fraction": 0.0,
            "nonzero_count": 0,
        }

    pos = float(np.mean(arr > 0))
    neg = float(np.mean(arr < 0))

    if pos >= neg:
        sign = 1
        consistency = pos
    else:
        sign = -1
        consistency = neg

    return {
        "dominant_sign": sign,
        "consistency": float(consistency),
        "positive_fraction": pos,
        "negative_fraction": neg,
        "nonzero_count": int(len(arr)),
    }


def load_repeat_noise(o2_dir):
    path = (
        o2_dir
        / "repeat_numerical_control.csv"
    )

    if not path.exists():
        return {}

    rows = read_csv(path)

    out = {}

    for r in rows:
        key = (
            r["case_id"],
            r["triplet"],
            float(r["step_size"]),
            int(r["target_class"]),
        )

        out[key] = {
            "std": safe_float(
                r["i3_soft_std_repeat"]
            ),
            "range": safe_float(
                r["i3_soft_max_minus_min"]
            ),
        }

    return out


def aggregate_repeats(rows, metric):
    grouped = defaultdict(list)

    for r in rows:
        value = safe_float(
            r[metric]
        )

        if not np.isfinite(value):
            continue

        key = (
            r["case_id"],
            r["triplet"],
            r["triplet_names"],
            float(r["step_size"]),
            int(r["target_class"]),
            r["target_name"],
            str(r["target_in_triplet"]).lower()
            == "true",
        )

        grouped[key].append(value)

    out = []

    for key, values in grouped.items():
        (
            case_id,
            triplet,
            triplet_names,
            step_size,
            target_class,
            target_name,
            target_in_triplet,
        ) = key

        out.append({
            "case_id": case_id,
            "triplet": triplet,
            "triplet_names": triplet_names,
            "step_size": step_size,
            "target_class": target_class,
            "target_name": target_name,
            "target_in_triplet": target_in_triplet,
            "i3_mean": mean_valid(values),
            "i3_std_repeat_direct": std_valid(values),
            "i3_abs_mean": abs(
                mean_valid(values)
            ),
            "num_repeats": len(values),
        })

    return out


def add_noise_test(
    agg_rows,
    repeat_noise,
    noise_multiplier,
    min_abs_effect,
):
    out = []

    for r in agg_rows:
        key = (
            r["case_id"],
            r["triplet"],
            float(r["step_size"]),
            int(r["target_class"]),
        )

        noise = repeat_noise.get(
            key,
            {},
        )

        std_noise = noise.get(
            "std",
            r["i3_std_repeat_direct"],
        )

        if not np.isfinite(std_noise):
            std_noise = 0.0

        threshold = max(
            min_abs_effect,
            noise_multiplier * std_noise,
        )

        above = (
            np.isfinite(r["i3_abs_mean"])
            and r["i3_abs_mean"] > threshold
        )

        q = dict(r)

        q.update({
            "repeat_noise_std": std_noise,
            "noise_threshold": threshold,
            "above_numerical_noise": bool(
                above
            ),
            "signal_to_repeat_std": (
                float(
                    r["i3_abs_mean"]
                    / std_noise
                )
                if std_noise > 0
                else (
                    float("inf")
                    if r["i3_abs_mean"] > 0
                    else 0.0
                )
            ),
        })

        out.append(q)

    return out



def estimate_eta_scaling(
    rows,
    min_abs_effect,
):
    """
    Estimate:

        |I3(eta)| ~ eta^p

    using log-log linear regression.

    This is a diagnostic of local response order,
    not a formal hypothesis test.
    """

    grouped = defaultdict(list)

    for r in rows:
        key = (
            r["triplet"],
            r["triplet_names"],
            r["target_class"],
            r["target_name"],
            r["target_in_triplet"],
        )

        eta = float(
            r["step_size"]
        )

        value = abs(
            float(
                r["i3_mean"]
            )
        )

        if (
            eta > 0
            and np.isfinite(value)
            and value > min_abs_effect
        ):
            grouped[key].append(
                (
                    eta,
                    value,
                )
            )

    output = []

    for key, points in grouped.items():
        by_eta = defaultdict(list)

        for eta, value in points:
            by_eta[eta].append(
                value
            )

        x = []
        y = []

        for eta, values in sorted(
            by_eta.items()
        ):
            mean_abs = float(
                np.mean(values)
            )

            if mean_abs <= 0:
                continue

            x.append(
                np.log(eta)
            )

            y.append(
                np.log(mean_abs)
            )

        if len(x) >= 3:
            x = np.asarray(
                x,
                dtype=float,
            )

            y = np.asarray(
                y,
                dtype=float,
            )

            slope, intercept = (
                np.polyfit(
                    x,
                    y,
                    1,
                )
            )

            pred = (
                slope * x
                + intercept
            )

            denom = float(
                np.sum(
                    (
                        y
                        - np.mean(y)
                    ) ** 2
                )
            )

            r2 = (
                1.0
                - float(
                    np.sum(
                        (
                            y
                            - pred
                        ) ** 2
                    )
                )
                / denom
                if denom > 0
                else np.nan
            )

        else:
            slope = np.nan
            r2 = np.nan

        output.append({
            "triplet": key[0],
            "triplet_names": key[1],
            "target_class": key[2],
            "target_name": key[3],
            "target_in_triplet": key[4],
            "num_step_sizes": len(x),
            "eta_scaling_exponent": (
                float(slope)
                if np.isfinite(slope)
                else np.nan
            ),
            "eta_scaling_r2": (
                float(r2)
                if np.isfinite(r2)
                else np.nan
            ),
        })

    return output


def summarize_triplets(
    rows,
    min_abs_effect,
    min_case_consistency,
    min_step_consistency,
):
    grouped = defaultdict(list)

    for r in rows:
        key = (
            r["triplet"],
            r["triplet_names"],
            r["target_class"],
            r["target_name"],
            r["target_in_triplet"],
        )

        grouped[key].append(r)

    summary = []

    for key, items in grouped.items():
        (
            triplet,
            triplet_names,
            target_class,
            target_name,
            target_in_triplet,
        ) = key

        # Case means across step sizes
        by_case = defaultdict(list)

        for r in items:
            by_case[
                r["case_id"]
            ].append(
                r["i3_mean"]
            )

        case_means = {
            case: mean_valid(vals)
            for case, vals
            in by_case.items()
        }

        # Step-size means across cases
        by_step = defaultdict(list)

        for r in items:
            by_step[
                r["step_size"]
            ].append(
                r["i3_mean"]
            )

        step_means = {
            step: mean_valid(vals)
            for step, vals
            in by_step.items()
        }

        case_sign = (
            dominant_sign_consistency(
                list(
                    case_means.values()
                ),
                eps=min_abs_effect,
            )
        )

        step_sign = (
            dominant_sign_consistency(
                list(
                    step_means.values()
                ),
                eps=min_abs_effect,
            )
        )

        above_flags = [
            bool(
                r[
                    "above_numerical_noise"
                ]
            )
            for r in items
        ]

        values = [
            r["i3_mean"]
            for r in items
        ]

        abs_values = [
            abs(r["i3_mean"])
            for r in items
        ]

        signal_fraction = float(
            np.mean(
                above_flags
            )
        ) if above_flags else np.nan

        gate1 = bool(
            signal_fraction >= 0.5
            and case_sign[
                "consistency"
            ] >= min_case_consistency
            and step_sign[
                "consistency"
            ] >= min_step_consistency
            and median_valid(
                abs_values
            ) > min_abs_effect
        )

        summary.append({
            "triplet": triplet,
            "triplet_names": triplet_names,
            "target_class": target_class,
            "target_name": target_name,
            "target_in_triplet": target_in_triplet,

            "num_case_step_points": len(
                items
            ),

            "num_cases": len(
                case_means
            ),

            "num_step_sizes": len(
                step_means
            ),

            "mean_i3": mean_valid(
                values
            ),

            "median_i3": median_valid(
                values
            ),

            "mean_abs_i3": mean_valid(
                abs_values
            ),

            "median_abs_i3": (
                median_valid(
                    abs_values
                )
            ),

            "fraction_above_numerical_noise": (
                signal_fraction
            ),

            "case_dominant_sign": (
                case_sign[
                    "dominant_sign"
                ]
            ),

            "case_sign_consistency": (
                case_sign[
                    "consistency"
                ]
            ),

            "step_dominant_sign": (
                step_sign[
                    "dominant_sign"
                ]
            ),

            "step_sign_consistency": (
                step_sign[
                    "consistency"
                ]
            ),

            "gate1_local_nonadditivity_pass": (
                gate1
            ),
        })

    return summary


def load_random_controls(
    random_root,
    metric,
):
    if random_root is None:
        return []

    all_rows = []

    for path in sorted(
        random_root.glob(
            "*/third_order_residuals.csv"
        )
    ):
        run_name = (
            path.parent.name
        )

        rows = read_csv(path)

        for r in rows:
            value = safe_float(
                r[metric]
            )

            if not np.isfinite(value):
                continue

            all_rows.append({
                "run_name": run_name,
                "triplet": r[
                    "triplet"
                ],
                "step_size": float(
                    r[
                        "step_size"
                    ]
                ),
                "target_class": int(
                    r[
                        "target_class"
                    ]
                ),
                "target_name": r[
                    "target_name"
                ],
                "value": value,
            })

    return all_rows


def compare_structured_random(
    structured_summary,
    random_rows,
):
    if not random_rows:
        return []

    # Aggregate each random run/triplet/target
    grouped = defaultdict(list)

    for r in random_rows:
        key = (
            r["run_name"],
            r["triplet"],
            r["target_class"],
            r["target_name"],
        )

        grouped[key].append(
            abs(
                r["value"]
            )
        )

    random_stats = []

    for key, vals in grouped.items():
        random_stats.append({
            "run_name": key[0],
            "triplet": key[1],
            "target_class": key[2],
            "target_name": key[3],
            "mean_abs_i3": (
                mean_valid(vals)
            ),
        })

    by_target = defaultdict(list)

    for r in random_stats:
        by_target[
            r["target_class"]
        ].append(
            r["mean_abs_i3"]
        )

    out = []

    for s in structured_summary:
        target = s[
            "target_class"
        ]

        null = np.asarray(
            by_target.get(
                target,
                [],
            ),
            dtype=float,
        )

        null = null[
            np.isfinite(null)
        ]

        observed = s[
            "mean_abs_i3"
        ]

        if len(null) == 0:
            continue

        percentile = float(
            np.mean(
                null <= observed
            )
        )

        empirical_p = float(
            (
                1
                + np.sum(
                    null >= observed
                )
            )
            / (
                len(null)
                + 1
            )
        )

        mu = float(
            np.mean(null)
        )

        sigma = float(
            np.std(null)
        )

        z = (
            float(
                (observed - mu)
                / sigma
            )
            if sigma > 0
            else None
        )

        out.append({
            "triplet": s[
                "triplet"
            ],
            "triplet_names": s[
                "triplet_names"
            ],
            "target_class": target,
            "target_name": s[
                "target_name"
            ],
            "structured_mean_abs_i3": (
                observed
            ),
            "random_n": int(
                len(null)
            ),
            "random_mean_abs_i3": (
                mu
            ),
            "random_std_abs_i3": (
                sigma
            ),
            "empirical_percentile": (
                percentile
            ),
            "empirical_p_like": (
                empirical_p
            ),
            "z_score": z,
        })

    return out


def write_csv(
    path,
    rows,
):
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


def json_safe(obj):
    if isinstance(
        obj,
        dict,
    ):
        return {
            k: json_safe(v)
            for k, v
            in obj.items()
        }

    if isinstance(
        obj,
        list,
    ):
        return [
            json_safe(v)
            for v in obj
        ]

    if isinstance(
        obj,
        (np.floating, float),
    ):
        if not np.isfinite(
            float(obj)
        ):
            return None

        return float(obj)

    if isinstance(
        obj,
        (np.integer, int),
    ):
        return int(obj)

    if isinstance(
        obj,
        (np.bool_, bool),
    ):
        return bool(obj)

    return obj


def main():
    args = parse_args()

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    residual_path = (
        args.o2_dir
        / "third_order_residuals.csv"
    )

    if not residual_path.exists():
        raise FileNotFoundError(
            residual_path
        )

    raw_rows = read_csv(
        residual_path
    )

    repeat_noise = (
        load_repeat_noise(
            args.o2_dir
        )
    )

    agg = aggregate_repeats(
        raw_rows,
        args.metric,
    )

    tested = add_noise_test(
        agg,
        repeat_noise,
        args.noise_multiplier,
        args.min_abs_effect,
    )

    summary = summarize_triplets(
        tested,
        args.min_abs_effect,
        args.min_case_consistency,
        args.min_step_consistency,
    )

    eta_scaling = estimate_eta_scaling(
        tested,
        args.min_abs_effect,
    )

    write_csv(
        args.out_dir
        / "eta_scaling_summary.csv",
        eta_scaling,
    )

    write_csv(
        args.out_dir
        / "case_step_nonadditivity.csv",
        tested,
    )

    write_csv(
        args.out_dir
        / "triplet_nonadditivity_summary.csv",
        summary,
    )

    random_rows = (
        load_random_controls(
            args.random_root,
            args.metric,
        )
    )

    random_comparison = (
        compare_structured_random(
            summary,
            random_rows,
        )
    )

    write_csv(
        args.out_dir
        / "structured_vs_random.csv",
        random_comparison,
    )

    gate1_pass_rows = [
        r
        for r in summary
        if r[
            "gate1_local_nonadditivity_pass"
        ]
    ]

    final = {
        "metric": args.metric,

        "thresholds": {
            "noise_multiplier": (
                args.noise_multiplier
            ),
            "min_abs_effect": (
                args.min_abs_effect
            ),
            "min_case_consistency": (
                args.min_case_consistency
            ),
            "min_step_consistency": (
                args.min_step_consistency
            ),
        },

        "num_triplet_target_summaries": (
            len(summary)
        ),

        "num_gate1_pass": (
            len(gate1_pass_rows)
        ),

        "gate1_any_pass": bool(
            len(
                gate1_pass_rows
            ) > 0
        ),

        "gate1_pass_items": [
            {
                "triplet": r[
                    "triplet"
                ],
                "triplet_names": r[
                    "triplet_names"
                ],
                "target_class": r[
                    "target_class"
                ],
                "target_name": r[
                    "target_name"
                ],
                "mean_abs_i3": r[
                    "mean_abs_i3"
                ],
                "fraction_above_numerical_noise": (
                    r[
                        "fraction_above_numerical_noise"
                    ]
                ),
                "case_sign_consistency": (
                    r[
                        "case_sign_consistency"
                    ]
                ),
                "step_sign_consistency": (
                    r[
                        "step_sign_consistency"
                    ]
                ),
            }
            for r in gate1_pass_rows
        ],

        "random_control_available": bool(
            len(
                random_comparison
            ) > 0
        ),

        "interpretation": {
            "gate1": (
                "Local non-additive interaction "
                "above repeat error and stable "
                "across cases/step sizes."
            ),

            "not_yet_proven": [
                "beyond complete pairwise information",
                "survives confound control",
                "hypergraph method necessity",
            ],
        },
    }

    with (
        args.out_dir
        / "o2_gate_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            json_safe(final),
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        "O2 ANALYSIS COMPLETE"
    )

    print(
        "Metric:",
        args.metric,
    )

    print(
        "Triplet-target summaries:",
        len(summary),
    )

    print(
        "Gate1 pass:",
        len(gate1_pass_rows),
    )

    print(
        "Output:",
        args.out_dir,
    )

    if len(
        gate1_pass_rows
    ) == 0:
        print()
        print(
            "Current result: NO reproducible "
            "local non-additive signal passed "
            "the preset Gate1 criteria."
        )

        print(
            "Do NOT proceed to a Hypergraph method "
            "based on this O2 result alone."
        )

    else:
        print()
        print(
            "At least one Gate1 candidate exists."
        )

        print(
            "Next requirement: O3/O4 pairwise + "
            "confound controls before any method claim."
        )


if __name__ == "__main__":
    main()
