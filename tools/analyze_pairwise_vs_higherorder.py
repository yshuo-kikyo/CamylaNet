#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path

import numpy as np


CONFOUNDERS = [
    "current_dice",
    "class_loss",
    "grad_norm",
    "organ_volume",
]


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)

    p.add_argument(
        "--group-col",
        default="run_id",
    )

    p.add_argument(
        "--outcome-col",
        default="future_dice_gain",
    )

    p.add_argument(
        "--pair-prefix",
        default="pair_",
    )

    p.add_argument(
        "--higher-prefix",
        default="higher_",
    )

    p.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[
            1e-4,
            1e-3,
            1e-2,
            1e-1,
            1.0,
            10.0,
            100.0,
        ],
    )

    return p.parse_args()


def read_rows(path):
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fval(x):
    try:
        v = float(x)
    except Exception:
        return np.nan

    return v if np.isfinite(v) else np.nan


def build_xy(rows, features, outcome, group_col):
    X = []
    y = []
    groups = []
    meta = []

    for r in rows:
        yy = fval(r.get(outcome))

        if not np.isfinite(yy):
            continue

        X.append([
            fval(r.get(c))
            for c in features
        ])

        y.append(yy)
        groups.append(r[group_col])
        meta.append(r)

    return (
        np.asarray(X, dtype=float),
        np.asarray(y, dtype=float),
        np.asarray(groups),
        meta,
    )


def fit_preprocessor(X):
    med = np.zeros(X.shape[1])

    for j in range(X.shape[1]):
        valid = X[:, j][
            np.isfinite(X[:, j])
        ]

        med[j] = (
            np.median(valid)
            if len(valid)
            else 0.0
        )

    Xi = X.copy()

    for j in range(X.shape[1]):
        bad = ~np.isfinite(Xi[:, j])
        Xi[bad, j] = med[j]

    mean = Xi.mean(axis=0)
    std = Xi.std(axis=0)

    keep = std > 1e-12

    return med, mean, std, keep


def transform(X, prep):
    med, mean, std, keep = prep

    X = X.copy()

    for j in range(X.shape[1]):
        bad = ~np.isfinite(X[:, j])
        X[bad, j] = med[j]

    X = X[:, keep]
    mean = mean[keep]
    std = std[keep]

    return (X - mean) / std


def ridge_fit(X, y, alpha):
    intercept = float(np.mean(y))
    yc = y - intercept

    if X.shape[1] == 0:
        beta = np.zeros(0)
    else:
        beta = np.linalg.solve(
            X.T @ X
            + alpha * np.eye(X.shape[1]),
            X.T @ yc,
        )

    return intercept, beta


def ridge_predict(model, X):
    intercept, beta = model
    return intercept + X @ beta


def calc_metrics(y, p):
    err = p - y

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    denom = float(
        np.sum(
            (y - np.mean(y)) ** 2
        )
    )

    r2 = (
        float(
            1
            - np.sum(err ** 2) / denom
        )
        if denom > 0
        else np.nan
    )

    if (
        len(y) > 1
        and np.std(y) > 0
        and np.std(p) > 0
    ):
        pearson = float(
            np.corrcoef(y, p)[0, 1]
        )
    else:
        pearson = np.nan

    return {
        "n": len(y),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson": pearson,
    }


def choose_alpha(
    X,
    y,
    groups,
    alphas,
):
    unique = sorted(set(groups))

    if len(unique) < 3:
        return float(alphas[len(alphas) // 2])

    results = []

    for alpha in alphas:
        errors = []

        for g in unique:
            tr = groups != g
            va = groups == g

            if not tr.any() or not va.any():
                continue

            prep = fit_preprocessor(X[tr])

            Xtr = transform(X[tr], prep)
            Xva = transform(X[va], prep)

            model = ridge_fit(
                Xtr,
                y[tr],
                alpha,
            )

            pred = ridge_predict(
                model,
                Xva,
            )

            errors.append(
                np.mean(
                    np.abs(
                        pred - y[va]
                    )
                )
            )

        if errors:
            results.append(
                (
                    float(np.mean(errors)),
                    alpha,
                )
            )

    if not results:
        return float(alphas[len(alphas) // 2])

    results.sort()
    return float(results[0][1])


def run_group_cv(
    rows,
    features,
    model_name,
    args,
):
    X, y, groups, meta = build_xy(
        rows,
        features,
        args.outcome_col,
        args.group_col,
    )

    unique = sorted(set(groups))

    if len(unique) < 2:
        raise RuntimeError(
            "O3 requires at least two independent run_id values. "
            f"Found: {unique}"
        )

    predictions = []
    folds = []

    for test_run in unique:
        te = groups == test_run
        tr = groups != test_run

        alpha = choose_alpha(
            X[tr],
            y[tr],
            groups[tr],
            args.alphas,
        )

        prep = fit_preprocessor(
            X[tr]
        )

        Xtr = transform(
            X[tr],
            prep,
        )

        Xte = transform(
            X[te],
            prep,
        )

        model = ridge_fit(
            Xtr,
            y[tr],
            alpha,
        )

        pred = ridge_predict(
            model,
            Xte,
        )

        fm = calc_metrics(
            y[te],
            pred,
        )

        folds.append({
            "model": model_name,
            "test_run": test_run,
            "alpha": alpha,
            **fm,
        })

        indices = np.where(te)[0]

        for k, idx in enumerate(indices):
            predictions.append({
                "model": model_name,
                "run_id": meta[idx][args.group_col],
                "timepoint": meta[idx].get(
                    "timepoint",
                    "",
                ),
                "target_class": meta[idx].get(
                    "target_class",
                    "",
                ),
                "y_true": float(y[idx]),
                "y_pred": float(pred[k]),
            })

    ya = np.asarray([
        r["y_true"]
        for r in predictions
    ])

    pa = np.asarray([
        r["y_pred"]
        for r in predictions
    ])

    overall = calc_metrics(
        ya,
        pa,
    )

    overall.update({
        "model": model_name,
        "num_runs": len(unique),
        "num_features": len(features),
    })

    return overall, folds, predictions


def write_csv(path, rows):
    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )

        w.writeheader()
        w.writerows(rows)


def safe_json(x):
    if isinstance(x, dict):
        return {
            k: safe_json(v)
            for k, v in x.items()
        }

    if isinstance(x, list):
        return [
            safe_json(v)
            for v in x
        ]

    if isinstance(x, (float, np.floating)):
        return (
            float(x)
            if np.isfinite(x)
            else None
        )

    if isinstance(x, (int, np.integer)):
        return int(x)

    if isinstance(x, (bool, np.bool_)):
        return bool(x)

    return x


def main():
    args = parse_args()

    rows = read_rows(args.data)

    if not rows:
        raise RuntimeError("Input data is empty")

    columns = list(rows[0].keys())

    for c in [
        args.group_col,
        args.outcome_col,
        *CONFOUNDERS,
    ]:
        if c not in columns:
            raise RuntimeError(
                f"Missing required column: {c}"
            )

    pair_features = sorted(
        c
        for c in columns
        if c.startswith(
            args.pair_prefix
        )
    )

    higher_features = sorted(
        c
        for c in columns
        if c.startswith(
            args.higher_prefix
        )
    )

    if not pair_features:
        raise RuntimeError(
            "No complete pairwise features found"
        )

    if not higher_features:
        raise RuntimeError(
            "No higher/intervention features found"
        )

    models = {
        "M0_confounders":
            CONFOUNDERS,

        "M1_pairwise":
            CONFOUNDERS
            + pair_features,

        "M2_pairwise_plus_higher":
            CONFOUNDERS
            + pair_features
            + higher_features,
    }

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    overall_rows = []
    fold_rows = []
    predictions = []

    for name, features in models.items():
        print("Running", name)

        overall, folds, preds = run_group_cv(
            rows,
            features,
            name,
            args,
        )

        overall_rows.append(overall)
        fold_rows.extend(folds)
        predictions.extend(preds)

    write_csv(
        args.out_dir / "overall_metrics.csv",
        overall_rows,
    )

    write_csv(
        args.out_dir / "run_level_cv_metrics.csv",
        fold_rows,
    )

    write_csv(
        args.out_dir / "heldout_predictions.csv",
        predictions,
    )

    table = {
        r["model"]: r
        for r in overall_rows
    }

    m1 = table["M1_pairwise"]
    m2 = table["M2_pairwise_plus_higher"]

    comparison = {
        "delta_mae_M2_minus_M1":
            m2["mae"] - m1["mae"],

        "delta_rmse_M2_minus_M1":
            m2["rmse"] - m1["rmse"],

        "delta_r2_M2_minus_M1":
            (
                m2["r2"] - m1["r2"]
                if (
                    np.isfinite(m2["r2"])
                    and np.isfinite(m1["r2"])
                )
                else None
            ),

        "higher_improves_mae":
            m2["mae"] < m1["mae"],

        "higher_improves_rmse":
            m2["rmse"] < m1["rmse"],
    }

    summary = {
        "split":
            "leave-one-entire-run-out",

        "models":
            table,

        "num_pairwise_features":
            len(pair_features),

        "num_higher_features":
            len(higher_features),

        "comparison":
            comparison,

        "interpretation":
            "M2 must improve held-out-run prediction over the "
            "complete-pairwise M1 baseline before claiming "
            "additional higher-order utility.",
    }

    with (
        args.out_dir
        / "o3_o4_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            safe_json(summary),
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("O3/O4 ANALYSIS COMPLETE")
    print("M1 MAE:", m1["mae"])
    print("M2 MAE:", m2["mae"])
    print(
        "Delta MAE:",
        comparison[
            "delta_mae_M2_minus_M1"
        ],
    )


if __name__ == "__main__":
    main()
