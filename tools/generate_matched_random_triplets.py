#!/usr/bin/env python3

import argparse
import csv
import json
import random
from itertools import combinations
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--o1-records", type=Path, required=True)
    p.add_argument(
        "--structured-triplet",
        nargs=3,
        type=int,
        action="append",
        required=True,
    )
    p.add_argument("--num-random", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--exclude-overlap", action="store_true")
    return p.parse_args()


def load_sizes(path):
    grouped = {}

    with path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("crop_mode") != "same_crop":
                continue

            try:
                c = int(r["class_id"])
                v = float(r["full_voxel_count"])
            except Exception:
                continue

            if np.isfinite(v) and v > 0:
                grouped.setdefault(c, []).append(v)

    return {
        c: float(np.median(v))
        for c, v in grouped.items()
        if v
    }


def profile(triplet, sizes):
    return np.sort(
        np.log1p(
            [sizes[c] for c in triplet]
        )
    )


def distance(a, b, sizes):
    pa = profile(a, sizes)
    pb = profile(b, sizes)
    return float(
        np.sqrt(
            np.mean((pa - pb) ** 2)
        )
    )


def main():
    args = parse_args()
    random.seed(args.seed)

    sizes = load_sizes(args.o1_records)

    if len(sizes) < 3:
        raise RuntimeError(
            "Not enough organ-size records. "
            "Run O1 first."
        )

    available = sorted(sizes)
    all_triplets = list(
        combinations(available, 3)
    )

    structured = [
        tuple(sorted(x))
        for x in args.structured_triplet
    ]

    result = {
        "seed": args.seed,
        "matching_metric":
            "RMS distance of sorted log1p median organ volumes",
        "organ_median_voxels": sizes,
        "structured": [],
    }

    for s in structured:
        if any(c not in sizes for c in s):
            raise RuntimeError(
                f"Missing O1 size information for {s}"
            )

        candidates = []

        for r in all_triplets:
            if r == s:
                continue

            if args.exclude_overlap:
                if set(r) & set(s):
                    continue

            candidates.append(
                (distance(s, r, sizes), r)
            )

        candidates.sort(key=lambda x: x[0])

        # choose among a near-matched pool to avoid one deterministic pattern
        pool_n = min(
            len(candidates),
            max(args.num_random * 5, args.num_random),
        )

        pool = candidates[:pool_n]
        random.shuffle(pool)
        selected = sorted(
            pool[:args.num_random],
            key=lambda x: x[0],
        )

        result["structured"].append({
            "triplet": list(s),
            "matched_random": [
                {
                    "triplet": list(r),
                    "distance": d,
                }
                for d, r in selected
            ],
        })

    args.out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.out.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("MATCHED RANDOM TRIPLETS READY")
    print(args.out)

    for item in result["structured"]:
        print("Structured:", item["triplet"])
        for x in item["matched_random"][:5]:
            print(
                " ",
                x["triplet"],
                "distance=",
                round(x["distance"], 5),
            )


if __name__ == "__main__":
    main()
