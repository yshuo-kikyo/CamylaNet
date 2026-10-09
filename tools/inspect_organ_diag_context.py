#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    return p.parse_args()


def compact_value(x):
    if isinstance(x, dict):
        return f"dict(keys={list(x.keys())})"
    if isinstance(x, (list, tuple)):
        if len(x) <= 10:
            return repr(x)
        return f"{type(x).__name__}(len={len(x)})"
    if torch.is_tensor(x):
        return f"Tensor(shape={tuple(x.shape)}, dtype={x.dtype})"
    return repr(x)


def main():
    args = parse_args()

    ckpt = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    print("=" * 80)
    print("CHECKPOINT")
    print(args.checkpoint)
    print("=" * 80)

    print("\n[Top-level keys]")
    for k in ckpt.keys():
        print(f"{k}: {compact_value(ckpt[k])}")

    print("\n[Likely metadata]")
    for k in [
        "trainer_name",
        "init_args",
        "current_epoch",
        "logging",
        "best_ema",
        "inference_allowed_mirroring_axes",
        "network_weights",
        "optimizer_state",
        "grad_scaler_state",
    ]:
        if k in ckpt:
            print(f"{k}: {compact_value(ckpt[k])}")

    init_args = ckpt.get("init_args", {})

    print("\n[init_args]")
    if isinstance(init_args, dict):
        for k, v in init_args.items():
            print(f"{k}: {compact_value(v)}")
    else:
        print(type(init_args), init_args)

    dataset_json = None

    if isinstance(init_args, dict):
        dataset_json = init_args.get("dataset_json")

    if dataset_json is None:
        dataset_json = ckpt.get("dataset_json")

    print("\n[dataset_json]")
    if isinstance(dataset_json, dict):
        print(json.dumps(dataset_json, indent=2, ensure_ascii=False))
    else:
        print("Not embedded directly in checkpoint.")

    state = ckpt.get("network_weights", None)

    if state is not None:
        print("\n[Network state_dict key preview]")
        keys = list(state.keys())

        for k in keys[:100]:
            print(k, tuple(state[k].shape))

        print("\nTotal network tensors:", len(keys))

        print("\n[Candidate decoder/head keys]")
        for k in keys:
            kl = k.lower()
            if (
                "decoder" in kl
                or "seg_layers" in kl
                or "segmentation" in kl
                or "output" in kl
                or "final" in kl
            ):
                print(k, tuple(state[k].shape))

    print("\nDONE")


if __name__ == "__main__":
    main()
