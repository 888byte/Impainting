# -*- coding: utf-8 -*-
"""Create a no-retrain hybrid checkpoint.

The hybrid keeps the current x7 checkpoint's added modules (BrushNet/MGLC/Mu)
but replaces every shared original ConditionalUNet trunk tensor with the
known-good baseline StrDiffusion checkpoint.

This is intended for inference ablation only:

    baseline trunk + x7 added modules

It does not modify the input checkpoints.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch


DEFAULT_BASELINE = (
    "/home/610-wws/Impainting/StrDiffusion+e00/train/texture/"
    "config/inpainting/log/ir-sde/models/best_G.pth"
)
DEFAULT_CURRENT = (
    "/home/610-wws/Impainting/StrDiffusion+e00/train-3/texture/"
    "config/inpainting/log/ir-sde-brushnet-ft-x7/models/32000_G.pth"
)
DEFAULT_OUT = (
    "/home/610-wws/Impainting/StrDiffusion+e00/train-3/texture/"
    "config/inpainting/log/ir-sde-brushnet-ft-x7/models/"
    "32000_G.baseline_trunk_x7_extra.pth"
)


EXTRA_PREFIXES = (
    "brushnet.",
    "mglc_",
    "mu_denoiser.",
    "main_guidance_proj.",
)


def _load_raw(path: str):
    return torch.load(path, map_location="cpu")


def _as_state_dict(obj) -> Tuple[Dict[str, torch.Tensor], str | None]:
    if isinstance(obj, dict):
        for key in ("params", "state_dict", "model", "netG"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key], key
        return obj, None
    raise TypeError(f"Unsupported checkpoint object type: {type(obj)!r}")


def _normalize_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    normalized = {}
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        normalized[key[7:] if key.startswith("module.") else key] = value.detach().cpu()
    return normalized


def _is_extra_key(key: str) -> bool:
    return key.startswith(EXTRA_PREFIXES)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    parser.add_argument("--current", default=DEFAULT_CURRENT)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    baseline_raw = _load_raw(args.baseline)
    current_raw = _load_raw(args.current)
    baseline_state_raw, _ = _as_state_dict(baseline_raw)
    current_state_raw, current_container_key = _as_state_dict(current_raw)

    baseline = _normalize_state(baseline_state_raw)
    current = _normalize_state(current_state_raw)
    hybrid = dict(current)

    replaced = 0
    skipped_shape = []
    missing_in_current = []
    for key, value in baseline.items():
        if _is_extra_key(key):
            continue
        if key not in current:
            missing_in_current.append(key)
            continue
        if tuple(current[key].shape) != tuple(value.shape):
            skipped_shape.append((key, tuple(value.shape), tuple(current[key].shape)))
            continue
        hybrid[key] = value
        replaced += 1

    extra_kept = sum(1 for key in hybrid if key not in baseline)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if current_container_key is None:
        torch.save(hybrid, out_path)
    else:
        current_raw[current_container_key] = hybrid
        torch.save(current_raw, out_path)

    print(f"Wrote hybrid checkpoint: {out_path}")
    print(f"Replaced shared trunk tensors: {replaced}")
    print(f"Kept extra current tensors: {extra_kept}")
    print(f"Missing baseline trunk keys in current: {len(missing_in_current)}")
    print(f"Skipped shape-mismatch tensors: {len(skipped_shape)}")
    if skipped_shape:
        print("Shape mismatch sample:")
        for item in skipped_shape[:20]:
            print("  ", item)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
