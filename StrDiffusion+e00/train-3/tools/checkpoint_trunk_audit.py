# -*- coding: utf-8 -*-
"""Compare the original StrDiffusion trunk weights with a later wrapper checkpoint.

This is a no-training diagnostic.  It answers a narrow question:

    "After BrushNet/MGLC/Mu-Denoiser training, how far did the original
     ConditionalUNet trunk drift from the known-good baseline checkpoint?"

Run it in the same Python environment that can load the training checkpoints.
The script only reads checkpoints and writes a small Markdown/JSON report.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch


DEFAULT_BASELINE = (
    "/home/610-wws/Impainting/StrDiffusion+e00/train/texture/"
    "config/inpainting/log/ir-sde/models/best_G.pth"
)
DEFAULT_CURRENT = (
    "/home/610-wws/Impainting/StrDiffusion+e00/train-3/texture/"
    "config/inpainting/log/ir-sde-brushnet-ft-x7/models/48000_G.pth"
)


EXTRA_PREFIXES = (
    "brushnet.",
    "mglc_",
    "mu_denoiser.",
    "main_guidance_proj.",
)


def _load_state(path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ("params", "state_dict", "model", "netG"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    if not isinstance(ckpt, dict):
        raise TypeError(f"Unsupported checkpoint object type: {type(ckpt)!r}")

    state: Dict[str, torch.Tensor] = {}
    for key, value in ckpt.items():
        if not torch.is_tensor(value):
            continue
        if key.startswith("module."):
            key = key[7:]
        state[key] = value.detach().cpu()
    return state


def _is_extra_key(key: str) -> bool:
    return key.startswith(EXTRA_PREFIXES)


def _group_key(key: str) -> str:
    parts = key.split(".")
    if key.startswith("downs.") and len(parts) >= 2:
        return f"downs.{parts[1]}"
    if key.startswith("ups.") and len(parts) >= 2:
        return f"ups.{parts[1]}"
    if key.startswith("mid_"):
        return parts[0]
    if key.startswith("final_"):
        return parts[0]
    if key.startswith("init_conv"):
        return "init_conv"
    if key.startswith("time_mlp"):
        return "time_mlp"
    return parts[0]


def _tensor_stats(current: torch.Tensor, baseline: torch.Tensor) -> Tuple[int, float, float, float]:
    cur = current.float()
    base = baseline.float()
    diff = cur - base
    n = diff.numel()
    diff_sq = float(diff.pow(2).sum().item())
    base_sq = float(base.pow(2).sum().item())
    abs_sum = float(diff.abs().sum().item())
    return n, diff_sq, base_sq, abs_sum


def compare(baseline_path: str, current_path: str) -> dict:
    baseline = _load_state(baseline_path)
    current = _load_state(current_path)

    shared = []
    shape_mismatch = []
    for key, base_value in baseline.items():
        if _is_extra_key(key):
            continue
        cur_value = current.get(key)
        if cur_value is None:
            continue
        if tuple(cur_value.shape) != tuple(base_value.shape):
            shape_mismatch.append(
                {
                    "key": key,
                    "baseline_shape": list(base_value.shape),
                    "current_shape": list(cur_value.shape),
                }
            )
            continue
        shared.append(key)

    missing_from_current = sorted(
        key for key in baseline.keys() if not _is_extra_key(key) and key not in current
    )
    extra_in_current = sorted(key for key in current.keys() if key not in baseline)

    groups = defaultdict(lambda: {"tensors": 0, "elements": 0, "diff_sq": 0.0, "base_sq": 0.0, "abs_sum": 0.0})
    global_stats = {"tensors": 0, "elements": 0, "diff_sq": 0.0, "base_sq": 0.0, "abs_sum": 0.0}

    top_keys = []
    for key in shared:
        n, diff_sq, base_sq, abs_sum = _tensor_stats(current[key], baseline[key])
        g = groups[_group_key(key)]
        g["tensors"] += 1
        g["elements"] += n
        g["diff_sq"] += diff_sq
        g["base_sq"] += base_sq
        g["abs_sum"] += abs_sum

        global_stats["tensors"] += 1
        global_stats["elements"] += n
        global_stats["diff_sq"] += diff_sq
        global_stats["base_sq"] += base_sq
        global_stats["abs_sum"] += abs_sum

        rel = (diff_sq / max(base_sq, 1e-12)) ** 0.5
        top_keys.append(
            {
                "key": key,
                "numel": n,
                "rms_diff": (diff_sq / max(n, 1)) ** 0.5,
                "mean_abs_diff": abs_sum / max(n, 1),
                "relative_rms": rel,
            }
        )

    def finalize(stats: dict) -> dict:
        n = max(stats["elements"], 1)
        base_sq = max(stats["base_sq"], 1e-12)
        return {
            "tensors": stats["tensors"],
            "elements": stats["elements"],
            "rms_diff": (stats["diff_sq"] / n) ** 0.5,
            "mean_abs_diff": stats["abs_sum"] / n,
            "relative_rms": (stats["diff_sq"] / base_sq) ** 0.5,
        }

    group_report = {
        name: finalize(stats)
        for name, stats in sorted(
            groups.items(),
            key=lambda item: finalize(item[1])["relative_rms"],
            reverse=True,
        )
    }

    return {
        "baseline_path": baseline_path,
        "current_path": current_path,
        "baseline_tensor_count": len(baseline),
        "current_tensor_count": len(current),
        "shared_trunk_tensor_count": len(shared),
        "missing_from_current_count": len(missing_from_current),
        "extra_in_current_count": len(extra_in_current),
        "shape_mismatch_count": len(shape_mismatch),
        "global_trunk_drift": finalize(global_stats),
        "groups": group_report,
        "top_changed_keys": sorted(top_keys, key=lambda item: item["relative_rms"], reverse=True)[:30],
        "missing_from_current_sample": missing_from_current[:50],
        "extra_in_current_sample": extra_in_current[:80],
        "shape_mismatch_sample": shape_mismatch[:30],
    }


def write_markdown(report: dict, path: Path) -> None:
    lines = []
    lines.append("# StrDiffusion trunk checkpoint drift audit")
    lines.append("")
    lines.append(f"- baseline: `{report['baseline_path']}`")
    lines.append(f"- current: `{report['current_path']}`")
    lines.append(f"- shared trunk tensors: `{report['shared_trunk_tensor_count']}`")
    lines.append(f"- extra current tensors: `{report['extra_in_current_count']}`")
    lines.append("")
    g = report["global_trunk_drift"]
    lines.append("## Global trunk drift")
    lines.append("")
    lines.append(
        f"- relative_rms: `{g['relative_rms']:.6f}`; "
        f"rms_diff: `{g['rms_diff']:.6f}`; "
        f"mean_abs_diff: `{g['mean_abs_diff']:.6f}`"
    )
    lines.append("")
    lines.append("## Drift by module group")
    lines.append("")
    lines.append("| group | tensors | relative_rms | rms_diff | mean_abs_diff |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, stats in report["groups"].items():
        lines.append(
            f"| `{name}` | {stats['tensors']} | {stats['relative_rms']:.6f} | "
            f"{stats['rms_diff']:.6f} | {stats['mean_abs_diff']:.6f} |"
        )
    lines.append("")
    lines.append("## Top changed trunk tensors")
    lines.append("")
    lines.append("| key | relative_rms | rms_diff | mean_abs_diff |")
    lines.append("|---|---:|---:|---:|")
    for item in report["top_changed_keys"]:
        lines.append(
            f"| `{item['key']}` | {item['relative_rms']:.6f} | "
            f"{item['rms_diff']:.6f} | {item['mean_abs_diff']:.6f} |"
        )
    lines.append("")
    lines.append("## Extra current keys sample")
    lines.append("")
    for key in report["extra_in_current_sample"]:
        lines.append(f"- `{key}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    parser.add_argument("--current", default=DEFAULT_CURRENT)
    parser.add_argument(
        "--out",
        default="./STRDIFFUSION_CHECKPOINT_DRIFT_48000.md",
        help="Markdown report path.",
    )
    parser.add_argument("--json-out", default=None, help="Optional JSON report path.")
    args = parser.parse_args(argv)

    report = compare(args.baseline, args.current)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(report, out)

    if args.json_out:
        json_out = Path(args.json_out)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote Markdown report: {out}")
    print(
        "Global trunk drift: "
        f"relative_rms={report['global_trunk_drift']['relative_rms']:.6f}, "
        f"shared_tensors={report['shared_trunk_tensor_count']}, "
        f"extra_current={report['extra_in_current_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
