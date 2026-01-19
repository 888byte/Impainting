#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocess pigment fading RGB logs + Raman/XRD excels into NPZ for training.

Fix for your current dataset:
- meta_json provides patch->sheet mapping, but Excel sheet names may differ in component order:
  e.g. meta_json: "66 铅白+密陀僧" but Excel: "66 密陀僧+铅白"
  -> old exact match fails, spectra become missing/all-zeros.

This version:
1) Auto-adapts sheet prefix (66/76) to each experiment tag (from --exp_tags or inferred by humidity).
2) Robustly resolves sheet names by canonicalizing base name (split by '+', sort components).
3) Writes meta_has_raman/meta_has_xrd (and has_raman/has_xrd alias) into NPZ.
4) Fail-fast: if Raman/XRD excel is provided but no spectra loaded, raise error unless --allow_empty_spectra.

Output NPZ keys:
- x0: (N, 2, 3) float32 normalized Lab, [t0(original, masked), t1(observed current)]
- mask: (N, 2, 3) float32, 1=observed, 0=missing
- raman: (N, R) float32 (optional)
- xrd: (N, X) float32 (optional)
- meta_has_raman / meta_has_xrd: (N,) int64 (optional)
- has_raman / has_xrd: (N,) int64 (alias, optional)
- raman_peaks / xrd_peaks: (N, 2*K) float32 (optional)
- meta_patch_id/meta_t/meta_exp_id/meta_exp_tag/meta_exp_humidity_median

Example:
  python -m pigment_task.preprocess_pigment \
    --rgb_logs "/path/12.19-12.23.txt,/path/12.19-12.23_Right.txt" \
    --output_dir data/pigment_npz_v2 \
    --use_patches "1-9" \
    --meta_json pigment_task/pigment_meta_example.json \
    --raman_excel "/path/拉曼.xlsx" \
    --xrd_excel "/path/xrd.xlsx" \
    --split_mode group_exp_patch
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# allow running from repo root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pigment_task.color_utils import LabNorm, rgb_to_lab
from pigment_task.io_utils import (
    adapt_sheet_name,
    extract_peak_features_xy,
    filter_rgb_log_by_humidity,
    guess_experiment_tag,
    load_raman_excel_sheet,
    load_xrd_excel_sheet,
    load_xy_from_excel,
    parse_rgb_log_txt,
)

# -----------------------------
# Helpers
# -----------------------------

def _parse_int_list(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    if "-" in s and "," not in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _load_meta_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_patch_sheet_maps(meta: Dict, exp_tag: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Return (patch_to_raman_sheet, patch_to_xrd_sheet) for a given experiment tag.

    Backward-compatible behaviors:
    - If meta has top-level patch_to_*, we use them and adapt the leading humidity prefix to exp_tag.
    - If meta has meta["experiments"][exp_tag]["patch_to_*"], we use those (also adapted).
    """
    if "experiments" in meta and isinstance(meta["experiments"], dict) and exp_tag in meta["experiments"]:
        m = meta["experiments"][exp_tag]
        p2r = {str(k): str(v) for k, v in m.get("patch_to_raman_sheet", {}).items()}
        p2x = {str(k): str(v) for k, v in m.get("patch_to_xrd_sheet", {}).items()}
    else:
        p2r = {str(k): str(v) for k, v in meta.get("patch_to_raman_sheet", {}).items()}
        p2x = {str(k): str(v) for k, v in meta.get("patch_to_xrd_sheet", {}).items()}

    # Adapt leading prefix (66/76) if present
    p2r = {k: adapt_sheet_name(v, exp_tag) for k, v in p2r.items()}
    p2x = {k: adapt_sheet_name(v, exp_tag) for k, v in p2x.items()}
    return p2r, p2x


_SHEET_PREFIX_RE = re.compile(r"^\s*(\d{2})\s+")

def _split_prefix_base(sheet_name: str) -> Tuple[str, str]:
    """
    "66 密陀僧+铅白" -> ("66", "密陀僧+铅白")
    If no prefix, prefix="".
    """
    s = (sheet_name or "").strip()
    m = _SHEET_PREFIX_RE.match(s)
    if not m:
        return "", s
    prefix = m.group(1)
    base = s[m.end():].strip()
    return prefix, base


def _canonical_base_name(base: str) -> str:
    """
    Make "铅白+密陀僧" and "密陀僧 + 铅白" canonical to the same key.
    """
    b = (base or "").strip().replace(" ", "")
    if not b:
        return ""
    parts = [p.strip() for p in b.split("+") if p.strip()]
    # sort to ignore order
    parts = sorted(parts)
    return "+".join(parts)


# Excel cache to avoid repeated IO
_EXCEL_CACHE: Dict[str, Dict] = {}

def _build_excel_cache(excel_path: str) -> Dict:
    if excel_path in _EXCEL_CACHE:
        return _EXCEL_CACHE[excel_path]

    xls = pd.ExcelFile(excel_path)
    sheet_names = list(xls.sheet_names)
    sheet_set = set(sheet_names)

    # index by prefix + canonical base
    canon_index: Dict[str, Dict[str, str]] = {}
    nospace_index: Dict[str, str] = {}

    for s in sheet_names:
        ss = str(s).strip()
        nospace_index[ss.replace(" ", "")] = ss

        prefix, base = _split_prefix_base(ss)
        canon = _canonical_base_name(base)
        canon_index.setdefault(prefix, {})
        # if collision, keep the first (stable)
        if canon and canon not in canon_index[prefix]:
            canon_index[prefix][canon] = ss

    cache = {
        "sheet_names": sheet_names,
        "sheet_set": sheet_set,
        "canon_index": canon_index,
        "nospace_index": nospace_index,
    }
    _EXCEL_CACHE[excel_path] = cache
    return cache


def _resolve_sheet_name(excel_path: str, wanted_sheet: str, exp_tag: str) -> str:
    """
    Try to resolve wanted_sheet to an actual sheet name in the workbook.
    Steps:
      1) adapt prefix to exp_tag (66/76)
      2) exact match
      3) match by removing spaces
      4) canonical match on base name (order-insensitive by '+')
    Return "" if not found.
    """
    if not excel_path or not wanted_sheet:
        return ""

    cache = _build_excel_cache(excel_path)

    # 1) adapt prefix (if any)
    want = adapt_sheet_name(wanted_sheet, exp_tag).strip()

    # 2) exact
    if want in cache["sheet_set"]:
        return want

    # 3) remove spaces
    want_ns = want.replace(" ", "")
    if want_ns in cache["nospace_index"]:
        return cache["nospace_index"][want_ns]

    # 4) canonical base match (A+B == B+A)
    prefix, base = _split_prefix_base(want)
    canon = _canonical_base_name(base)
    if prefix in cache["canon_index"] and canon in cache["canon_index"][prefix]:
        return cache["canon_index"][prefix][canon]

    # If prefix missing (rare), try all prefixes
    if not prefix:
        for px, mp in cache["canon_index"].items():
            if canon in mp:
                return mp[canon]

    return ""


def _is_nonzero(arr: np.ndarray, eps: float = 1e-8) -> bool:
    if arr is None:
        return False
    a = np.asarray(arr, dtype=np.float32)
    return bool(np.any(np.abs(a) > eps))


def _group_split_indices(
    groups: Sequence[Tuple],
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert len(groups) > 0
    uniq = list(dict.fromkeys(groups))  # stable unique
    rng = np.random.default_rng(int(seed))
    rng.shuffle(uniq)

    n = len(uniq)
    n_test = int(round(n * float(test_ratio))) if test_ratio > 0 else 0
    n_val = int(round(n * float(val_ratio))) if val_ratio > 0 else 0

    if test_ratio > 0 and n_test == 0 and n >= 3:
        n_test = 1
    if val_ratio > 0 and n_val == 0 and n >= 3:
        n_val = 1

    n_test = min(n_test, n)
    n_val = min(n_val, n - n_test)

    test_keys = set(uniq[:n_test])
    val_keys = set(uniq[n_test:n_test + n_val])
    train_keys = set(uniq[n_test + n_val:])

    idx = np.arange(len(groups))
    train_idx = idx[[g in train_keys for g in groups]]
    val_idx = idx[[g in val_keys for g in groups]]
    test_idx = idx[[g in test_keys for g in groups]]
    return train_idx, val_idx, test_idx


def _random_split_indices(
    n: int,
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(round(n * float(test_ratio))) if test_ratio > 0 else 0
    n_val = int(round(n * float(val_ratio))) if val_ratio > 0 else 0
    n_test = min(n_test, n)
    n_val = min(n_val, n - n_test)
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]
    return train_idx, val_idx, test_idx


def _save_npz(path: str, arrays: Dict[str, np.ndarray]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arrays)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb_logs", type=str, required=True,
                    help="Comma-separated TXT log paths (each treated as an independent experiment)")
    ap.add_argument("--exp_tags", type=str, default="",
                    help="Optional comma-separated experiment tags (e.g., 66,76) aligned with rgb_logs order")
    ap.add_argument("--hum_tol", type=float, default=15.0,
                    help="Filter RGB blocks whose humidity deviates from median by more than this (%%RH)")
    ap.add_argument("--use_patches", type=str, default="1-9",
                    help="Patch ids, e.g. '1-9' or '1,2,3'")

    ap.add_argument("--meta_json", type=str, default="",
                    help="JSON mapping patch->(raman/xrd sheet). Can be for 66; will auto-adapt to 76.")
    ap.add_argument("--raman_excel", type=str, default="", help="Raman Excel (.xlsx)")
    ap.add_argument("--xrd_excel", type=str, default="", help="XRD Excel (.xlsx)")

    ap.add_argument("--peak_features", action="store_true", help="Compute Raman/XRD peak feature vectors (optional)")
    ap.add_argument("--peak_top_k", type=int, default=32)
    ap.add_argument("--peak_prominence", type=float, default=0.05)

    ap.add_argument("--allow_empty_spectra", action="store_true",
                    help="If set, do NOT error even if no spectra are loaded (not recommended).")
    ap.add_argument("--print_sheet_map", action="store_true",
                    help="Print resolved sheet mapping per experiment (debug).")

    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--split_mode", type=str, default="group_exp_patch",
                    choices=["random", "group_exp_patch", "group_patch", "group_exp"],
                    help="How to split train/val/test. group_* prevents leakage across related samples.")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    args = ap.parse_args()

    rgb_logs = [p.strip() for p in args.rgb_logs.split(",") if p.strip()]
    if not rgb_logs:
        raise ValueError("No rgb_logs provided")

    exp_tags_in: List[str] = [t.strip() for t in args.exp_tags.split(",") if t.strip()]
    if exp_tags_in and len(exp_tags_in) != len(rgb_logs):
        raise ValueError("--exp_tags length must match --rgb_logs length")

    patch_ids = _parse_int_list(args.use_patches)
    if not patch_ids:
        raise ValueError("No patches selected")

    meta: Dict = {}
    raman_len = 1024
    xrd_len = 2048
    if args.meta_json:
        meta = _load_meta_json(args.meta_json)
        raman_len = int(meta.get("raman_len", raman_len))
        xrd_len = int(meta.get("xrd_len", xrd_len))

    lab_norm = LabNorm()

    # Pre-load spectra per (exp, patch)
    raman_by_exp_patch: Dict[Tuple[int, int], np.ndarray] = {}
    xrd_by_exp_patch: Dict[Tuple[int, int], np.ndarray] = {}
    has_raman_by_exp_patch: Dict[Tuple[int, int], int] = {}
    has_xrd_by_exp_patch: Dict[Tuple[int, int], int] = {}
    raman_peaks_by_exp_patch: Dict[Tuple[int, int], np.ndarray] = {}
    xrd_peaks_by_exp_patch: Dict[Tuple[int, int], np.ndarray] = {}

    exp_meta: List[Dict] = []

    loaded_raman = 0
    loaded_xrd = 0

    for exp_id, log_path in enumerate(rgb_logs):
        log = parse_rgb_log_txt(log_path)
        log = filter_rgb_log_by_humidity(log, tol=float(args.hum_tol))
        hum_med = float(np.median(np.asarray(log.humidity, dtype=np.float32)))

        exp_tag = exp_tags_in[exp_id] if exp_tags_in else guess_experiment_tag(hum_med)
        exp_meta.append({
            "exp_id": exp_id,
            "log_path": log_path,
            "exp_tag": exp_tag,
            "humidity_median": hum_med,
            "T_blocks": int(log.rgb.shape[0]),
        })

        # Mapping for this experiment (adapted to exp_tag)
        p2r, p2x = _get_patch_sheet_maps(meta, exp_tag) if meta else ({}, {})

        # Resolve + load Raman
        if args.raman_excel:
            if not meta:
                raise ValueError("You passed --raman_excel but did not provide --meta_json. "
                                 "Please provide meta_json mapping patch->sheet.")
            if args.print_sheet_map:
                print(f"[INFO] (exp_tag={exp_tag}) Raman raw mapping: {p2r}")

            for pid in patch_ids:
                raw = p2r.get(str(pid), "")
                sheet = _resolve_sheet_name(args.raman_excel, raw, exp_tag)
                if not sheet:
                    has_raman_by_exp_patch[(exp_id, pid)] = 0
                    continue
                try:
                    spec = load_raman_excel_sheet(args.raman_excel, sheet, new_len=raman_len)
                    raman_by_exp_patch[(exp_id, pid)] = spec.astype(np.float32)
                    hr = 1 if _is_nonzero(spec) else 0
                    has_raman_by_exp_patch[(exp_id, pid)] = hr
                    loaded_raman += hr

                    if args.peak_features:
                        x, y = load_xy_from_excel(args.raman_excel, sheet, x_col=0, y_col=1, header=0)
                        raman_peaks_by_exp_patch[(exp_id, pid)] = extract_peak_features_xy(
                            x, y, top_k=int(args.peak_top_k), prominence=float(args.peak_prominence)
                        )
                except Exception as e:
                    print(f"[WARN] Raman load failed exp_tag={exp_tag} patch={pid} sheet='{sheet}' (raw='{raw}'): {e}")
                    has_raman_by_exp_patch[(exp_id, pid)] = 0

            if args.print_sheet_map:
                resolved = {str(pid): _resolve_sheet_name(args.raman_excel, p2r.get(str(pid), ""), exp_tag)
                            for pid in patch_ids}
                print(f"[INFO] (exp_tag={exp_tag}) Raman resolved mapping: {resolved}")

        # Resolve + load XRD
        if args.xrd_excel:
            if not meta:
                raise ValueError("You passed --xrd_excel but did not provide --meta_json. "
                                 "Please provide meta_json mapping patch->sheet.")
            if args.print_sheet_map:
                print(f"[INFO] (exp_tag={exp_tag}) XRD raw mapping: {p2x}")

            for pid in patch_ids:
                raw = p2x.get(str(pid), "")
                sheet = _resolve_sheet_name(args.xrd_excel, raw, exp_tag)
                if not sheet:
                    has_xrd_by_exp_patch[(exp_id, pid)] = 0
                    continue
                try:
                    spec = load_xrd_excel_sheet(args.xrd_excel, sheet, new_len=xrd_len)
                    xrd_by_exp_patch[(exp_id, pid)] = spec.astype(np.float32)
                    hx = 1 if _is_nonzero(spec) else 0
                    has_xrd_by_exp_patch[(exp_id, pid)] = hx
                    loaded_xrd += hx

                    if args.peak_features:
                        x, y = load_xy_from_excel(args.xrd_excel, sheet, x_col=0, y_col=1, header=0)
                        xrd_peaks_by_exp_patch[(exp_id, pid)] = extract_peak_features_xy(
                            x, y, top_k=int(args.peak_top_k), prominence=float(args.peak_prominence)
                        )
                except Exception as e:
                    print(f"[WARN] XRD load failed exp_tag={exp_tag} patch={pid} sheet='{sheet}' (raw='{raw}'): {e}")
                    has_xrd_by_exp_patch[(exp_id, pid)] = 0

            if args.print_sheet_map:
                resolved = {str(pid): _resolve_sheet_name(args.xrd_excel, p2x.get(str(pid), ""), exp_tag)
                            for pid in patch_ids}
                print(f"[INFO] (exp_tag={exp_tag}) XRD resolved mapping: {resolved}")

    # Fail-fast if user provided excels but we loaded nothing
    if args.raman_excel and not args.allow_empty_spectra and loaded_raman == 0:
        cache = _build_excel_cache(args.raman_excel)
        raise ValueError(
            "Raman excel provided but NO spectra loaded (loaded_raman=0). "
            "Check your meta_json mapping or sheet names.\n"
            f"Example available sheets: {cache['sheet_names'][:15]}"
        )
    if args.xrd_excel and not args.allow_empty_spectra and loaded_xrd == 0:
        cache = _build_excel_cache(args.xrd_excel)
        raise ValueError(
            "XRD excel provided but NO spectra loaded (loaded_xrd=0). "
            "Check your meta_json mapping or sheet names.\n"
            f"Example available sheets: {cache['sheet_names'][:15]}"
        )

    # Build samples
    x0_list: List[np.ndarray] = []
    mask_list: List[np.ndarray] = []

    raman_list: List[np.ndarray] = []
    xrd_list: List[np.ndarray] = []
    has_raman_list: List[int] = []
    has_xrd_list: List[int] = []

    raman_peaks_list: List[np.ndarray] = []
    xrd_peaks_list: List[np.ndarray] = []

    meta_patch: List[int] = []
    meta_t: List[int] = []
    meta_exp_id: List[int] = []
    meta_exp_tag: List[int] = []
    meta_exp_hum: List[float] = []

    for exp_id, log_path in enumerate(rgb_logs):
        log = parse_rgb_log_txt(log_path)
        log = filter_rgb_log_by_humidity(log, tol=float(args.hum_tol))
        rgb = log.rgb  # (T,P,3)
        T, P, _ = rgb.shape
        hum_med = float(np.median(np.asarray(log.humidity, dtype=np.float32)))
        exp_tag_int = int(exp_meta[exp_id]["exp_tag"])

        patch_idx = [pid - 1 for pid in patch_ids]
        lab_series = rgb_to_lab(rgb[:, patch_idx, :])  # (T, len(patch), 3)
        lab0 = lab_series[0]  # (len(patch), 3)

        for p_i, pid in enumerate(patch_ids):
            for t in range(1, T):
                # sequence length=2: [t0 missing/original, t1 observed/current]
                seq = np.stack([lab0[p_i], lab_series[t, p_i]], axis=0)  # (2,3)
                seq_n = lab_norm.normalize(seq).astype(np.float32)
                mask = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)

                x0_list.append(seq_n)
                mask_list.append(mask)

                if args.raman_excel:
                    r = raman_by_exp_patch.get((exp_id, pid), np.zeros((raman_len,), dtype=np.float32))
                    hr = int(has_raman_by_exp_patch.get((exp_id, pid), 0))
                    raman_list.append(r.astype(np.float32))
                    has_raman_list.append(hr)
                    if args.peak_features:
                        rp = raman_peaks_by_exp_patch.get(
                            (exp_id, pid),
                            np.zeros((2 * int(args.peak_top_k),), dtype=np.float32),
                        )
                        raman_peaks_list.append(rp.astype(np.float32))

                if args.xrd_excel:
                    x = xrd_by_exp_patch.get((exp_id, pid), np.zeros((xrd_len,), dtype=np.float32))
                    hx = int(has_xrd_by_exp_patch.get((exp_id, pid), 0))
                    xrd_list.append(x.astype(np.float32))
                    has_xrd_list.append(hx)
                    if args.peak_features:
                        xp = xrd_peaks_by_exp_patch.get(
                            (exp_id, pid),
                            np.zeros((2 * int(args.peak_top_k),), dtype=np.float32),
                        )
                        xrd_peaks_list.append(xp.astype(np.float32))

                meta_patch.append(int(pid))
                meta_t.append(int(t))
                meta_exp_id.append(int(exp_id))
                meta_exp_tag.append(int(exp_tag_int))
                meta_exp_hum.append(float(hum_med))

    x0 = np.stack(x0_list, axis=0)
    mask = np.stack(mask_list, axis=0)
    arrays_all: Dict[str, np.ndarray] = {
        "x0": x0,
        "mask": mask,
        "meta_patch_id": np.asarray(meta_patch, dtype=np.int64),
        "meta_t": np.asarray(meta_t, dtype=np.int64),
        "meta_exp_id": np.asarray(meta_exp_id, dtype=np.int64),
        "meta_exp_tag": np.asarray(meta_exp_tag, dtype=np.int64),
        "meta_exp_humidity_median": np.asarray(meta_exp_hum, dtype=np.float32),
    }

    if args.raman_excel:
        arrays_all["raman"] = np.stack(raman_list, axis=0).astype(np.float32)
        arrays_all["meta_has_raman"] = np.asarray(has_raman_list, dtype=np.int64)
        arrays_all["has_raman"] = np.asarray(has_raman_list, dtype=np.int64)  # alias
        if args.peak_features:
            arrays_all["raman_peaks"] = np.stack(raman_peaks_list, axis=0).astype(np.float32)

    if args.xrd_excel:
        arrays_all["xrd"] = np.stack(xrd_list, axis=0).astype(np.float32)
        arrays_all["meta_has_xrd"] = np.asarray(has_xrd_list, dtype=np.int64)
        arrays_all["has_xrd"] = np.asarray(has_xrd_list, dtype=np.int64)  # alias
        if args.peak_features:
            arrays_all["xrd_peaks"] = np.stack(xrd_peaks_list, axis=0).astype(np.float32)

    N = int(x0.shape[0])
    if N == 0:
        raise ValueError("No samples generated. Check logs and patch selection.")

    # split
    if args.split_mode == "random":
        train_idx, val_idx, test_idx = _random_split_indices(N, args.seed, args.val_ratio, args.test_ratio)
        split_detail = {"mode": "random"}
    else:
        if args.split_mode == "group_exp_patch":
            groups = [(int(e), int(p)) for e, p in zip(meta_exp_id, meta_patch)]
        elif args.split_mode == "group_patch":
            groups = [(int(p),) for p in meta_patch]
        elif args.split_mode == "group_exp":
            groups = [(int(e),) for e in meta_exp_id]
        else:
            raise ValueError(f"Unknown split_mode: {args.split_mode}")

        train_idx, val_idx, test_idx = _group_split_indices(groups, args.seed, args.val_ratio, args.test_ratio)
        split_detail = {"mode": args.split_mode, "num_groups": len(set(groups))}

    def _slice(arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
        return arr[idx]

    arrays_train = {k: _slice(v, train_idx) for k, v in arrays_all.items()}
    arrays_val = {k: _slice(v, val_idx) for k, v in arrays_all.items()}
    arrays_test = {k: _slice(v, test_idx) for k, v in arrays_all.items()}

    os.makedirs(args.output_dir, exist_ok=True)
    _save_npz(os.path.join(args.output_dir, "train.npz"), arrays_train)
    _save_npz(os.path.join(args.output_dir, "val.npz"), arrays_val)
    _save_npz(os.path.join(args.output_dir, "test.npz"), arrays_test)
    _save_npz(os.path.join(args.output_dir, "all.npz"), arrays_all)

    meta_out = {
        "experiments": exp_meta,
        "patch_ids": patch_ids,
        "raman_len": raman_len,
        "xrd_len": xrd_len,
        "loaded_raman_cnt": int(loaded_raman),
        "loaded_xrd_cnt": int(loaded_xrd),
        "peak_features": bool(args.peak_features),
        "peak_top_k": int(args.peak_top_k),
        "split": {
            "seed": int(args.seed),
            "val_ratio": float(args.val_ratio),
            "test_ratio": float(args.test_ratio),
            **split_detail,
        },
        "N_total": N,
        "N_train": int(len(train_idx)),
        "N_val": int(len(val_idx)),
        "N_test": int(len(test_idx)),
    }
    with open(os.path.join(args.output_dir, "preprocess_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_out, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved to {args.output_dir}")
    print(json.dumps(meta_out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
