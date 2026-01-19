
"""
IO + preprocessing helpers for the pigment fading task.

- parse RGB log TXT (your sensor log)
- load Raman/XRD spectra from Excel
- resample spectra to fixed length
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import re

from scipy import sparse
from scipy.signal import savgol_filter, find_peaks


_RGB_LINE_RE = re.compile(r"NO\.(\d+)\s+R:(\d+)\s+G:(\d+)\s+B:(\d+)")
_TEMP_LINE_RE = re.compile(r"(\d+)\s*Temperature:\s*([0-9.]+)\s*C\s*Humidity:\s*([0-9.]+)")


@dataclass
class RGBLog:
    rgb: np.ndarray  # (T, P, 3) float64
    block_idx: List[int]  # length T
    temperature: List[float]  # length T
    humidity: List[float]  # length T


def parse_rgb_log_txt(path: str) -> RGBLog:
    """
    Parse the color sensor TXT log like:
      01Temperature: 23.97 C       Humidity: 62.69 %RH
      NO.1   R:196  G:242  B:242
      ...
    Returns:
      rgb array of shape (T, P, 3) where P = max NO. index.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f.readlines()]

    blocks: List[Tuple[int, float, float, List[Tuple[int, int, int, int]]]] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if "Temperature" in ln and "Humidity" in ln:
            m = _TEMP_LINE_RE.match(ln)
            if m:
                b_idx = int(m.group(1))
                temp = float(m.group(2))
                hum = float(m.group(3))
            else:
                # fallback
                b_idx = -1
                temp = float(re.search(r"Temperature:\s*([0-9.]+)", ln).group(1))
                hum = float(re.search(r"Humidity:\s*([0-9.]+)", ln).group(1))
            i += 1
            colors: List[Tuple[int, int, int, int]] = []
            while i < len(lines):
                ln2 = lines[i]
                m2 = _RGB_LINE_RE.match(ln2)
                if m2:
                    no = int(m2.group(1))
                    r = int(m2.group(2))
                    g = int(m2.group(3))
                    b = int(m2.group(4))
                    colors.append((no, r, g, b))
                    i += 1
                else:
                    break
            if colors:
                blocks.append((b_idx, temp, hum, colors))
        else:
            i += 1

    if not blocks:
        raise ValueError(f"No valid RGB blocks found in {path}")

    P = max(no for _, _, _, colors in blocks for no, _, _, _ in colors)
    T = len(blocks)
    rgb = np.zeros((T, P, 3), dtype=np.float64)
    block_idx: List[int] = []
    temperature: List[float] = []
    humidity: List[float] = []

    for t, (b_idx, temp, hum, colors) in enumerate(blocks):
        block_idx.append(b_idx)
        temperature.append(temp)
        humidity.append(hum)
        for no, r, g, b in colors:
            rgb[t, no - 1, :] = [r, g, b]

    return RGBLog(rgb=rgb, block_idx=block_idx, temperature=temperature, humidity=humidity)


def resample_1d(
    x: np.ndarray,
    y: np.ndarray,
    new_len: int,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
) -> np.ndarray:
    """
    Resample a 1D curve (x,y) to fixed length new_len by linear interpolation.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x_min is None:
        x_min = float(np.nanmin(x))
    if x_max is None:
        x_max = float(np.nanmax(x))
    new_x = np.linspace(x_min, x_max, new_len)
    # sort by x
    order = np.argsort(x)
    x_s = x[order]
    y_s = y[order]
    # remove NaNs
    ok = np.isfinite(x_s) & np.isfinite(y_s)
    x_s = x_s[ok]
    y_s = y_s[ok]
    if len(x_s) < 2:
        raise ValueError("Not enough valid points to resample.")
    new_y = np.interp(new_x, x_s, y_s)
    return new_y.astype(np.float32)


def _standardize_intensity(intensity: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    intensity = intensity.astype(np.float32)
    mu = float(np.mean(intensity))
    sigma = float(np.std(intensity))
    return (intensity - mu) / (sigma + eps)


def load_raman_excel_sheet(
    excel_path: str,
    sheet_name: str,
    new_len: int = 1024,
    standardize: bool = True,
) -> np.ndarray:
    """
    Load one Raman spectrum from an Excel sheet with columns like:
      拉曼位移(cm⁻¹), 强度(Intensity)
    Returns intensity vector of length new_len.
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    if df.shape[1] < 2:
        raise ValueError(f"Raman sheet {sheet_name} has <2 columns.")
    x = df.iloc[:, 0].to_numpy()
    y = df.iloc[:, 1].to_numpy()
    y_rs = resample_1d(x, y, new_len=new_len)
    if standardize:
        y_rs = _standardize_intensity(y_rs)
    return y_rs


def load_xrd_excel_sheet(
    excel_path: str,
    sheet_name: str,
    new_len: int = 2048,
    standardize: bool = True,
) -> np.ndarray:
    """
    Load one XRD spectrum from an Excel sheet with columns like:
      X, <intensity>
    Returns intensity vector of length new_len.
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    if df.shape[1] < 2:
        raise ValueError(f"XRD sheet {sheet_name} has <2 columns.")
    x = df.iloc[:, 0].to_numpy()
    y = df.iloc[:, 1].to_numpy()
    y_rs = resample_1d(x, y, new_len=new_len)
    if standardize:
        y_rs = _standardize_intensity(y_rs)
    return y_rs


def load_rruff_raman_sheet(
    excel_path: str,
    sheet_name: str,
    new_len: int = 1024,
    standardize: bool = True,
) -> np.ndarray:
    """
    Load one 'RRUFF-style' Raman sheet (from 拉曼汇总.xlsx).
    Those sheets often have headers mixed with metadata; easiest is header=None.
    We read first two columns as (x,y).
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)
    if df.shape[1] < 2:
        raise ValueError(f"RRUFF Raman sheet {sheet_name} has <2 columns.")
    x = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
    y = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < 10:
        raise ValueError(f"RRUFF Raman sheet {sheet_name} has too few numeric points.")
    y_rs = resample_1d(x, y, new_len=new_len)
    if standardize:
        y_rs = _standardize_intensity(y_rs)
    return y_rs


### EXTRA HELPERS ###

def filter_rgb_log_by_humidity(log: RGBLog, tol: float = 15.0) -> RGBLog:
    """Filter out RGB blocks whose humidity is an outlier.

    This is useful because your TXT logs may contain a few stray measurements
    under a very different humidity (e.g., ~27%RH) mixed into an otherwise
    stable experiment (~62%RH or ~76%RH).

    We keep blocks within `tol` (%RH absolute) from the median humidity.
    """
    hum = np.asarray(log.humidity, dtype=np.float32)
    if hum.size == 0:
        return log
    med = float(np.median(hum))
    keep = np.abs(hum - med) <= float(tol)
    if np.all(keep):
        return log
    rgb = log.rgb[keep]
    block_idx = [b for b, k in zip(log.block_idx, keep) if k]
    temperature = [t for t, k in zip(log.temperature, keep) if k]
    humidity = [h for h, k in zip(log.humidity, keep) if k]
    return RGBLog(rgb=rgb, block_idx=block_idx, temperature=temperature, humidity=humidity)


def guess_experiment_tag(median_humidity: float) -> str:
    """Heuristic: map median humidity to experiment tag used in Raman/XRD sheet names.

    Your Raman/XRD Excel sheets are named like "66 ..." and "76 ...".
    The RGB logs show median humidity around ~62 and ~74, so we map:
      - < 70 -> "66"
      - >=70 -> "76"

    You can override this in preprocess by passing --exp_tags.
    """
    return "66" if float(median_humidity) < 70.0 else "76"


_SHEET_PREFIX_RE = re.compile(r"^\s*(\d{2})\s+")


def adapt_sheet_name(sheet_name: str, exp_tag: str) -> str:
    """Replace the leading humidity prefix (e.g., 66/76) with exp_tag.

    Example:
      adapt_sheet_name("66 密陀僧_532", "76") -> "76 密陀僧_532"
    """
    if not sheet_name:
        return sheet_name
    return _SHEET_PREFIX_RE.sub(f"{exp_tag} ", sheet_name.strip())


def load_xy_from_excel(
    excel_path: str,
    sheet_name: str,
    x_col: int = 0,
    y_col: int = 1,
    header: Optional[int] = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load numeric (x,y) arrays from an Excel sheet.

    It tolerates non-numeric rows by coercing to NaN and dropping.
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name, header=header)
    if df.shape[1] <= max(x_col, y_col):
        raise ValueError(f"Sheet {sheet_name} has <={max(x_col,y_col)} columns")
    x = pd.to_numeric(df.iloc[:, x_col], errors="coerce").to_numpy()
    y = pd.to_numeric(df.iloc[:, y_col], errors="coerce").to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok].astype(np.float32)
    y = y[ok].astype(np.float32)
    if x.size < 10:
        raise ValueError(f"Sheet {sheet_name} has too few numeric points")
    return x, y


def baseline_als(y: np.ndarray, lam: float = 1e6, p: float = 0.01, niter: int = 10) -> np.ndarray:
    """Asymmetric least squares (AsLS) baseline correction.

    Classic baseline removal for spectroscopy: solves
      min_z  sum_i w_i (y_i - z_i)^2 + lam * ||D^2 z||^2
    with asymmetric weights controlled by p (0<p<1).

    References:
    - Eilers & Boelens (2005) AsLS baseline correction (widely used; many implementations).
      (We cite a review/derivation in the written report.)
    """
    y = y.astype(np.float64)
    L = y.size
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    w = np.ones(L)
    for _ in range(int(niter)):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * (D @ D.T)
        z = sparse.linalg.spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z.astype(np.float32)


def preprocess_spectrum_for_peaks(
    x: np.ndarray,
    y: np.ndarray,
    do_baseline: bool = True,
    do_smooth: bool = True,
    smooth_window: int = 11,
    smooth_poly: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Basic spectroscopy preprocessing before peak picking."""
    y2 = y.astype(np.float32)
    if do_baseline:
        base = baseline_als(y2, lam=1e6, p=0.01, niter=10)
        y2 = y2 - base
    if do_smooth and y2.size >= smooth_window:
        # window length must be odd
        if smooth_window % 2 == 0:
            smooth_window += 1
        y2 = savgol_filter(y2, window_length=smooth_window, polyorder=smooth_poly).astype(np.float32)
    # normalize to [0,1] for stability
    y2 = y2 - float(np.min(y2))
    denom = float(np.max(y2)) + 1e-8
    y2 = y2 / denom
    return x.astype(np.float32), y2.astype(np.float32)


def extract_peak_features_xy(
    x: np.ndarray,
    y: np.ndarray,
    top_k: int = 32,
    prominence: float = 0.05,
    distance: Optional[int] = None,
) -> np.ndarray:
    """Extract a fixed-length peak feature vector from (x,y).

    Output shape: (2*top_k,)
      - first top_k: normalized peak positions in [0,1]
      - second top_k: normalized peak heights in [0,1]

    If fewer than top_k peaks are found, the rest are padded with 0.
    """
    x2, y2 = preprocess_spectrum_for_peaks(x, y, do_baseline=True, do_smooth=True)
    peaks, props = find_peaks(y2, prominence=prominence, distance=distance)
    if peaks.size == 0:
        return np.zeros((2 * int(top_k),), dtype=np.float32)

    heights = y2[peaks]
    # sort peaks by height descending
    order = np.argsort(-heights)
    peaks = peaks[order]
    heights = heights[order]

    k = min(int(top_k), peaks.size)
    peaks = peaks[:k]
    heights = heights[:k]

    # normalize positions to [0,1] using x-range
    x_min = float(np.min(x2))
    x_max = float(np.max(x2))
    if x_max <= x_min:
        pos = np.zeros_like(heights, dtype=np.float32)
    else:
        pos = ((x2[peaks] - x_min) / (x_max - x_min)).astype(np.float32)

    h = heights.astype(np.float32)
    # pad
    pos_pad = np.zeros((int(top_k),), dtype=np.float32)
    h_pad = np.zeros((int(top_k),), dtype=np.float32)
    pos_pad[:k] = pos
    h_pad[:k] = h
    return np.concatenate([pos_pad, h_pad], axis=0).astype(np.float32)
