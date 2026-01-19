
"""
Color utility functions for pigment fading task.

- sRGB <-> CIE Lab (D65, 2°)
- simple Lab normalization helpers
- DeltaE2000 (optional metric)

This module is self-contained (no skimage / colour-science dependency) so it can
run in the SSD-TS environment directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

# Reference white (D65, 2°)
_XN: float = 0.95047
_YN: float = 1.0
_ZN: float = 1.08883


def _srgb_to_linear(u: np.ndarray) -> np.ndarray:
    u = u / 255.0
    return np.where(u <= 0.04045, u / 12.92, ((u + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(u: np.ndarray) -> np.ndarray:
    u = np.where(u <= 0.0031308, 12.92 * u, 1.055 * np.power(np.clip(u, 0.0, None), 1 / 2.4) - 0.055)
    return np.clip(u, 0.0, 1.0)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """
    Convert sRGB (0..255) to CIE Lab (D65, 2°).

    Args:
        rgb: array (..., 3), dtype float/int in 0..255

    Returns:
        lab: array (..., 3), float64, L in [0,100]
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    r_lin = _srgb_to_linear(rgb[..., 0])
    g_lin = _srgb_to_linear(rgb[..., 1])
    b_lin = _srgb_to_linear(rgb[..., 2])

    # sRGB -> XYZ (D65)
    X = 0.4124564 * r_lin + 0.3575761 * g_lin + 0.1804375 * b_lin
    Y = 0.2126729 * r_lin + 0.7151522 * g_lin + 0.0721750 * b_lin
    Z = 0.0193339 * r_lin + 0.1191920 * g_lin + 0.9503041 * b_lin

    x = X / _XN
    y = Y / _YN
    z = Z / _ZN

    delta = 6 / 29

    def f(t: np.ndarray) -> np.ndarray:
        return np.where(t > delta**3, np.cbrt(t), t / (3 * delta**2) + 4 / 29)

    fx, fy, fz = f(x), f(y), f(z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """
    Convert CIE Lab (D65, 2°) to sRGB (0..255).
    Returns uint8.
    """
    lab = np.asarray(lab, dtype=np.float64)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200

    delta = 6 / 29

    def finv(t: np.ndarray) -> np.ndarray:
        return np.where(t > delta, t**3, 3 * delta**2 * (t - 4 / 29))

    x = finv(fx)
    y = finv(fy)
    z = finv(fz)

    X = x * _XN
    Y = y * _YN
    Z = z * _ZN

    # XYZ -> linear RGB
    r_lin = 3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z
    g_lin = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z
    b_lin = 0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z

    rgb_lin = np.stack([r_lin, g_lin, b_lin], axis=-1)
    rgb = _linear_to_srgb(rgb_lin)
    return np.round(rgb * 255).astype(np.uint8)


@dataclass(frozen=True)
class LabNorm:
    """
    Simple fixed scaling for Lab:
    - L* in [0, 100] -> divide by 100
    - a*, b* typically in [-128, 128] -> divide by 128

    This keeps values roughly in [-1, 1] which stabilizes diffusion training.
    """
    L_scale: float = 100.0
    ab_scale: float = 128.0

    def normalize(self, lab: np.ndarray) -> np.ndarray:
        lab = np.asarray(lab, dtype=np.float32).copy()
        lab[..., 0] = lab[..., 0] / self.L_scale
        lab[..., 1] = lab[..., 1] / self.ab_scale
        lab[..., 2] = lab[..., 2] / self.ab_scale
        return lab

    def denormalize(self, lab_n: np.ndarray) -> np.ndarray:
        lab_n = np.asarray(lab_n, dtype=np.float32).copy()
        lab_n[..., 0] = lab_n[..., 0] * self.L_scale
        lab_n[..., 1] = lab_n[..., 1] * self.ab_scale
        lab_n[..., 2] = lab_n[..., 2] * self.ab_scale
        return lab_n


def delta_e2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """
    Compute CIEDE2000 color difference.

    Args:
        lab1, lab2: (..., 3) arrays

    Returns:
        deltaE: (...) array
    """
    # Implementation based on Sharma et al. (2005), standard formula.
    lab1 = np.asarray(lab1, dtype=np.float64)
    lab2 = np.asarray(lab2, dtype=np.float64)

    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    # Weighting factors
    kL = 1.0
    kC = 1.0
    kH = 1.0

    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    C_bar = (C1 + C2) / 2.0

    G = 0.5 * (1 - np.sqrt((C_bar**7) / (C_bar**7 + 25**7)))
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2

    C1p = np.sqrt(a1p**2 + b1**2)
    C2p = np.sqrt(a2p**2 + b2**2)
    C_bar_p = (C1p + C2p) / 2.0

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = h2p - h1p
    dhp = np.where((C1p * C2p) == 0, 0.0, dhp)
    dhp = np.where(dhp > 180, dhp - 360, dhp)
    dhp = np.where(dhp < -180, dhp + 360, dhp)

    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2.0))

    L_bar_p = (L1 + L2) / 2.0

    # Average hue
    h_bar_p = (h1p + h2p) / 2.0
    h_bar_p = np.where(np.abs(h1p - h2p) > 180, h_bar_p + 180, h_bar_p)
    h_bar_p = np.where((C1p * C2p) == 0, h1p + h2p, h_bar_p)
    h_bar_p = h_bar_p % 360.0

    T = (
        1
        - 0.17 * np.cos(np.radians(h_bar_p - 30))
        + 0.24 * np.cos(np.radians(2 * h_bar_p))
        + 0.32 * np.cos(np.radians(3 * h_bar_p + 6))
        - 0.20 * np.cos(np.radians(4 * h_bar_p - 63))
    )

    dtheta = 30 * np.exp(-((h_bar_p - 275) / 25) ** 2)
    Rc = 2 * np.sqrt((C_bar_p**7) / (C_bar_p**7 + 25**7))
    Sl = 1 + (0.015 * (L_bar_p - 50) ** 2) / np.sqrt(20 + (L_bar_p - 50) ** 2)
    Sc = 1 + 0.045 * C_bar_p
    Sh = 1 + 0.015 * C_bar_p * T
    Rt = -np.sin(np.radians(2 * dtheta)) * Rc

    dE = np.sqrt(
        (dLp / (kL * Sl)) ** 2
        + (dCp / (kC * Sc)) ** 2
        + (dHp / (kH * Sh)) ** 2
        + Rt * (dCp / (kC * Sc)) * (dHp / (kH * Sh))
    )
    return dE
