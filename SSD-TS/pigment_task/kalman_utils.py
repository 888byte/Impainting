"""Simple Kalman filter + RTS smoother for 3D Lab sequences.

We use a random-walk state model:
  x_t = x_{t-1} + w_t,   w_t ~ N(0, Q)
  y_t = x_t + v_t,       v_t ~ N(0, R_t)

All operations are done per-dimension (L,a,b) assuming independence.
This is intentionally lightweight (no external deps) and meant as an
optional post-processing step.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def rts_smooth_random_walk(
    y: np.ndarray,
    r_var: np.ndarray,
    q_var: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """RTS smoother for random-walk model.

    Args:
        y: (L, D) observations (D=3 for Lab)
        r_var: (L, D) measurement variance for each time step and dim
        q_var: (D,) or (1,D) process noise variance per step

    Returns:
        x_smooth: (L, D)
        p_smooth: (L, D) posterior variance
    """
    y = np.asarray(y, dtype=np.float64)
    r_var = np.asarray(r_var, dtype=np.float64)
    q_var = np.asarray(q_var, dtype=np.float64).reshape(-1)
    if y.ndim != 2:
        raise ValueError("y must be (L,D)")
    L, D = y.shape
    if r_var.shape != (L, D):
        raise ValueError("r_var must be (L,D)")
    if q_var.size == 1:
        q_var = np.full((D,), float(q_var.item()), dtype=np.float64)
    if q_var.shape != (D,):
        raise ValueError("q_var must be (D,) or scalar")

    # Forward filter
    x_f = np.zeros((L, D), dtype=np.float64)
    p_f = np.zeros((L, D), dtype=np.float64)

    # init: use y0
    x_f[0] = y[0]
    p_f[0] = np.maximum(r_var[0], 1e-9)

    for t in range(1, L):
        # predict
        x_pred = x_f[t - 1]
        p_pred = p_f[t - 1] + q_var

        # update
        s = p_pred + r_var[t]
        k = p_pred / np.maximum(s, 1e-12)
        innov = y[t] - x_pred
        x_f[t] = x_pred + k * innov
        p_f[t] = (1.0 - k) * p_pred

    # RTS backward
    x_s = np.zeros((L, D), dtype=np.float64)
    p_s = np.zeros((L, D), dtype=np.float64)

    x_s[-1] = x_f[-1]
    p_s[-1] = p_f[-1]

    for t in range(L - 2, -1, -1):
        p_pred = p_f[t] + q_var
        g = p_f[t] / np.maximum(p_pred, 1e-12)  # smoother gain
        x_s[t] = x_f[t] + g * (x_s[t + 1] - x_f[t])
        p_s[t] = p_f[t] + (g ** 2) * (p_s[t + 1] - p_pred)

    return x_s.astype(np.float32), np.maximum(p_s, 1e-9).astype(np.float32)
