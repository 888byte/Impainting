"""Evaluation protocol helpers."""
from __future__ import annotations

import numpy as np


def fps_lab_palette(n: int, seed: int = 0, cand_n: int = 20000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cand = np.empty((cand_n, 3), dtype=np.float32)
    cand[:, 0] = rng.uniform(0, 100, size=cand_n)
    cand[:, 1] = rng.uniform(-110, 110, size=cand_n)
    cand[:, 2] = rng.uniform(-110, 110, size=cand_n)
    sel = np.empty((n, 3), dtype=np.float32)
    sel[0] = cand[rng.integers(0, cand_n)]
    d2 = np.sum((cand - sel[0]) ** 2, axis=1)
    for i in range(1, n):
        idx = int(np.argmax(d2))
        sel[i] = cand[idx]
        d2 = np.minimum(d2, np.sum((cand - sel[i]) ** 2, axis=1))
    return sel
