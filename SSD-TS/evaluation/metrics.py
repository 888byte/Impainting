"""Evaluation metrics."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def summarize_delta_e(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float32)
    return {
        'mean': float(np.mean(arr)) if arr.size else float('nan'),
        'std': float(np.std(arr)) if arr.size else float('nan'),
        'median': float(np.median(arr)) if arr.size else float('nan'),
    }
