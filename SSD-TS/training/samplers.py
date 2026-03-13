"""Group-aware samplers for many-RGB-to-one-spectrum training."""
from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Optional

import torch
from torch.utils.data import WeightedRandomSampler


def build_parent_sampler(rows: Optional[List[dict]], key: str = 'spectral_parent_id'):
    if not rows:
        return None
    groups = [row.get(key, '') for row in rows]
    counts = Counter(groups)
    weights = torch.tensor([1.0 / max(counts[group], 1) for group in groups], dtype=torch.double)
    return WeightedRandomSampler(weights=weights, num_samples=len(rows), replacement=True)
