"""Runtime view objects bound to NPZ arrays and sidecar sample index files."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class SampleIndexRow:
    sample_id: str
    npz_row: int
    source_log: str
    exp_id: str
    exp_tag: str
    side: str
    patch_id: str
    time_index: str
    sequence_parent_id: str
    spectral_parent_id: str
    augmentation_parent_id: str
    is_augmented: str
    split_group_id: str
    has_raman: str
    has_xrd: str


@dataclass
class NPZView:
    npz_path: str
    index_path: str
    rows: List[Dict[str, str]]

    @classmethod
    def from_csv(cls, npz_path: str, index_path: str) -> 'NPZView':
        with open(index_path, 'r', encoding='utf-8', newline='') as f:
            rows = list(csv.DictReader(f))
        return cls(npz_path=npz_path, index_path=index_path, rows=rows)
