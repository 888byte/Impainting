#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Inspect NPZ contents from the command line."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def describe_array(name: str, arr: np.ndarray) -> None:
    print(f"\n[Array] {name}")
    print(f"  shape: {arr.shape}")
    print(f"  dtype: {arr.dtype}")
    print(f"  ndim:  {arr.ndim}")
    print(f"  size:  {arr.size}")
    if arr.size > 1000:
        preview = arr[:5, :5] if arr.ndim >= 2 else arr[:10]
        print("  preview:")
        print(preview)
    else:
        print("  values:")
        print(arr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('npz_path', type=str, nargs='?', default='pigment_lut33.npz')
    args = ap.parse_args()

    path = Path(args.npz_path)
    if not path.exists():
        raise FileNotFoundError(f'NPZ not found: {path}')

    data = np.load(path, allow_pickle=True)
    try:
        print('=' * 60)
        print(f'NPZ: {path}')
        print('arrays:')
        for idx, name in enumerate(data.files, start=1):
            print(f'  {idx}. {name}')
        print('=' * 60)
        for name in data.files:
            describe_array(name, data[name])
    finally:
        data.close()


if __name__ == '__main__':
    main()
