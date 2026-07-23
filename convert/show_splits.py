#!/usr/bin/env python3
"""Display contents of .npy files in ~/code/convert/peru/splits/default."""

import numpy as np
import pathlib
import sys


SPLITS_DIR = pathlib.Path("~/code/convert/peru/splits/default").expanduser()


def show_array(path: pathlib.Path) -> None:
    print(f"\n{'='*60}")
    print(f"File: {path.name}")
    print(f"Size on disk: {path.stat().st_size / 1024:.1f} KB")
    arr = np.load(path)
    print(f"Shape: {arr.shape}")
    print(f"Dtype: {arr.dtype}")
    print(f"Min: {arr.min():.6f}")
    print(f"Max: {arr.max():.6f}")
    print(f"Mean: {arr.mean():.6f}")
    print(f"Std: {arr.std():.6f}")
    # Show first few rows
    rows = min(arr.shape[0], 5)
    cols = min(arr.shape[1], 10) if arr.ndim > 1 else 0
    print(f"First {rows} rows (first {cols} cols):")
    if arr.ndim == 1:
        print(arr[:rows])
    else:
        print(arr[:rows, :cols])


def main() -> None:
    if not SPLITS_DIR.is_dir():
        print(f"Error: directory not found: {SPLITS_DIR}", file=sys.stderr)
        sys.exit(1)

    npy_files = sorted(SPLITS_DIR.glob("*.npy"))
    if not npy_files:
        print(f"No .npy files found in {SPLITS_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"Contents of {SPLITS_DIR}")
    print(f"Found {len(npy_files)} file(s): {[f.name for f in npy_files]}")

    for f in npy_files:
        show_array(f)

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()
