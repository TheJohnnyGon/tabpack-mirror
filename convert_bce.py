"""
Convert Nirvana TSV data (separate train / test files) → tabpack format.

Real TSV layout (no header rows):
  Outer cols (real tab-separated): GroupId | subkey | value
  Inner 'value' field (split by literal r'\\t'):
    vcol 0  : Label  → y  (0 or 1)
    vcol 1  : JSON       (skip – DocId / request metadata)
    vcol 2  : Weight     (skip)
    vcol 3…821 : 819 numeric FI_* features → x_num

Output in out_dir/:
  x_num.npy                float32  (N_train + N_test, 819)
  y.npy                    int64    (N_train + N_test,)
  info.json
  splits/default/train.npy   int32   (indices into combined array)
  splits/default/val.npy     int32   (indices into combined array)
  splits/default/test.npy    int32   (indices into combined array)

The train TSV rows come first in the combined array (indices 0…N_train-1),
test TSV rows follow (indices N_train…N_train+N_test-1).
The train portion is further split into train / val using --val fraction.

Usage:
  python convert_bce.py train.tsv test.tsv [--out-dir .] [--seed 42] [--val 0.1]
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ── inner-value column positions (after splitting value by literal r'\\t') ────
VCOL_LABEL      = 0    # binary click label (0 or 1)
VCOL_JSON       = 1    # JSON metadata – skip
VCOL_WEIGHT     = 2    # row weight – skip
VCOL_FEAT_START = 3    # first FI_* feature
N_FEATURES      = 819  # features: vcols 3…821


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert Nirvana TSV (train + test) → tabpack format (BCE task)"
    )
    p.add_argument("train_tsv", help="Path to train TSV file")
    p.add_argument("test_tsv",  help="Path to test TSV file")
    p.add_argument(
        "--out-dir", default=None,
        help="Output directory (default: same dir as train TSV)"
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument(
        "--val", type=float, default=0.1,
        help="Validation fraction taken from the train file (default: 0.1)"
    )
    return p.parse_args()


# After flattening, the column layout in the combined TSV is:
#   flat col 0  : GroupId   (outer col 0)
#   flat col 1  : subkey    (outer col 1)
#   flat col 2  : label     (inner vcol 0)
#   flat col 3  : JSON      (inner vcol 1)  – skip
#   flat col 4  : weight    (inner vcol 2)  – skip
#   flat col 5…823 : 819 FI_* features
_FLAT_LABEL     = 2
_FLAT_FEAT_START = 5
_FLAT_FEAT_COLS = list(range(_FLAT_FEAT_START, _FLAT_FEAT_START + N_FEATURES))  # 5..823
_USECOLS        = [_FLAT_LABEL] + _FLAT_FEAT_COLS                               # 820 cols


def load_tsv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_num float32 (N, 819), y int64 (N,)).

    Fastest path: one C-level bytes.replace turns the literal \\t inner
    separator into a real tab, producing a flat TSV that pd.read_csv can
    parse entirely in C with no Python loops.
    """
    # ── 1. Read & flatten ─────────────────────────────────────────────────────
    # Replace literal two-char sequence b'\\t' (0x5C 0x74) → real tab (0x09).
    # After this the file is a plain tab-separated table with 824 columns.
    data = path.read_bytes().replace(b"\\t", b"\t")

    # ── 2. Single pd.read_csv pass in C ───────────────────────────────────────
    df = pd.read_csv(
        io.BytesIO(data),
        sep="\t",
        header=None,
        usecols=_USECOLS,       # read only label + 819 feat cols; skip the rest
        dtype=np.float32,       # parse everything as float32 directly
        encoding="utf-8",
        on_bad_lines="warn",
    )

    # ── 3. Split label / features ─────────────────────────────────────────────
    y     = df[_FLAT_LABEL].to_numpy(dtype=np.int64)
    x_num = df[_FLAT_FEAT_COLS].to_numpy(dtype=np.float32)

    return x_num, y


def split_train_val(n_train: int, *, val_frac: float, seed: int):
    """
    Split the first n_train rows into train / val index arrays (int32).
    Indices are local to the train block (0…n_train-1).
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_train).astype(np.int32)

    n_val   = max(1, int(n_train * val_frac))
    n_tr    = n_train - n_val

    train_idx = idx[:n_tr]
    val_idx   = idx[n_tr:]
    return train_idx, val_idx


def main() -> None:
    args = parse_args()

    train_path = Path(args.train_tsv)
    test_path  = Path(args.test_tsv)
    out_dir    = Path(args.out_dir) if args.out_dir else train_path.parent

    # ── load ─────────────────────────────────────────────────────────────────
    print(f"Reading train: {train_path} …")
    x_train, y_train = load_tsv(train_path)
    n_train = len(y_train)
    print(f"  Loaded {n_train} rows, {x_train.shape[1]} features")
    print(f"  Label distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")

    print(f"Reading test:  {test_path} …")
    x_test, y_test = load_tsv(test_path)
    n_test = len(y_test)
    print(f"  Loaded {n_test} rows, {x_test.shape[1]} features")
    print(f"  Label distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")

    # ── combine ───────────────────────────────────────────────────────────────
    # train rows occupy indices 0…n_train-1
    # test  rows occupy indices n_train…n_train+n_test-1
    x_num = np.concatenate([x_train, x_test], axis=0)
    y     = np.concatenate([y_train, y_test], axis=0)
    n_total = len(y)
    print(f"\nCombined: {n_total} rows total")

    # ── save arrays ──────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "x_num.npy", x_num)
    np.save(out_dir / "y.npy",     y)
    print(f"  Saved x_num.npy {x_num.shape} float32")
    print(f"  Saved y.npy     {y.shape} int64")

    # ── info.json ────────────────────────────────────────────────────────────
    info = {"task": {"type": "binclass", "score": "accuracy"}}
    (out_dir / "info.json").write_text(json.dumps(info, indent=4))
    print("  Saved info.json")

    # ── splits ───────────────────────────────────────────────────────────────
    # Split train block into train / val; test block becomes test split
    local_train_idx, local_val_idx = split_train_val(
        n_train, val_frac=args.val, seed=args.seed
    )

    # Test indices are the last n_test positions in the combined array
    test_idx  = np.arange(n_train, n_total, dtype=np.int32)

    split_dir = out_dir / "splits" / "default"
    split_dir.mkdir(parents=True, exist_ok=True)
    np.save(split_dir / "train.npy", local_train_idx)
    np.save(split_dir / "val.npy",   local_val_idx)
    np.save(split_dir / "test.npy",  test_idx)
    print(
        f"  Saved splits/default/  "
        f"train={len(local_train_idx)}  val={len(local_val_idx)}  test={len(test_idx)}"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
