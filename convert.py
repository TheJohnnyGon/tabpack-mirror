"""
Convert Nirvana TSV data → tabpack format.

Real TSV layout (no header rows):
  Outer cols (real tab-separated): GroupId | subkey | value
  Inner 'value' field (split by literal r'\t'):
    vcol 0  : Label  → y  (0 or 1)
    vcol 1  : JSON       (skip – DocId / request metadata)
    vcol 2  : Weight     (skip)
    vcol 3…821 : 819 numeric FI_* features → x_num

Output in out_dir/:
  x_num.npy                float32  (N, 819)
  y.npy                    int64    (N,)
  info.json
  splits/default/train.npy   int32
  splits/default/val.npy     int32
  splits/default/test.npy    int32

Usage:
  python convert.py input.tsv [--out-dir .] [--seed 42] [--val 0.15] [--test 0.15]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# ── inner-value column positions (after splitting value by literal r'\t') ────
VCOL_LABEL      = 0    # binary click label (0 or 1)
VCOL_JSON       = 1    # JSON metadata – skip
VCOL_WEIGHT     = 2    # row weight – skip
VCOL_FEAT_START = 3    # first FI_* feature
N_FEATURES      = 819  # features: vcols 3…821


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert Nirvana TSV → tabpack format")
    p.add_argument("tsv", help="Path to input TSV file")
    p.add_argument("--out-dir", default=None,
                   help="Output directory (default: same dir as TSV)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val",  type=float, default=0.15, help="Val fraction")
    p.add_argument("--test", type=float, default=0.15, help="Test fraction")
    return p.parse_args()


def load_tsv(path: Path):
    """Return (x_num float32 (N,819), y int64 (N,))."""
    xs = []
    ys = []

    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue

            # outer split by real tab → [GroupId, subkey, value]
            outer = line.split("\t")
            if len(outer) < 3:
                raise ValueError(
                    f"Line {lineno}: expected ≥3 outer tab-cols, got {len(outer)}"
                )

            # inner split by literal backslash-t
            vcols = outer[2].split(r"\t")

            # ── label ──────────────────────────────────────────────────────
            try:
                label = int(vcols[VCOL_LABEL])
            except (IndexError, ValueError) as e:
                raise ValueError(
                    f"Line {lineno}: cannot parse label at vcol {VCOL_LABEL}: {e!r}"
                )

            # ── features ───────────────────────────────────────────────────
            feat_vcols = vcols[VCOL_FEAT_START: VCOL_FEAT_START + N_FEATURES]
            if len(feat_vcols) < N_FEATURES:
                raise ValueError(
                    f"Line {lineno}: expected {N_FEATURES} feature vcols, "
                    f"got {len(feat_vcols)} (total vcols={len(vcols)})"
                )
            try:
                feats = [float(v) for v in feat_vcols]
            except ValueError as e:
                raise ValueError(f"Line {lineno}: cannot parse feature: {e!r}")

            ys.append(label)
            xs.append(feats)

    x_num = np.array(xs, dtype=np.float32)
    y     = np.array(ys, dtype=np.int64)
    return x_num, y


def make_splits(n, *, val_frac, test_frac, seed):
    """Return (train_idx, val_idx, test_idx) as int32 arrays."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n).astype(np.int32)

    n_test  = max(1, int(n * test_frac))
    n_val   = max(1, int(n * val_frac))
    n_train = n - n_val - n_test

    train_idx = idx[:n_train]
    val_idx   = idx[n_train: n_train + n_val]
    test_idx  = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


def main() -> None:
    args = parse_args()
    tsv_path = Path(args.tsv)
    out_dir  = Path(args.out_dir) if args.out_dir else tsv_path.parent

    print(f"Reading {tsv_path} …")
    x_num, y = load_tsv(tsv_path)
    n = len(y)
    print(f"  Loaded {n} rows, {x_num.shape[1]} features")
    print(f"  Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

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
    train_idx, val_idx, test_idx = make_splits(
        n, val_frac=args.val, test_frac=args.test, seed=args.seed
    )
    split_dir = out_dir / "splits" / "default"
    split_dir.mkdir(parents=True, exist_ok=True)
    np.save(split_dir / "train.npy", train_idx)
    np.save(split_dir / "val.npy",   val_idx)
    np.save(split_dir / "test.npy",  test_idx)
    print(
        f"  Saved splits/default/  "
        f"train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}"
    )

    print("Done.")


if __name__ == "__main__":
    main()
