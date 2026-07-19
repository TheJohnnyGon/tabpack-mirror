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

Memory / speed strategy (tuned for ~35 GB inputs on a 20–80 GB RAM box):
  * The file is streamed in large byte blocks (default 256 MB) instead of a
    single read_bytes(), so we never hold two full copies of the text at once.
  * Each block is still parsed by ONE pd.read_csv call in C (no per-row Python
    loop) → full-speed parsing preserved.
  * The combined (train+test) array is assembled directly into a memmapped
    x_num.npy; per-block arrays are freed as they are written, so there is no
    np.concatenate spike that would double the ~18 GB float32 array.
  Peak RAM ≈ size of the decoded float32 data (~18 GB) + one block, well under
  the available budget, with no loss of throughput.

Usage:
  python convert_bce.py train.tsv test.tsv [--out-dir .] [--seed 42] [--val 0.1]
                        [--block-mb 256]
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
    p.add_argument(
        "--block-mb", type=int, default=256,
        help="Streaming block size in MiB (default: 256). Larger = faster, "
             "more RAM per block."
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


def _parse_block(raw: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Parse one block of *complete* lines → (x float32 (k,819), y int64 (k,)).

    `raw` must contain only whole lines (end on a newline boundary) so that the
    literal r'\\t' inner separator is never split across a block edge. The
    replace turns b'\\t' (0x5C 0x74) → real tab (0x09), yielding a flat 824-col
    tab table that pd.read_csv parses entirely in C.
    """
    flat = raw.replace(b"\\t", b"\t")
    df = pd.read_csv(
        io.BytesIO(flat),
        sep="\t",
        header=None,
        usecols=_USECOLS,       # read only label + 819 feat cols; skip the rest
        dtype=np.float32,       # parse everything as float32 directly
        encoding="utf-8",
        on_bad_lines="warn",
    )
    y     = df[_FLAT_LABEL].to_numpy(dtype=np.int64)
    x_num = df[_FLAT_FEAT_COLS].to_numpy(dtype=np.float32)
    return x_num, y


def load_tsv(path: Path, *, block_bytes: int) -> tuple[list[np.ndarray], list[np.ndarray], int]:
    """Stream `path` in blocks → (list of x-blocks, list of y-blocks, total rows).

    The file is read in `block_bytes` chunks; a `remainder` buffer keeps the
    trailing partial line so every block handed to _parse_block ends on a
    newline. Peak transient text held in RAM ≈ one block, not the whole file.
    """
    x_blocks: list[np.ndarray] = []
    y_blocks: list[np.ndarray] = []
    n_rows = 0

    remainder = b""
    with path.open("rb") as f:
        while True:
            chunk = f.read(block_bytes)
            if not chunk:
                break
            buf = remainder + chunk
            nl = buf.rfind(b"\n")
            if nl == -1:
                # no newline yet – keep accumulating (extremely long line)
                remainder = buf
                continue
            complete, remainder = buf[: nl + 1], buf[nl + 1:]
            xb, yb = _parse_block(complete)
            if len(yb):
                x_blocks.append(xb)
                y_blocks.append(yb)
                n_rows += len(yb)

    # trailing line without a final newline
    tail = remainder.strip()
    if tail:
        xb, yb = _parse_block(remainder)
        if len(yb):
            x_blocks.append(xb)
            y_blocks.append(yb)
            n_rows += len(yb)

    return x_blocks, y_blocks, n_rows


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


def _drain_into(dst_x: np.ndarray, dst_y: np.ndarray, pos: int,
                x_blocks: list[np.ndarray], y_blocks: list[np.ndarray]) -> int:
    """Copy per-block arrays into the preallocated dst at `pos`, freeing blocks.

    Blocks are popped from the front so the RAM they occupy is released as the
    (disk-backed) memmap fills up — this replaces np.concatenate and avoids the
    double-size spike.
    """
    for i in range(len(x_blocks)):
        xb = x_blocks[i]
        yb = y_blocks[i]
        k = len(yb)
        dst_x[pos: pos + k] = xb
        dst_y[pos: pos + k] = yb
        pos += k
        # release references so the block memory can be reclaimed promptly
        x_blocks[i] = None  # type: ignore[call-overload]
        y_blocks[i] = None  # type: ignore[call-overload]
    return pos


def main() -> None:
    args = parse_args()

    train_path = Path(args.train_tsv)
    test_path  = Path(args.test_tsv)
    out_dir    = Path(args.out_dir) if args.out_dir else train_path.parent
    block_bytes = max(1, args.block_mb) * 1024 * 1024

    # ── load (streamed in blocks) ────────────────────────────────────────────
    print(f"Reading train: {train_path} …  (block {args.block_mb} MiB)")
    x_train_b, y_train_b, n_train = load_tsv(train_path, block_bytes=block_bytes)
    y_train_all = np.concatenate(y_train_b) if y_train_b else np.empty(0, np.int64)
    print(f"  Loaded {n_train} rows, {N_FEATURES} features")
    print(f"  Label distribution: {dict(zip(*np.unique(y_train_all, return_counts=True)))}")
    del y_train_all

    print(f"Reading test:  {test_path} …  (block {args.block_mb} MiB)")
    x_test_b, y_test_b, n_test = load_tsv(test_path, block_bytes=block_bytes)
    y_test_all = np.concatenate(y_test_b) if y_test_b else np.empty(0, np.int64)
    print(f"  Loaded {n_test} rows, {N_FEATURES} features")
    print(f"  Label distribution: {dict(zip(*np.unique(y_test_all, return_counts=True)))}")
    del y_test_all

    # ── combine directly into memmapped output (no concatenate spike) ─────────
    # train rows occupy indices 0…n_train-1
    # test  rows occupy indices n_train…n_train+n_test-1
    n_total = n_train + n_test
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCombined: {n_total} rows total")

    x_num = np.lib.format.open_memmap(
        out_dir / "x_num.npy", mode="w+",
        dtype=np.float32, shape=(n_total, N_FEATURES),
    )
    y = np.lib.format.open_memmap(
        out_dir / "y.npy", mode="w+",
        dtype=np.int64, shape=(n_total,),
    )

    pos = 0
    pos = _drain_into(x_num, y, pos, x_train_b, y_train_b)
    pos = _drain_into(x_num, y, pos, x_test_b, y_test_b)
    assert pos == n_total, f"row count mismatch: filled {pos}, expected {n_total}"

    x_num.flush()
    y.flush()
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
