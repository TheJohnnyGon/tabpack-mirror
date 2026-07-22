#!/usr/bin/env python3
"""
Convert YT/YQL TSV dumps → tabpack dataset format.

Input files (--train / --val / --test, at least one required) are TSV.
Two layouts are auto-detected per file:

1. "clean" layout (preferred, produced by YQL export) — FAST, vectorized
   via pandas.read_csv (C engine, chunked, constant memory):
       key \t label \t f_1 \t f_2 \t ... \t f_K
   Real tabs, no header. `key` is the group key (string or int),
   label is the target, then K feature columns.

2. "raw" layout (legacy YT dump) — slow Python-loop fallback:
       key=... \t subkey=... \t value=LABEL\t{json}\tWEIGHT\tf_1\t...\tf_K
   Outer separator is a real tab; the `value` field is split by the
   literal two-character sequence backslash-t. vcol0 = label,
   vcol1 = json metadata (skipped), vcol2 = weight (skipped),
   vcols 3.. = features.

Column descriptor (--cd, CatBoost-style cd.txt):
       <col_idx> \t <type> [\t <name>]
   Rows with type GroupId/Label/DocId/Weight/Auxiliary are auxiliary;
   Num/Categ rows are features. Feature index = order among feature rows
   (0-based), matching CatBoost's ignored-features numbering.
   Without --cd all features are Num.

--ignored-features accepts a string like "11:21-23:26:811-1000" or a path
to a file with that string. Indices are FEATURE indices (0-based, aux
columns excluded), ranges are inclusive and clipped to the actual range.
Ignored columns are simply never materialized (pandas `usecols`).

Output (--out-dir):
    x_num.npy                  float32 (N, n_num)
    x_cat.npy                  <U*     (N, n_cat)   — only if Categ features exist
    y.npy                      int64 (binclass/multiclass) | float32 (regression)
    key.npy                    uint64  (N,)  — blake2b-8byte hash of the group key
    info.json                  {"task": {"type": ..., "score": ...}}
    splits/default/train.npy   int32  (contiguous ranges, original row order
    splits/default/val.npy      is preserved → groups stay contiguous,
    splits/default/test.npy     ready for future pairlogit/group tasks)

Usage:
    python convert.py --train train.tsv --val val.tsv --test test.tsv \
        --out-dir data/peru --cd cd.txt --ignored-features ignored-features \
        --task-type binclass
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

CHUNK_ROWS = 200_000  # rows per pandas chunk (~650 MB of float32 at 819 feats)

# raw layout: value-field columns (split by literal backslash-t)
RAW_VCOL_LABEL = 0
RAW_VCOL_FEAT_START = 3

TASK_SCORES = {
    "binclass": "accuracy",
    "multiclass": "accuracy",
    "regression": "rmse",
}

NA_STRINGS = {"", "nan", "null", "none", "\\n", "NaN", "NULL"}


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert YT/YQL TSV dumps → tabpack format",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--train", type=Path, help="Train split TSV")
    p.add_argument("--val", type=Path, help="Validation split TSV")
    p.add_argument("--test", type=Path, help="Test split TSV")
    p.add_argument("--out-dir", type=Path, required=True, help="Output dataset dir")
    p.add_argument("--cd", type=Path, default=None,
                   help="Column descriptor (cd.txt); without it all features are Num")
    p.add_argument("--ignored-features", default=None,
                   help='Spec string "11:21-23:26" or path to a file with it '
                        "(0-based FEATURE indices, ranges inclusive, clipped)")
    p.add_argument("--task-type", choices=sorted(TASK_SCORES), default="binclass")
    args = p.parse_args()
    if not (args.train or args.val or args.test):
        p.error("at least one of --train / --val / --test is required")
    return args


# ── column descriptor ────────────────────────────────────────────────────────
AUX_TYPES = {"groupid", "label", "docid", "weight", "auxiliary",
             "queryid", "subgroupid", "baseline", "timestamp", "position"}


def load_cd(path: Path | None):
    """Return is_cat: bool array per FEATURE index, or None."""
    if path is None:
        return None
    feats = []  # (col_idx, is_cat)
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        col_idx, ftype = int(parts[0]), parts[1].strip().lower()
        if ftype in AUX_TYPES:
            continue
        if ftype not in ("num", "categ"):
            raise ValueError(f"cd: unknown column type {parts[1]!r} (col {col_idx})")
        feats.append((col_idx, ftype == "categ"))
    feats.sort()
    return np.array([c for _, c in feats], dtype=bool)


def parse_ignored(spec: str | None, n_features: int) -> np.ndarray:
    """Parse '11:21-23:26' → sorted unique feature indices, clipped to n_features.

    `spec` is either the string itself or a path to a text file with it
    (separators: ':', ',', whitespace/newlines — all equivalent).
    """
    if not spec:
        return np.empty(0, dtype=np.int64)
    path = Path(spec)
    if path.is_file():
        spec = path.read_text()
    idx: set[int] = set()
    for token in re.split(r"[:,\s]+", spec.strip()):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo, hi = token.split("-")
            idx.update(range(int(lo), min(int(hi), n_features - 1) + 1))
        elif int(token) < n_features:
            idx.add(int(token))
    return np.array(sorted(i for i in idx if 0 <= i < n_features), dtype=np.int64)


# ── helpers ──────────────────────────────────────────────────────────────────
def hash_key(key: str) -> int:
    """Deterministic uint64 hash of the group key."""
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "little")


def count_rows(path: Path) -> int:
    """Fast newline count (binary, 16 MB blocks)."""
    n = 0
    last = b"\n"
    with path.open("rb") as f:
        while block := f.read(1 << 24):
            n += block.count(b"\n")
            last = block
    if not last.endswith(b"\n"):
        n += 1
    return n


def sniff(path: Path):
    """Return (layout: 'clean'|'raw', n_features) from the first line."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("key=") and "\tsubkey=" in line:
                value = line.split("\t", 2)[2]
                if value.startswith("value="):
                    value = value[len("value="):]
                return "raw", len(value.split(r"\t")) - RAW_VCOL_FEAT_START
            return "clean", len(line.split("\t")) - 2
    raise ValueError(f"{path}: empty file")


def hash_keys_vec(raw_keys: np.ndarray, memo: dict) -> np.ndarray:
    """Vectorized-ish key hashing: hash each unique key once per chunk."""
    codes, uniques = pd.factorize(raw_keys)
    uh = np.array([memo.get(u) or memo.setdefault(u, hash_key(u)) for u in uniques],
                  dtype=np.uint64)
    return uh[codes]


# ── loaders (write rows [row0, row0+n) of the output arrays) ─────────────────
def load_clean(path, row0, x_num, x_cat, ys, keys, num_cols, cat_cols):
    """Vectorized pandas path for the clean layout. Returns rows written."""
    col_num = [2 + int(i) for i in num_cols]
    col_cat = [2 + int(i) for i in cat_cols]
    dtypes = {0: str, 1: np.float64}
    dtypes.update({c: np.float32 for c in col_num})
    dtypes.update({c: str for c in col_cat})
    reader = pd.read_csv(
        path, sep="\t", header=None, engine="c", chunksize=CHUNK_ROWS,
        usecols=[0, 1] + col_num + col_cat, dtype=dtypes,
        na_values=list(NA_STRINGS), keep_default_na=True)
    row = row0
    memo: dict = {}
    # Count total chunks upfront for progress display
    total_rows = count_rows(path)
    total_chunks = (total_rows + CHUNK_ROWS - 1) // CHUNK_ROWS
    print(f"  Chunks: {total_chunks} (~{CHUNK_ROWS:,} rows/chunk)")
    for chunk_idx, chunk in enumerate(reader, 1):
        n = len(chunk)
        sl = slice(row, row + n)
        labels = chunk[1].to_numpy(np.float64)
        if np.isnan(labels).any():
            bad = row - row0 + int(np.flatnonzero(np.isnan(labels))[0]) + 1
            sys.exit(f"error: {path} row {bad}: label is null/missing — почини YQL")
        ys[sl] = labels
        keys[sl] = hash_keys_vec(chunk[0].to_numpy(), memo)
        x_num[sl] = chunk[col_num].to_numpy(np.float32)
        if col_cat:
            x_cat.extend(chunk[col_cat].astype(str).to_numpy().tolist())
        row += n
        print(f"  chunk {chunk_idx}/{total_chunks} ({row - row0:,} rows)")
    return row - row0


def load_raw(path, row0, x_num, x_cat, ys, keys, num_cols, cat_cols):
    """Python-loop fallback for the legacy raw layout. Returns rows written."""
    n_features = x_num.shape[1] + len(cat_cols)  # not used for validation here
    row = row0
    memo: dict = {}
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            key, _, value = line.split("\t", 2)
            key = key[len("key="):]
            if value.startswith("value="):
                value = value[len("value="):]
            vcols = value.split(r"\t")
            label = vcols[RAW_VCOL_LABEL]
            feats = vcols[RAW_VCOL_FEAT_START:]
            if label.lower() in NA_STRINGS:
                sys.exit(f"error: {path}:{lineno}: label is null/missing — почини YQL")
            ys[row] = float(label)
            keys[row] = memo.get(key) or memo.setdefault(key, hash_key(key))
            x_num[row] = [float("nan") if feats[i] in NA_STRINGS else float(feats[i])
                          for i in num_cols]
            if len(cat_cols):
                x_cat.append([feats[i] for i in cat_cols])
            row += 1
    return row - row0


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    splits = [(name, getattr(args, name)) for name in ("train", "val", "test")
              if getattr(args, name) is not None]
    for name, path in splits:
        if not path.is_file():
            sys.exit(f"error: --{name} file not found: {path}")

    # ── sniff layouts, count rows ────────────────────────────────────────────
    layouts, n_feats_seen = {}, set()
    for name, path in splits:
        layouts[name], nf = sniff(path)
        n_feats_seen.add(nf)
    if len(n_feats_seen) != 1:
        sys.exit(f"error: inconsistent feature counts across files: {n_feats_seen}")
    n_features = n_feats_seen.pop()
    if any(l == "clean" for l in layouts.values()) and pd is None:
        sys.exit("error: pandas is required for TSV input: pip install pandas")

    counts = {name: count_rows(path) for name, path in splits}
    n_total = sum(counts.values())
    print(f"Rows: {counts} (total {n_total}), features: {n_features}, "
          f"layouts: {layouts}")

    # ── cd / ignored ─────────────────────────────────────────────────────────
    is_cat = load_cd(args.cd)
    if is_cat is None:
        is_cat = np.zeros(n_features, dtype=bool)
    if len(is_cat) != n_features:
        sys.exit(f"error: cd describes {len(is_cat)} features, data has {n_features}")

    ignored = parse_ignored(args.ignored_features, n_features)
    keep = np.setdiff1d(np.arange(n_features), ignored)
    num_cols = keep[~is_cat[keep]]
    cat_cols = keep[is_cat[keep]]
    print(f"Ignored: {len(ignored)}, kept: {len(keep)} "
          f"({len(num_cols)} num, {len(cat_cols)} cat)")

    # ── allocate outputs ─────────────────────────────────────────────────────
    args.out_dir.mkdir(parents=True, exist_ok=True)
    x_num = np.lib.format.open_memmap(
        args.out_dir / "x_num.npy", mode="w+",
        dtype=np.float32, shape=(n_total, len(num_cols)))
    x_cat: list[list[str]] = []
    ys = np.empty(n_total, dtype=np.float64)
    keys = np.empty(n_total, dtype=np.uint64)

    # ── convert ──────────────────────────────────────────────────────────────
    row, offsets = 0, {}
    for name, path in splits:
        print(f"Loading {name} ({counts[name]:,} rows)...")
        loader = load_clean if layouts[name] == "clean" else load_raw
        n = loader(path, row, x_num, x_cat, ys, keys, num_cols, cat_cols)
        if n != counts[name]:
            sys.exit(f"error: {path}: parsed {n} rows, counted {counts[name]}")
        offsets[name] = (row, row + n)
        row += n
        print(f"  {name}: rows [{offsets[name][0]}, {offsets[name][1]})")
    x_num.flush()

    # ── y / key / x_cat / info / splits ──────────────────────────────────────
    y_dtype = np.float32 if args.task_type == "regression" else np.int64
    y = ys.astype(y_dtype)
    np.save(args.out_dir / "y.npy", y)
    # int64 view (bit-reinterpret): torch.as_tensor does not support uint64,
    # and lib.data.load_data picks up every *.npy file in the dataset dir.
    np.save(args.out_dir / "key.npy", keys.view(np.int64))
    if len(cat_cols):
        np.save(args.out_dir / "x_cat.npy", np.array(x_cat, dtype=np.str_))

    info = {"task": {"type": args.task_type, "score": TASK_SCORES[args.task_type]}}
    (args.out_dir / "info.json").write_text(json.dumps(info, indent=4) + "\n")

    split_dir = args.out_dir / "splits" / "default"
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, (a, b) in offsets.items():
        np.save(split_dir / f"{name}.npy", np.arange(a, b, dtype=np.int32))

    print(f"Saved to {args.out_dir}:")
    print(f"  x_num.npy {x_num.shape} float32")
    if len(cat_cols):
        print(f"  x_cat.npy ({n_total}, {len(cat_cols)}) str")
    print(f"  y.npy {y.shape} {y.dtype}, key.npy int64, info.json, "
          f"splits/default/{{{', '.join(n for n, _ in splits)}}}.npy")
    if args.task_type != "regression":
        uniq, cnt = np.unique(y, return_counts=True)
        print(f"  label distribution: {dict(zip(uniq.tolist(), cnt.tolist()))}")
    print("Done.")


if __name__ == "__main__":
    main()
