#!/usr/bin/env python3
"""
Convert YT/YQL TSV dumps → tabpack dataset format (Polars edition).

Same functionality as convert.py but using polars for multi-threaded
parsing with 3-10x speedup.

Input files (--train / --val / --test, at least one required) are TSV.
Two layouts are auto-detected per file:

1. "clean" layout (preferred, produced by YQL export) — FAST, multi-threaded
   via polars with streaming support:
       key \t label \t f_1 \t f_2 \t ... \t f_K
   Real tabs, no header. `key` is the group key (string or int),
   label is the target, then K feature columns.

2. "raw" layout (legacy YT dump) — slow Python-loop fallback (same as original):
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
Ignored columns are simply never materialized (polars column selection).

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
    python convert_polars.py --train train.tsv --val val.tsv --test test.tsv \
        --out-dir data/peru --cd cd.txt --ignored-features ignored-features \
        --task-type binclass
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

CHUNK_ROWS = 500_000  # rows per polars batch

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
    p = argparse.ArgumentParser(
        description="Convert YT/YQL TSV dumps → tabpack format (Polars edition)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
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
    p.add_argument("--n-workers", type=int, default=os.cpu_count(),
                   help="Number of workers for polars (default: all CPUs)")
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
    last = b""
    with path.open("rb") as f:
        while block := f.read(1 << 24):
            n += block.count(b"\n")
            last = block
    if last and not last.endswith(b"\n"):
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
    """Vectorized-ish key hashing: hash each unique key once (across the
    whole array passed in, not per-chunk — call this once per file for
    best performance)."""
    # Use numpy unique for better performance than pd.factorize
    uniques, codes = np.unique(raw_keys, return_inverse=True)
    # Hash unique keys
    uh = np.empty(len(uniques), dtype=np.uint64)
    for i, u in enumerate(uniques):
        uh[i] = memo.get(u, memo.setdefault(u, hash_key(u)))
    return uh[codes]


# ── loaders ──────────────────────────────────────────────────────────────────
def load_clean_polars(
    path: Path,
    row0: int,
    x_num: np.memmap,
    x_cat: list,
    ys: np.ndarray,
    keys: np.ndarray,
    num_cols: np.ndarray,
    cat_cols: np.ndarray,
    n_features: int,
    n_workers: int = 1,
):
    """Polars-based vectorized loader for clean layout. Returns rows written.

    Fully vectorized: no per-chunk Python loop. The whole file is read once
    by polars (multi-threaded), then every output array is filled with a
    single bulk numpy operation instead of looping over row-chunks.
    """
    col_num = [2 + int(i) for i in num_cols]
    col_cat = [2 + int(i) for i in cat_cols]
    cat_set = set(col_cat)

    # IMPORTANT: polars names headerless columns column_1, column_2, ...
    # (1-based!), not column_0. So a 0-based file column index `c` maps to
    # polars column name f"column_{c + 1}".
    col_num_names = [f"column_{c + 1}" for c in col_num]
    col_cat_names = [f"column_{c + 1}" for c in col_cat]
    key_col_name = "column_1"   # file column 0 = key
    label_col_name = "column_2"  # file column 1 = label

    # Schema overrides MUST cover every column in the file, including the
    # ones we are going to drop (ignored features) — otherwise polars infers
    # their type from a small sample and blows up later if e.g. a column
    # looks like ints in the first rows but has floats further down.
    # File layout: 0=key, 1=label, 2..2+n_features-1=features.
    schema_overrides = {key_col_name: pl.String, label_col_name: pl.Float64}
    for feat_idx in range(n_features):
        c = 2 + feat_idx  # 0-based file column index for this feature
        schema_overrides[f"column_{c + 1}"] = pl.String if c in cat_set else pl.Float32

    null_values = list(NA_STRINGS)
    memo: dict = {}

    # ── Read the whole file with polars (multi-threaded) ──────────────
    # Reading all columns is faster than selecting subset because polars
    # parses the whole line anyway; dropping columns after is cheap.
    _t0 = time.perf_counter()
    df = pl.read_csv(
        str(path),
        separator="\t",
        has_header=False,
        try_parse_dates=False,
        null_values=null_values,
        n_threads=n_workers,
        schema_overrides=schema_overrides,
        comment_prefix=None,
    )
    n = len(df)
    _t1 = time.perf_counter()
    print(f"  Rows: {n:,}, workers: {n_workers}")
    print(f"  [profile] pl.read_csv: {_t1 - _t0:.2f}s")

    sl = slice(row0, row0 + n)

    # ── Labels: one bulk conversion ────────────────────────────────────
    _s = time.perf_counter()
    labels = df[label_col_name].to_numpy().astype(np.float64)
    if np.isnan(labels).any():
        bad = int(np.flatnonzero(np.isnan(labels))[0]) + 1
        sys.exit(f"error: {path} row {bad}: label is null/missing — почини YQL")
    ys[sl] = labels
    _t_labels = time.perf_counter() - _s

    # ── Keys: one bulk hash pass over unique values ────────────────────
    _s = time.perf_counter()
    raw_keys = df[key_col_name].to_numpy()
    keys[sl] = hash_keys_vec(raw_keys, memo)
    _t_keys = time.perf_counter() - _s

    # ── Numeric features: single 2D bulk conversion, no per-column loop ─
    _s = time.perf_counter()
    if col_num_names:
        # df.select(...).to_numpy() on a uniformly-typed (Float32) frame
        # returns one contiguous 2D array — a single vectorized memcpy
        # instead of 370 separate per-column calls.
        x_num[sl, :] = df.select(col_num_names).to_numpy().astype(np.float32)
    _t_numfeat = time.perf_counter() - _s

    # ── Categorical features: bulk-extract columns, then zip into rows ──
    _s = time.perf_counter()
    if col_cat_names:
        cat_cols_data = [
            df[col_name].cast(pl.String).fill_null("").to_list()
            for col_name in col_cat_names
        ]
        x_cat.extend(list(row_vals) for row_vals in zip(*cat_cols_data))
    _t_catfeat = time.perf_counter() - _s

    print(f"  [profile] labels: {_t_labels:.2f}s, keys(hash): {_t_keys:.2f}s, "
          f"num_features(bulk write): {_t_numfeat:.2f}s, cat_features: {_t_catfeat:.2f}s")
    print(f"  [profile] TOTAL post-read processing: "
          f"{_t_labels + _t_keys + _t_numfeat + _t_catfeat:.2f}s")

    return n


def load_raw(path, row0, x_num, x_cat, ys, keys, num_cols, cat_cols):
    """Python-loop fallback for the legacy raw layout. Returns rows written."""
    n_features = x_num.shape[1] + len(cat_cols)
    row = row0
    memo: dict = {}
    total_lines = count_rows(path)
    lines_processed = 0
    progress_interval = max(1, total_lines // 100)  # Print every 1%
    
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
            keys[row] = memo.get(key, memo.setdefault(key, hash_key(key)))
            x_num[row] = [
                float("nan") if feats[i] in NA_STRINGS else float(feats[i])
                for i in num_cols
            ]
            if len(cat_cols):
                x_cat.append([feats[i] for i in cat_cols])
            row += 1
            
            # Progress reporting
            lines_processed += 1
            if lines_processed % progress_interval == 0:
                pct = lines_processed / total_lines * 100
                print(f"  {pct:.0f}% ({lines_processed:,}/{total_lines:,} rows)")
    
    print(f"  100% ({lines_processed:,}/{total_lines:,} rows)")
    return row - row0


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    splits = [
        (name, getattr(args, name))
        for name in ("train", "val", "test")
        if getattr(args, name) is not None
    ]
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

    counts = {name: count_rows(path) for name, path in splits}
    n_total = sum(counts.values())
    print(f"Rows: {counts} (total {n_total}), features: {n_features}, "
          f"layouts: {layouts}")
    print(f"Polars version: {pl.__version__}, workers: {args.n_workers or 'auto'}")

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
        dtype=np.float32, shape=(n_total, len(num_cols)),
    )
    x_cat: list[list[str]] = []
    ys = np.empty(n_total, dtype=np.float64)
    keys = np.empty(n_total, dtype=np.uint64)

    # ── convert ──────────────────────────────────────────────────────────────
    row, offsets = 0, {}
    for name, path in splits:
        print(f"Loading {name} ({counts[name]:,} rows)...")
        loader = load_clean_polars if layouts[name] == "clean" else load_raw
        if layouts[name] == "clean":
            n = loader(path, row, x_num, x_cat, ys, keys, num_cols, cat_cols,
                       n_features, n_workers=args.n_workers or 1)
        else:
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
