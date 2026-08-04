#!/usr/bin/env python3
"""
Generate YQL SQL query and column list for TSV export.

Reads cd.txt and ignored-features as txt files,
writes code.txt (SQL query) and columns.txt (column list).
"""
import re
from pathlib import Path

AUX_TYPES = {"groupid", "label", "docid", "weight", "auxiliary",
             "queryid", "subgroupid", "baseline", "timestamp", "position"}


def parse_cd(cd_text: str):
    """Parse cd.txt content, return list of (col_idx, is_cat, name) for features only."""
    features = []
    for line in cd_text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        col_idx = int(parts[0])
        ftype = parts[1].strip().lower()
        name = parts[2].strip() if len(parts) > 2 else f"f_{col_idx}"
        if ftype in AUX_TYPES:
            continue
        if ftype not in ("num", "categ"):
            raise ValueError(f"Unknown column type {ftype!r} at col {col_idx}")
        features.append((col_idx, ftype == "categ", name))
    features.sort()
    return features


def parse_ignored(spec: str, n_features: int):
    """Parse '11:21-23:26' → set of feature indices (0-based among features)."""
    if not spec:
        return set()
    idx = set()
    for token in re.split(r"[:,\s]+", spec.strip()):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo, hi = token.split("-")
            idx.update(range(int(lo), min(int(hi), n_features - 1) + 1))
        elif int(token) < n_features:
            idx.add(int(token))
    return idx


def generate_yql(cd_text: str, ignored_spec: str) -> tuple[str, list[str]]:
    """
    Generate YQL query and column list.
    
    Args:
        cd_text: Content of cd.txt file
        ignored_spec: Content of ignored-features file (e.g., "11:21-23:26")
    
    Returns:
        tuple of (sql_query_string, columns_list)
    """
    # Parse inputs
    features = parse_cd(cd_text)
    n_features = len(features)
    ignored = parse_ignored(ignored_spec, n_features)

    # Filter features
    keep = [(col_idx, is_cat, name, feat_idx)
            for feat_idx, (col_idx, is_cat, name) in enumerate(features)
            if feat_idx not in ignored]

    # Generate SQL query
    sql_lines = [
        "DECLARE $input1 AS String;",
        "DECLARE $output1 AS String;",
        "",
        "INSERT INTO $output1",
        "SELECT",
        "    key,",
        "    CAST(arr[0] AS Int32) AS label,",
    ]

    # Add feature columns
    for i, (col_idx, is_cat, name, feat_idx) in enumerate(keep):
        arr_idx = col_idx - 1  # arr[0]=label, arr[1]=json, arr[2]=weight, arr[3]=f_1 (col_idx=4)
        f_name = f"f_{col_idx - 3}"  # f_1 for col_idx=4, f_2 for col_idx=5, ...
        comma = "," if i < len(keep) - 1 else ""
        sql_lines.append(f"    arr[{arr_idx}] AS {f_name}{comma}")

    sql_lines.extend([
        "FROM",
        "(",
        "    SELECT",
        "        key,",
        "        String::SplitToList(value, '\\t') AS arr",
        "    FROM $input1",
        ");",
    ])

    sql = "\n".join(sql_lines)

    # Generate columns list
    columns = ["key", "label"] + [f"f_{col_idx - 3}" for col_idx, _, _, _ in keep]

    return sql, columns


# Example usage in orchestrator:
# cd_text = cd.read_text()
# ignored_spec = ignored_features.read_text()
# sql, columns = generate_yql(cd_text, ignored_spec)
# code.write_text(sql)
# columns_file.write_text(str(columns))
