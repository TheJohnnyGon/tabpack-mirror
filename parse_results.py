#!/usr/bin/env python3
"""
Parse evaluation/report.json files from tabpack-ag experiments.

Extracts the greedy online-ensemble test score for each seed from:
  exp/{variant}/0/{dataset}/evaluation/report.json
  → experiments[i].report.online_ensembles.greedy.report.metrics.test.score

Usage:
    uv run parse_results.py
    uv run parse_results.py --exp-dir exp/tabpack/0 --datasets california churn
    uv run parse_results.py --variants tabpack tabpack-cosine
"""

import argparse
import json
import statistics
from pathlib import Path
from typing import Optional

# ── configuration ─────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
EXP_DIR = BASE_DIR / 'exp' / 'tabpack' / '0'

# All known experiment variants (in display order)
ALL_VARIANTS = ['tabpack', 'tabpack-cosine']

VARIANT_LABELS = {
    'tabpack': 'tabpack (MuonAdamW)',
    'tabpack-cosine': 'tabpack-cosine (cosine embeddings + MuonAdamW)',
}


def discover_datasets(exp_dir: Path) -> list[str]:
    """Find all datasets that have a completed evaluation/report.json."""
    datasets = []
    if not exp_dir.exists():
        return datasets
    for dataset_dir in sorted(exp_dir.iterdir()):
        report_path = dataset_dir / 'evaluation' / 'report.json'
        if report_path.exists():
            datasets.append(dataset_dir.name)
    return datasets


def extract_seed_scores(report_path: Path, ensemble: str = 'greedy') -> list[Optional[float]]:
    """
    Extract per-seed test scores from evaluation/report.json.

    Each item in report['experiments'] corresponds to one seed run.
    The test score lives at:
      experiment['report']['online_ensembles'][ensemble]['report']['metrics']['test']['score']

    For metrics where higher is better (accuracy, roc-auc, r2),
    the score is already positive.
    For metrics where lower is better (rmse, mae, cross-entropy),
    the score is stored as negative → abs() to get the actual value.
    """
    try:
        report = json.loads(report_path.read_text(encoding='utf-8'))
    except Exception as e:
        print(f'[ERROR] Failed to read {report_path}: {e}')
        return []

    scores = []
    for exp in report.get('experiments', []):
        try:
            metrics = (
                exp['report']['online_ensembles'][ensemble]['report']['metrics']
            )
            score = metrics['test']['score']
            scores.append(abs(float(score)))
        except KeyError as e:
            print(f'[WARN] Missing key {e} in seed={exp.get("config", {}).get("seed", "?")} of {report_path}')
            scores.append(None)

    return scores


def get_metric_name(report_path: Path, ensemble: str = 'greedy') -> str:
    """
    Identify the primary metric — the one whose absolute value matches |score|.
    Falls back to listing all non-score keys if nothing matches.
    """
    try:
        report = json.loads(report_path.read_text(encoding='utf-8'))
        metrics = (
            report['experiments'][0]['report']['online_ensembles'][ensemble]['report']['metrics']['test']
        )
        score = abs(float(metrics['score']))
        for key, val in metrics.items():
            if key == 'score':
                continue
            try:
                if abs(abs(float(val)) - score) < 1e-9:
                    return key
            except (TypeError, ValueError):
                continue
        # Fallback: list scalar numeric keys only
        scalar_keys = [k for k, v in metrics.items() if k != 'score' and isinstance(v, (int, float))]
        return ', '.join(scalar_keys) if scalar_keys else '?'
    except Exception:
        return '?'


def collect_results(exp_dir: Path, datasets: list[str], ensemble: str) -> tuple[
    dict[str, list[Optional[float]]], dict[str, str]
]:
    """Collect scores and metric names for all datasets in exp_dir."""
    results: dict[str, list[Optional[float]]] = {}
    metric_names: dict[str, str] = {}

    for dataset in datasets:
        report_path = exp_dir / dataset / 'evaluation' / 'report.json'
        if not report_path.exists():
            print(f'[MISSING] {report_path}')
            results[dataset] = []
            metric_names[dataset] = '?'
            continue

        scores = extract_seed_scores(report_path, ensemble)
        results[dataset] = scores
        metric_names[dataset] = get_metric_name(report_path, ensemble)

    return results, metric_names


def print_section(label: str, datasets: list[str],
                  results: dict[str, list[Optional[float]]],
                  metric_names: dict[str, str]) -> None:
    """Pretty-print a per-seed table + mean±std table for one variant."""
    HW = 20   # dataset column width
    CW = 12   # per-seed column width
    MW = 22   # mean±std column width

    max_seeds = max((len(s) for s in results.values()), default=0)

    print(f'\n{"=" * 60}')
    print(f'  {label}')
    print(f'{"=" * 60}')

    # Per-seed table
    seed_header = ''.join(f'{"seed"+str(i):>{CW}}' for i in range(max_seeds))
    print(f'{"Dataset":<{HW}}{seed_header}')
    print('-' * (HW + CW * max_seeds))

    for dataset in datasets:
        scores = results[dataset]
        row = ''.join(
            f'{(f"{s:.6f}" if s is not None else "N/A"):>{CW}}'
            for s in scores
        )
        row += ''.join(f'{"N/A":>{CW}}' for _ in range(max_seeds - len(scores)))
        metric = metric_names.get(dataset, '?')
        print(f'{dataset:<{HW}}{row}  [{metric}]')

    # Mean ± std table
    print()
    print(f'{"Dataset":<{HW}}{"Metric":<{HW}}{"Mean ± Std":>{MW}}{"N seeds":>{8}}')
    print('-' * (HW + HW + MW + 8))

    for dataset in datasets:
        scores = results[dataset]
        valid = [s for s in scores if s is not None]
        if valid:
            mean = statistics.mean(valid)
            std = statistics.stdev(valid) if len(valid) > 1 else 0.0
            cell = f'{mean:.4f} ± {std:.4f}'
        else:
            cell = 'N/A'
        metric = metric_names.get(dataset, '?')
        print(f'{dataset:<{HW}}{metric:<{HW}}{cell:>{MW}}{len(valid):>{8}}')


# ── argument parsing ───────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description='Parse tabpack-ag experiment results')
parser.add_argument('--exp-dir', type=Path, default=None,
                    help='Path to a single exp/variant/0 directory (overrides --variants)')
parser.add_argument('--variants', nargs='+', default=ALL_VARIANTS,
                    help=f'Experiment variants to parse (default: {ALL_VARIANTS})')
parser.add_argument('--datasets', nargs='+', default=None,
                    help='Datasets to include (default: all discovered)')
parser.add_argument('--ensemble', default='greedy',
                    help='Online ensemble name (default: greedy)')
parser.add_argument('--tsv', type=Path, default=None,
                    help='Path to write TSV output (default: exp/../results.tsv)')
args = parser.parse_args()

ensemble: str = args.ensemble

# ── build list of (variant_label, exp_dir) pairs ──────────────────────────────

if args.exp_dir is not None:
    # Legacy single-directory mode
    run_list = [(args.exp_dir.parent.parent.name, args.exp_dir)]
else:
    run_list = [
        (variant, BASE_DIR / 'exp' / variant / '0')
        for variant in args.variants
    ]

# ── iterate over variants ──────────────────────────────────────────────────────

all_csv_rows: list[str] = []

for variant, exp_dir in run_list:
    if not exp_dir.exists():
        print(f'[SKIP] {exp_dir} does not exist')
        continue

    if args.datasets:
        datasets = args.datasets
    else:
        datasets = discover_datasets(exp_dir)

    if not datasets:
        print(f'[SKIP] No completed evaluation experiments found in {exp_dir}')
        continue

    print(f'\n[{variant}] Found datasets: {datasets}')

    results, metric_names = collect_results(exp_dir, datasets, ensemble)

    label = VARIANT_LABELS.get(variant, variant)
    print_section(label, datasets, results, metric_names)

    # Accumulate CSV rows
    for dataset in datasets:
        scores = results[dataset]
        metric = metric_names.get(dataset, '?')
        for seed_idx, score in enumerate(scores):
            val = f'{score:.6f}' if score is not None else 'N/A'
            all_csv_rows.append(f'{variant}\t{dataset}\t{seed_idx}\t{val}\t{metric}')

# ── write TSV ──────────────────────────────────────────────────────────────────

tsv_path = args.tsv or (BASE_DIR / 'exp' / 'results.tsv')
with open(tsv_path, 'w', encoding='utf-8') as f:
    f.write('variant\tdataset\tseed\ttest_score\tmetric\n')
    for row in all_csv_rows:
        f.write(row + '\n')

print(f'\nTSV saved → {tsv_path}')
