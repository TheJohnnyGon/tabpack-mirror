import functools
import itertools
import string
import typing
import warnings
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

import lib
import lib.data
import lib.experiment
import lib.util
from lib.types import JSONDict

from . import datasets as devdatasets

# ======================================================================================
# Extended reports
# ======================================================================================
# Extended reports are dictionaries that will be flattened and used as rows in the
# dataframes when summarizing results. As the name suggests, an extended report contains
# the original experiment report, plus extra fields representing future dataframe
# columns expected by the computational functions in this module.
#
# Quick notes on API:
#
# - Use `make_extended_report` to create extended reports.
# - Use the `load_*_extended_reports` functions below to load extended reports
#   from typical experiments.
# - Implement project-specific `load_*_extended_reports` functions for loading extended
#   reports from custom experiments.
type ExtendedReport = JSONDict


@functools.lru_cache(None)
def _load_extended_dataset_info_cached(*args, **kwargs) -> dict[str, Any]:
    return devdatasets.load_extended_info(*args, **kwargs)


@functools.lru_cache(None)
def _load_report_cached(*args, **kwargs):
    return lib.experiment.load_report(*args, **kwargs)


def make_extended_report(
    *,
    report: lib.experiment.Report,
    config: JSONDict,
    name: str,
    seed: str,
    dataset_dir: str | Path,
    dataset_split: None | lib.data.SplitIDLike,
    cache_dataset_info: bool = True,
) -> ExtendedReport:
    assert 'metrics' in report, 'The report is missing the required "metrics" field'

    load_extended_dataset_info_fn = (
        _load_extended_dataset_info_cached
        if cache_dataset_info
        else devdatasets.load_extended_info
    )
    dataset_split = (
        lib.data.DEFAULT_SPLIT_ID
        if dataset_split is None
        else lib.data.make_split_id(dataset_split)
    )

    extended_report = deepcopy(report)
    del report

    extended_report['config'] = config
    # By convention, all new field names start with the capital letter
    # (except for the "config").
    extended_report['Name'] = name
    extended_report['Seed'] = seed
    extended_report['Dataset'] = {
        'name': dataset_dir,
        'path': dataset_dir,
        'split': '/'.join(dataset_split),
        **load_extended_dataset_info_fn(dataset_dir, dataset_split),
    }

    # Add the unified score for `compute_relative_metrics_`.
    for part_metrics in extended_report['metrics'].values():
        part_metrics['unified_score'] = part_metrics[
            'r2' if extended_report['Dataset']['task_type'] == 'regression' else 'score'
        ]

    return extended_report


def load_evaluation_extended_reports(
    exp: str | Path,
    report: None | lib.experiment.Report,
    *,
    n_seeds: None | int = None,
    **kwargs,
) -> list[ExtendedReport]:
    if report is None:
        report = lib.experiment.load_report(exp)
    experiments = report['experiments']
    if n_seeds is not None:
        experiments = experiments[:n_seeds]
    return [
        make_extended_report(
            **x,
            seed=x['config']['seed'],
            dataset_dir=x['config']['data']['path'],
            dataset_split=x['config']['data'].get('split'),
            **kwargs,
        )
        for x in experiments
    ]


def load_ensemble_extended_reports(
    exp: str | Path,
    report: None | lib.experiment.Report,
    *,
    n_ensembles: None | int = None,
    **kwargs,
) -> list[ExtendedReport]:
    if report is None:
        report = lib.experiment.load_report(exp)
    experiments = report['experiments']
    if n_ensembles is not None:
        experiments = experiments[:n_ensembles]
    return [
        make_extended_report(
            **x,
            seed=','.join(map(str, x['report']['seeds'])),
            dataset_dir=x['config']['data']['path'],
            dataset_split=x['config']['data'].get('split'),
            **kwargs,
        )
        for x in experiments
    ]


def load_tuning_extended_reports(
    exp: str | Path, report: None | lib.experiment.Report, **kwargs
) -> list[ExtendedReport]:
    if report is None:
        report = lib.experiment.load_report(exp)
    best_experiment = report['best']
    return [
        make_extended_report(
            **best_experiment,
            seed=best_experiment['config']['seed'],
            dataset_dir=best_experiment['config']['data']['path'],
            dataset_split=best_experiment['config']['data'].get('split'),
            **kwargs,
        )
    ]


type LoadFn = Callable[..., list[ExtendedReport]]


def load_records(
    named_experiments_: dict[str, str | Path | list[str] | list[Path]],
    /,
    load_fn: None | LoadFn | dict[str, LoadFn] = None,
    *,
    require_done: bool = True,
    allow_skipping: bool = False,
    progress_bar: bool = False,
    verbose: bool = False,
    cache: bool = False,
    **kwargs,
) -> list[JSONDict]:
    """Load records for a future dataframe.

    This is a high-level function for loading records of many experiments.

    Args:
        named_experiments_: the experiments to load.
        load_fn: the function(s) to use for loading extended reports.
            If a dictionary, it must be a mapping from a function name
            (will be taken from `report['function']`) to a loader function.
        require_done: if True, load only the experiments that are done.
        allow_skipping: if True, do _not_ raise an error if some of the experiments
            do not exist or are not done
            (the latter is applicable when `require_done=True`).
        progress_bar: show the progress bar while loading results. Useful when
            loading many experiments.
        verbose: if True, print a message on every skipped experiment
            (applicable when `allow_skipping=True`).
        cache: if True, cache common IO results. Use `clear_cache` to clear the cache.
    """
    named_experiments = {
        k: v if isinstance(v, list) else [v] for k, v in named_experiments_.items()
    }
    del named_experiments_

    if load_fn is None:
        load_fn = {
            'bin.evaluate.main': load_evaluation_extended_reports,
            'bin.ensemble.main': load_ensemble_extended_reports,
            'bin.tune.main': load_tuning_extended_reports,
        }

    records = []
    skipped = {}
    with tqdm(
        total=sum(map(len, named_experiments.values())), disable=not progress_bar
    ) as pbar:
        for name, exps in named_experiments.items():
            for exp in exps:
                if lib.experiment.is_experiment(exp) and (
                    lib.experiment.is_done(exp) or not require_done
                ):
                    report = (
                        _load_report_cached(Path(exp).resolve())
                        if cache
                        else lib.experiment.load_report(exp)
                    )
                    exp_load_fn = (
                        load_fn[report['function']]
                        if isinstance(load_fn, dict)
                        else load_fn
                    )
                    records.extend(
                        lib.util.flatten_dict(x)
                        for x in exp_load_fn(
                            exp, report, name=name, cache_dataset_info=cache, **kwargs
                        )
                    )

                elif allow_skipping:
                    if verbose:
                        print(f'Skipping {name=} {exp=}')
                    skipped.setdefault(name, []).append(exp)

                else:
                    raise RuntimeError(
                        f'The experiment is either missing or not done: {name=} {exp=}.'
                        ' Consider passing require_done=False or allow_skipping=True.'
                    )

                pbar.update()

    if skipped:
        logger.warning(f'#Skipped: { {k: len(v) for k, v in skipped.items()} }')

    return records


def clear_cache():
    _load_extended_dataset_info_cached.cache_clear()
    _load_report_cached.cache_clear()


# ======================================================================================
# Dataframes
# ======================================================================================
# NOTE
# Some of the functions below expect dataframes to be created
# from records loaded with `load_records`.


def load_dataframe(
    *experiments: (
        # `experiments` is either *one* dictionary:
        dict[str, str | Path | list[str] | list[Path]]
        # or any number of individual experiments and/or experiment lists.
        | str
        | Path
        | list[str]
        | list[Path]
    ),
    name_fn: None | Callable[[str | Path], str] = None,
    **kwargs,
) -> pd.DataFrame:
    """Create a dataframe from many experiments.

    This is a shortcut for `pd.DataFrame.from_records(load_records(...))`
    that also allows passing experiments as positional arguments,
    in which case the experiment names are generated automatically or
    with the `name_fn` function.

    **Usage**

    To load experiments with automatically generated names, pass any number of
    experiments or experiment lists as positional arguments:

    >>> load_dataframe(
    ...     'exp/mlp/churn/evaluation',                        # Name='A'
    ...     'exp/mlp/california/evaluation',                   # Name='B'
    ...     [f'exp/resnet/{d}/evaluation' for d in datasets],  # Name='C'
    ...     glob('exp/densenet/*/evaluation'),                 # Name='D'
    ... )

    To set experiment names manually:

    >>> load_dataframe(
    ...     {
    ...         'MLP': ['exp/mlp/churn/evaluation', 'exp/mlp/california/evaluation'],
    ...         'ResNet': [f'exp/resnet/{d}/evaluation' for d in datasets],
    ...         'DenseNet': glob('exp/densenet/*/evaluation'),
    ...     }
    ... )

    To generate custom experiment names:

    >>> load_dataframe(
    ...     'exp/mlp/churn/evaluation',
    ...     'exp/mlp/california/evaluation',
    ...     [f'exp/resnet/{d}/evaluation' for d in datasets],
    ...     glob('exp/densenet/*/evaluation'),
    ...     name_fn=lambda x: x.split('/', 3)[1]  # "exp/mlp/..." -> "mlp"
    ... )
    """
    if len(experiments) == 1 and isinstance(experiments[0], dict):
        if name_fn is not None:
            raise ValueError(
                'When experiments are passed as a dictionary,'
                ' name_fn must not be provided'
            )
        named_experiments = experiments[0]

    else:
        if not all(isinstance(x, str | Path | list) for x in experiments):
            raise ValueError('Invalid input. See the documentation')

        experiments = typing.cast(
            tuple[str | Path | list[str] | list[Path], ...], experiments
        )

        if name_fn is None:
            letters = string.ascii_uppercase
            named_experiments = {
                letter * (i // len(letters) + 1): x
                for i, (x, letter) in enumerate(
                    zip(experiments, itertools.cycle(letters))
                )
            }

        else:
            named_experiments = {
                name_fn(x[0] if isinstance(x, list) else x): x for x in experiments
            }
            if len(named_experiments) < len(experiments):
                raise RuntimeError(
                    'The provided name_fn generated the same name for different'
                    ' experiments'
                )

    records = load_records(named_experiments, **kwargs)
    return pd.DataFrame.from_records(records)


def aggregate_all_columns(
    df: pd.DataFrame,
    num_statistics: str | list[str],
    *,
    by: None | str | list[str],
) -> pd.DataFrame:
    """Aggregate all columns of a dataframe.

    The aggregation rules are as follows:

    * The columns in `by` (if provided) are not aggregated.
    * For columns with unclear aggregation semantics (e.g. random seed),
      the values are collected in lists.
    * For "config.*" columns, the first value is taken.
    * For the remaining non-numerical columns, the first value is taken.
    * For the remaining numerical columns, `num_statistics` are used.
    """
    if isinstance(num_statistics, str):
        num_statistics = [num_statistics]

    aggregations = {}
    for column in df.columns:
        if by is not None and column in by:
            continue

        aggregations.setdefault('Count', (column, 'count'))

        # The following condition catches numeric columns that should not be aggregated
        # as numbers. The condition may be incomplete and is maintained on the
        # best-effort basis.
        if column.lower() == 'seed' or column.endswith('.seed'):
            aggregations[column] = (column, list)

        elif not pd.api.types.is_numeric_dtype(df[column].dtype):
            aggregations[column] = (column, 'first')

        else:
            aggregations.update((f'{column}.{x}', (column, x)) for x in num_statistics)

    return (df if by is None else df.groupby(by)).agg(**aggregations)


def drop_incomplete_datasets(df: pd.DataFrame) -> pd.DataFrame:
    """Drop datasets where not all models are available.

    Call this function before computing ranks to avoid misleading rank values.
    """
    name_column = 'Name'
    n_models = (
        df[name_column].nunique()
        if name_column in df.columns
        else len(df.index.unique(level=name_column))
    )
    return df.groupby('Dataset.name').filter(lambda x: len(x) == n_models)


def _compute_ranks_impl_(
    df: pd.DataFrame, mean_column: str, std_column: None | str
) -> pd.DataFrame:
    df = (
        df.sort_values(mean_column, ascending=False)
        if std_column is None
        else df.sort_values([mean_column, std_column], ascending=[False, True])
    )
    ranks = []
    current_mean = None
    current_std = None
    for _, columns in df.iterrows():
        mean = columns[mean_column]
        std = 0.0 if std_column is None else columns[std_column]
        if current_mean is None:
            ranks.append(1)
            current_mean = mean
            current_std = std
        elif current_mean - mean <= current_std:
            ranks.append(ranks[-1])
        else:
            ranks.append(ranks[-1] + 1)
            current_mean = mean
            current_std = std
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', pd.errors.PerformanceWarning)
        df[f'{mean_column}.rank'] = ranks
    return df


def compute_ranks(
    df: pd.DataFrame,
    mean_column: str,
    std_column: None | str,
    *,
    by: None | str | list[str],
    inplace: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Compute ranks based on a column."""
    if by is None:
        if not inplace:
            df = df.copy()
        _compute_ranks_impl_(df, mean_column, **kwargs)
        return df
    else:
        if inplace:
            raise ValueError('inplace=True is not allowed when `by` is not None')
        return df.groupby(by, group_keys=False).apply(
            _compute_ranks_impl_,  # type: ignore
            mean_column=mean_column,
            std_column=std_column,
            **kwargs,
        )


def compute_relative_metrics_(
    df: pd.DataFrame,
    reference_name: str,
    metric_columns: list[str],
    *,
    by: None | str | list[str],
) -> None:
    """Compute relative metrics."""
    for column in metric_columns:
        if column not in df.columns:
            raise ValueError(f'The dataframe does not have the column "{column}"')
        del column
    name_column = 'Name'
    if by is None:
        by = []
    elif isinstance(by, str):
        by = [by]

    # Temporarily modify the index (it will be restored in the end).
    index = df.index.names
    if index == [None]:
        index = None
    if index is not None:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', pd.errors.PerformanceWarning)
            df.reset_index(inplace=True)

    df.set_index([*by, name_column], inplace=True)

    # Example for the following code
    #
    # by:             ['Dataset.name', 'Dataset.split']
    # name_column:    'Name'
    # metric_columns: ['A', 'B', 'C']
    #
    # df[metric_columns]:
    #                                       |  A  |  B  |  C  |
    # | Dataset.name | Dataset.split | Name |     |     |     |
    # | ------------ | ------------- | ---- | --- | --- | --- |
    # | california   | 0             | MLP  | ... | ... | ... |
    # | california   | 0             | TabM | ... | ... | ... |
    # | california   | 0             | FT-T | ... | ... | ... |
    #
    # df_reference_metrics:
    #                                       |  A  |  B  |  C  |
    # | Dataset.name | Dataset.split | Name |     |     |     |
    # | ------------ | ------------- | ---- | --- | --- | --- |
    # | california   | 0             | MLP  | ... | ... | ... |
    #
    # df_reference_metrics.droplevel(name_column):
    #                                |  A  |  B  |  C  |
    # | Dataset.name | Dataset.split |     |     |     |
    # | ------------ | ------------- | --- | --- | --- |
    # | california   | 0             | ... | ... | ... |

    # Keep only the reference rows and only the metric columns.
    df_reference_metrics = df.loc[
        (*[slice(None) for _ in by], reference_name), metric_columns
    ]
    # During the division, the two operands will be matched on all `by` columns,
    # and the right operand with the missing level (because of droplevel)
    # will be broadcasted along the missing level.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', pd.errors.PerformanceWarning)
        df[[f'{x}.relative.{reference_name}' for x in metric_columns]] = 100.0 * (
            df[metric_columns] / df_reference_metrics.droplevel(name_column) - 1.0
        )

    # Restore the original index.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', pd.errors.PerformanceWarning)
        df.reset_index(inplace=True)
    if index is not None:
        df.set_index(index, inplace=True)


# ======================================================================================
# Utilities
# ======================================================================================


def interquantile_mean(
    x: np.ndarray | pd.Series, /, lower: None | float, upper: None | float
) -> float:
    if x.ndim != 1:
        raise ValueError(
            'The input must be a one-dimensional NumPy array or a Pandas series'
        )
    if lower is None and upper is None:
        raise ValueError('At least one of lower and upper must be provided')
    if lower is not None and upper is not None and lower >= upper:
        raise ValueError(
            f'lower must be less than upper, however: {lower=} and {upper=}'
        )

    lower_mask = None if lower is None else x >= np.quantile(x, lower)
    upper_mask = None if upper is None else x < np.quantile(x, upper)
    mask = (
        lower_mask
        if upper_mask is None
        else upper_mask
        if lower_mask is None
        else (lower_mask & upper_mask)
    )
    assert mask is not None
    x_masked = x[mask] if isinstance(x, np.ndarray) else x.loc[mask]
    return float(x_masked.mean())
