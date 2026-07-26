import dataclasses
import enum
import gc
import hashlib
import json
import os
import shutil
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import sklearn.preprocessing
import torch
from loguru import logger
from torch import Tensor

from . import env, util
from .metrics import calculate_metrics as calculate_metrics_
from .types import DataKey, PartKey, PredictionType, TaskType

# ==================================================================================
# Data
# ==================================================================================
_X_NUM_DTYPE = np.float32
_X_BIN_DTYPE = np.float32
_X_CAT_INT_DTYPE = np.int64
# NOTE
# `np.str_` is a special data type that is handled differently from numeric data types.
# In particular, the following dtype check is _not_ correct and always returns `False`:
# `np.array([...], dtype=np.str_).dtype == np.str_`
# The following dtype check returns `True`:
# `isinstance(np.array([...], dtype=np.str_).dtype, np.dtypes.StrDType)`
_X_CAT_STR_DTYPE = np.str_
_Y_REG_DTYPE = np.float32
_Y_CLF_DTYPE = np.int64
_SPLIT_DTYPE = np.int32

# NOTE
# Split is a flat dictionary of indices, e.g. `{"train": ..., "val": ..., "test": ...}`
type Split = dict[PartKey, np.ndarray]

# NOTE
# For a given dataset, splits are stored in the `splits/` directory. This directory
# has a tree-like layout representing different split parts, with some splits
# potentially sharing some of the parts. Consider the following example:
#
# ```
# splits/
#     a/
#         test.npy
#         0/
#             train.npy
#             val.npy
#         1/
#             train.npy
#             val.npy
#     b/
#         test.npy
#         train.npy
#         val.npy
# ```
#
# The above layout represents _three_ splits, where:
#
# - Two splits are stored in the a/ subdirectory and have the same test indices,
#   but different test and validation indices.
# - The third split is stored in the b/ directory and does not share any of its parts
#   with other splits.
#
# Thus, a split is fully identified by a sequence of nested directories that must be
# traversed to collect all of its parts. Formally, in the above example,
# the three splits have the following IDs:
#
# 1. ("a", "0")
# 2. ("a", "1")
# 3. ("b",)
type SplitID = tuple[str, ...]

# NOTE
# Some parts of the API also accept `list[str]` as a split ID to make it possible
# to read a split ID from a text file (e.g. in a TOML or JSON format, where tuples
# cannot be represented) and pass it as-is.
type SplitIDLike = list[str] | SplitID

DEFAULT_SPLIT_ID: SplitID = ('default',)


def make_split_id(split_id: SplitIDLike) -> SplitID:
    assert isinstance(split_id, tuple | list)
    return tuple(split_id)


def load_info(dataset_dir: str | Path) -> dict[str, Any]:
    dataset_dir = _check_dataset_dir(dataset_dir)
    return json.loads(dataset_dir.joinpath('info.json').read_text())


def _check_dataset_dir(dataset_dir: str | Path) -> Path:
    if isinstance(dataset_dir, str):
        dataset_dir = dataset_dir.removeprefix(':')
        dataset_dir = Path(dataset_dir)
    assert dataset_dir.exists(), f'The dataset does not exist: {dataset_dir}'
    return dataset_dir


def _is_npy_path(path: Path) -> bool:
    return path.suffix == '.npy' and not path.is_dir()


def load_split(dataset_dir: str | Path, split_id: SplitIDLike) -> Split:
    dataset_dir = _check_dataset_dir(dataset_dir)
    assert split_id, 'split_id must be non-empty'

    split_id = make_split_id(split_id)
    split = {}
    directory = dataset_dir / 'splits'
    for dirname in split_id:
        directory = directory / dirname
        assert directory.is_dir()
        for path in directory.iterdir():
            if _is_npy_path(path):
                part = path.stem
                assert part not in split, (
                    f'The part "{part}" is presented more than once'
                    f' in the split {split_id}'
                )
                value = np.load(path)
                assert value.dtype == _SPLIT_DTYPE
                split[part] = value
    return split


def apply_split(data: Any, split: Split, *, copy: bool = False) -> Any:
    """Split a NumPy array or a container holding NumPy arrays.

    * If `data` is `np.ndarray`, the function returns `dict[PartKey, np.ndarray]`
      based on the provided `split`.
    * If `data` is a (nested) mapping, then the function traverses `data` and replaces
      all `np.ndarray`s with `dict[PartKey, np.ndarray]` using the provided `split`.
    """
    if isinstance(data, np.ndarray):
        # Fancy indexing always creates a copy in NumPy.
        # When `copy=False` and data is mmap, the slice is still materialized
        # (NumPy has no way to avoid it for non-sequential indices), but at least
        # the full mmap array stays non-resident.
        return {k: data[v].copy() if copy else data[v] for k, v in split.items()}
    elif isinstance(data, Mapping):
        return type(data)(
            (k, apply_split(v, split, copy=copy))
            for k, v in data.items()  # type: ignore
        )
    else:
        raise ValueError(f'Unsupported type of `data`: {type(data)}')


def load_data(
    dataset_dir: str | Path,
    split_id: SplitIDLike,
    *,
    mmap: bool = False,
) -> dict[DataKey, dict[PartKey, np.ndarray]]:
    """Load dataset from disk.

    When ``mmap=True``, uses memory-mapped files so the full dataset never
    becomes resident in RAM — only the split slices are materialized.
    In DDP mode this means all processes share the same OS page cache
    instead of each holding an independent copy.
    """
    dataset_dir = _check_dataset_dir(dataset_dir)
    load_kwargs = {'mmap_mode': 'r'} if mmap else {}
    raw = {
        x.stem: np.load(x, **load_kwargs)
        for x in dataset_dir.iterdir()
        if _is_npy_path(x)
    }
    split = load_split(dataset_dir, split_id)
    data = apply_split(raw, split)

    # Free the mmap base arrays after splitting so the OS can drop pages.
    if mmap:
        del raw
        gc.collect()

    info = load_info(dataset_dir)
    y_expected_dtype = (
        _Y_REG_DTYPE
        if info['task']['type'] == TaskType.REGRESSION.value
        else _Y_CLF_DTYPE
    )
    for key, expected_dtypes in [
        ('x_num', (_X_NUM_DTYPE,)),
        ('x_bin', (_X_BIN_DTYPE,)),
        ('x_cat', (_X_CAT_INT_DTYPE, _X_CAT_STR_DTYPE)),
        ('y', (y_expected_dtype,)),
    ]:
        subdata = data.get(key)
        if subdata is not None:
            assert any(
                all(
                    (
                        isinstance(x.dtype, np.dtypes.StrDType)
                        if expected_dtype is np.str_
                        else x.dtype == expected_dtype
                    )
                    for x in subdata.values()
                )
                for expected_dtype in expected_dtypes
            ), (
                f'Invalid data type of "{key}".'
                f' Expected one of: {expected_dtypes}.'
                f' Actual: {next(iter(subdata.values())).dtype}'
            )

    return data


# ==================================================================================
# Preprocessing
# ==================================================================================
class NumPolicy(enum.Enum):
    STANDARD = 'standard'
    NOISY_QUANTILE = 'noisy-quantile'


def transform_num(
    X_num: dict[PartKey, np.ndarray],
    policy: None | str | NumPolicy,
    seed: None | int,
    *,
    transform_chunk_size: int = 5_000_000,
) -> dict[PartKey, np.ndarray]:
    """Preprocess numerical features with minimal memory overhead.

    Key optimizations vs. the original implementation:
    - Noise is generated in float32 and added in-place.
    - QuantileTransformer fits on a small subsample (10M rows max).
    - ``transform`` processes each part in chunks, writing into a
      pre-allocated float32 buffer (no intermediate float64 arrays).
    - ``nan_to_num`` and ``astype`` use ``copy=False`` to avoid
      unnecessary allocations when the data is already clean.
    - Constant-column mask is computed on a subsample of train.
    """
    if policy is not None:
        policy = NumPolicy(policy)

        if policy == NumPolicy.NOISY_QUANTILE:
            assert seed is not None
            # Generate noise in float32 directly using the new Generator API
            # (RandomState.standard_normal does not accept dtype parameter).
            rng = np.random.default_rng(seed)
            noise = rng.standard_normal(X_num['train'].shape, dtype=np.float32) * 1e-5
            # Write back into the dict so the noisy data flows through both
            # fit and transform below.  Rebinding a local variable would lose
            # the noise for the chunked transform loop.
            X_num['train'] = X_num['train'] + noise
            del noise
            del rng

            # Fit QuantileTransformer on a manageable subsample.
            # Fitting on all rows provides no statistical benefit for
            # large datasets and creates huge internal sort buffers.
            subsample_size = min(X_num['train'].shape[0], 10_000_000)
            normalizer = sklearn.preprocessing.QuantileTransformer(
                n_quantiles=max(min(X_num['train'].shape[0] // 30, 1000), 10),
                output_distribution='normal',
                subsample=subsample_size,
                random_state=seed,
            )
        elif policy == NumPolicy.STANDARD:
            normalizer = sklearn.preprocessing.StandardScaler()
        else:
            raise ValueError(f'Unknown policy={policy}')

        normalizer.fit(X_num['train'])

        # Transform each part in chunks, writing into a pre-allocated
        # float32 buffer to avoid QuantileTransformer returning float64.
        result = {}
        for part, arr in X_num.items():
            n_rows = arr.shape[0]
            n_cols = arr.shape[1]
            buf = np.empty((n_rows, n_cols), dtype=_X_NUM_DTYPE)
            for start in range(0, n_rows, transform_chunk_size):
                end = min(start + transform_chunk_size, n_rows)
                chunk_out = normalizer.transform(arr[start:end])
                chunk_out = chunk_out.astype(_X_NUM_DTYPE, copy=False)
                buf[start:end] = chunk_out
            result[part] = buf
        X_num = result

    # NOTE
    # (This is not a good way to process NaNs)
    # This is a quick hack to stop failing on some datasets because of NaNs.
    # NaNs are replaced with zeros (zero is the mean value for all features after
    # the conventional preprocessing techniques).
    X_num = {k: np.nan_to_num(v, copy=False) for k, v in X_num.items()}

    # Remove columns with one constant value.
    # Compute the mask on a subsample to avoid sorting every column.
    # Use a seeded RNG to ensure deterministic results across runs and DDP ranks.
    train = X_num['train']
    rng = np.random.RandomState(seed if seed is not None else 42)
    subsample_idx = rng.choice(
        train.shape[0], min(train.shape[0], 5_000_000), replace=False
    )
    mask = np.array(
        [len(np.unique(train[subsample_idx, col])) > 1 for col in range(train.shape[1])]
    )
    X_num = {k: v[:, mask] for k, v in X_num.items()}

    X_num = {k: v.astype(_X_NUM_DTYPE, copy=False) for k, v in X_num.items()}
    return X_num


def _extract_bin_from_num(
    X_num: dict[PartKey, np.ndarray],
) -> tuple[None | dict[PartKey, np.ndarray], None | dict[PartKey, np.ndarray]]:
    X_num_train = X_num['train']
    n_cols = X_num_train.shape[1]

    # Check for missing values and unique values per column across ALL parts.
    # Use np.unique per part + np.union1d to merge (C-speed, sorted output).
    # Early-exit when unique count exceeds 2 (column is not binary).
    has_missing_values = np.zeros(n_cols, dtype=bool)
    unique_values: list[np.ndarray] = []
    for col in range(n_cols):
        has_nan_col = False
        merged: np.ndarray | None = None
        for part_arr in X_num.values():
            col_data = part_arr[:, col]
            if not has_nan_col:
                has_nan_col = np.any(np.isnan(col_data))
            pu = np.unique(col_data)
            merged = pu if merged is None else np.union1d(merged, pu)
            if len(merged) > 2:
                break  # not binary, no point continuing
        has_missing_values[col] = has_nan_col
        if merged is None:
            # Empty X_num — should not happen in normal operation, but guard
            # against len(None) → TypeError downstream.
            unique_values.append(np.array([], dtype=X_num_train.dtype))
        else:
            unique_values.append(merged)

    unique_counts = np.array([len(u) for u in unique_values])

    bin_mask = (unique_counts == 2) & ~has_missing_values
    bin_idx = np.nonzero(bin_mask)[0]

    if len(bin_idx) > 0:
        transformer = sklearn.preprocessing.OrdinalEncoder(
            categories=[unique_values[i] for i in bin_idx]
        )
        transformer.fit(X_num['train'][:, bin_idx])
        if len(bin_idx) == n_cols:
            # All the features are binary.
            return (
                {k: transformer.transform(v).astype(bool) for k, v in X_num.items()},
                None,
            )
        else:
            # Some of the features are binary.
            return (
                {
                    k: transformer.transform(v[:, bin_idx]).astype(bool)
                    for k, v in X_num.items()
                },
                {k: v[:, ~bin_mask] for k, v in X_num.items()},
            )
    else:
        # No binary features.
        return None, X_num


class BinPolicy(enum.Enum):
    CONVERT_TO_CAT = 'convert-to-cat'


class CatPolicy(enum.Enum):
    ORDINAL = 'ordinal'
    ONE_HOT = 'one-hot'


def transform_cat(
    X_cat: dict[PartKey, np.ndarray], policy: None | str | CatPolicy
) -> dict[PartKey, np.ndarray]:
    if policy is None:
        return X_cat

    policy = CatPolicy(policy)

    # The first step is always the ordinal encoding,
    # even for the one-hot encoding.
    unknown_value = np.iinfo('int64').max - 3
    encoder = sklearn.preprocessing.OrdinalEncoder(
        handle_unknown='use_encoded_value',  # type: ignore
        unknown_value=unknown_value,  # type: ignore
        dtype='int64',  # type: ignore
    ).fit(X_cat['train'])
    X_cat = {k: encoder.transform(v) for k, v in X_cat.items()}
    max_values = X_cat['train'].max(axis=0)
    for part in ['val', 'test']:
        part = cast(PartKey, part)
        for column_idx in range(X_cat[part].shape[1]):
            X_cat[part][X_cat[part][:, column_idx] == unknown_value, column_idx] = (
                max_values[column_idx] + 1
            )

    if policy == CatPolicy.ORDINAL:
        return X_cat
    elif policy == CatPolicy.ONE_HOT:
        encoder = sklearn.preprocessing.OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False,
            dtype=np.float32,  # type: ignore
        )
        encoder.fit(X_cat['train'])
        return {k: cast(np.ndarray, encoder.transform(v)) for k, v in X_cat.items()}
    else:
        raise ValueError(f'Unknown policy={policy}')


@dataclass(frozen=True, kw_only=True)
class RegressionLabelStats:
    mean: float
    std: float


def standardize_labels(
    y: dict[PartKey, np.ndarray],
) -> tuple[dict[PartKey, np.ndarray], RegressionLabelStats]:
    assert y['train'].dtype == np.dtype('float32')
    mean = float(y['train'].mean())
    std = float(y['train'].std())
    return {k: (v - mean) / std for k, v in y.items()}, RegressionLabelStats(
        mean=mean, std=std
    )


# ==================================================================================
# Task
# ==================================================================================
class Score(enum.Enum):
    ACCURACY = 'accuracy'
    CROSS_ENTROPY = 'cross-entropy'
    MAE = 'mae'
    R2 = 'r2'
    RMSE = 'rmse'
    ROC_AUC = 'roc-auc'


_SCORE_HIGHER_IS_BETTER = {
    Score.ACCURACY: True,
    Score.CROSS_ENTROPY: False,
    Score.MAE: False,
    Score.R2: True,
    Score.RMSE: False,
    Score.ROC_AUC: True,
}


@dataclass(frozen=True)
class Task:
    labels: dict[PartKey, np.ndarray]
    type_: TaskType
    score: Score

    @classmethod
    def from_dir(cls, dataset_dir: str | Path, split_id: SplitIDLike) -> 'Task':
        dataset_dir = _check_dataset_dir(dataset_dir)
        y = np.load(dataset_dir / 'y.npy')
        split = load_split(dataset_dir, split_id)
        task_info = load_info(dataset_dir)['task']
        return Task(
            apply_split(y, split, copy=True),
            TaskType(task_info['type']),
            Score(task_info['score']),
        )

    @classmethod
    def from_data(
        cls,
        data: dict[DataKey, dict[PartKey, np.ndarray]],
        dataset_dir: str | Path,
    ) -> 'Task':
        """Build a Task from already-loaded data (avoids loading y.npy a second time).

        The ``y`` arrays are extracted from ``data`` and copied so the Task owns
        an independent reference (preventing accidental mutation).
        """
        y = data['y']
        task_info = load_info(dataset_dir)['task']
        # Copy the labels so the Task owns an independent reference.
        labels = {k: v.copy() for k, v in y.items()}
        return Task(
            labels,
            TaskType(task_info['type']),
            Score(task_info['score']),
        )

    def __post_init__(self):
        assert isinstance(self.type_, TaskType)
        assert isinstance(self.score, Score)
        if self.is_regression:
            assert all(value.dtype == _Y_REG_DTYPE for value in self.labels.values()), (
                f'Regression labels must have the {_Y_REG_DTYPE} data type'
            )

    @property
    def is_regression(self) -> bool:
        return self.type_ == TaskType.REGRESSION

    @property
    def is_binclass(self) -> bool:
        return self.type_ == TaskType.BINCLASS

    @property
    def is_multiclass(self) -> bool:
        return self.type_ == TaskType.MULTICLASS

    @property
    def is_classification(self) -> bool:
        return self.is_binclass or self.is_multiclass

    def compute_n_classes(self) -> int:
        assert self.is_binclass or self.is_classification
        return len(np.unique(np.concatenate(list(self.labels.values()))))

    def try_compute_n_classes(self) -> None | int:
        return None if self.is_regression else self.compute_n_classes()

    def calculate_metrics(
        self,
        predictions: dict[PartKey, np.ndarray],
        prediction_type: str | PredictionType,
    ) -> dict[PartKey, Any]:
        metrics = {
            part: calculate_metrics_(
                y_true=self.labels[part],
                y_pred=predictions[part],
                task_type=self.type_,
                prediction_type=prediction_type,
            )
            for part in predictions
        }
        for part_metrics in metrics.values():
            part_metrics['score'] = (
                1.0 if _SCORE_HIGHER_IS_BETTER[self.score] else -1.0
            ) * part_metrics[self.score.value]
        return metrics  # type: ignore


# ==================================================================================
# Dataset
# ==================================================================================
@dataclass
class Dataset[T: np.ndarray | Tensor]:
    """Dataset = Data + Task + simple methods for convenience.

    The task is stored separately to ensure that the original labels never change.
    """

    data: dict[DataKey, dict[PartKey, T]]
    task: Task

    @classmethod
    def from_dir(cls, path: str | Path, split_id: SplitIDLike) -> 'Dataset[np.ndarray]':
        data = load_data(path, split_id, mmap=True)
        task = Task.from_data(data, path)
        return Dataset(data, task)

    def _is_numpy(self) -> bool:
        return isinstance(self.data['y']['train'], np.ndarray)

    def to_torch(self, device: None | str | torch.device) -> 'Dataset[Tensor]':
        return Dataset(
            {
                key: {
                    part: torch.as_tensor(value, device=device)
                    for part, value in self.data[key].items()
                }
                for key in self.data
            },
            self.task,
        )

    @property
    def n_num_features(self) -> int:
        return self.data['x_num']['train'].shape[1] if 'x_num' in self.data else 0

    @property
    def n_bin_features(self) -> int:
        return self.data['x_bin']['train'].shape[1] if 'x_bin' in self.data else 0

    @property
    def n_cat_features(self) -> int:
        return self.data['x_cat']['train'].shape[1] if 'x_cat' in self.data else 0

    @property
    def n_features(self) -> int:
        return self.n_num_features + self.n_bin_features + self.n_cat_features

    def size(self, part: None | PartKey) -> int:
        return (
            sum(map(len, self.data['y'].values()))
            if part is None
            else len(self.data['y'][part])
        )

    def parts(self) -> Iterable[PartKey]:
        return self.data['y'].keys()

    def compute_cat_cardinalities(self) -> list[int]:
        x_cat = self.data.get('x_cat')
        if x_cat is None:
            return []
        unique = np.unique if self._is_numpy() else torch.unique
        return (
            []
            if x_cat is None
            else [len(unique(column)) for column in x_cat['train'].T]
        )

    def convert_bin_features_to_cat_(self: 'Dataset[np.ndarray]') -> None:
        assert self._is_numpy()

        x_bin = self.data.pop('x_bin', None)
        if x_bin is None:
            return

        x_cat = self.data.get('x_cat')
        if x_cat is None:
            x_bin_as_cat = {
                k: np.where(np.isnan(v), 2.0, v).astype(_X_CAT_INT_DTYPE)
                for k, v in x_bin.items()
            }
        else:
            dtype = next(iter(x_cat.values())).dtype
            x_bin_as_cat = {k: v.astype(dtype) for k, v in x_bin.items()}

        if x_cat is None:
            self.data['x_cat'] = x_bin_as_cat
        else:
            assert x_cat.keys() == x_bin_as_cat.keys()
            for part in x_bin_as_cat:
                x_cat[part] = np.column_stack([x_cat[part], x_bin_as_cat[part]])

    def standardize_labels_(self: 'Dataset[np.ndarray]') -> 'RegressionLabelStats':
        assert self._is_numpy()
        assert self.task.is_regression
        self.data['y'], regression_label_stats = standardize_labels(self.data['y'])
        return regression_label_stats

    def try_standardize_labels_(
        self: 'Dataset[np.ndarray]',
    ) -> 'None | RegressionLabelStats':
        assert self._is_numpy()
        return self.standardize_labels_() if self.task.is_regression else None


def _cache_key(
    path: Path,
    split_id: SplitIDLike,
    extract_bin_from_num: bool,
    num_policy: None | str | NumPolicy,
    bin_policy: None | str | BinPolicy,
    cat_policy: None | str | CatPolicy,
    task_score: None | str | Score,
    seed: int,
) -> Path:
    """Compute the cache directory path for a given build_dataset configuration."""
    args = {
        'split_id': split_id,
        'extract_bin_from_num': extract_bin_from_num,
        'num_policy': num_policy,
        'bin_policy': bin_policy,
        'cat_policy': cat_policy,
        'task_score': task_score,
        'seed': seed,
    }
    h = hashlib.md5(str(args).encode('utf-8')).hexdigest()
    return env.get_cache_dir() / f'build_dataset__{path.name}__{h}'


def _cache_is_complete(cache_dir: Path) -> bool:
    """Check that the cache directory has the _COMPLETE marker file.

    This ensures we never read a partially-written cache (e.g. from a
    crashed previous run).
    """
    return (cache_dir / '_COMPLETE').exists()


def _save_dataset_cache(dataset: Dataset[np.ndarray], cache_dir: Path) -> None:
    """Save a dataset to cache as individual .npy files + metadata json.

    Writes to a temporary directory first, then atomically renames to the
    final path and creates a _COMPLETE marker. This prevents other processes
    from reading a partially-written cache.
    """
    tmp_dir = cache_dir.with_suffix('.tmp')
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    # Save each array as a separate .npy file.
    for data_key, parts in dataset.data.items():
        part_dir = tmp_dir / 'data' / data_key
        part_dir.mkdir(parents=True, exist_ok=True)
        for part_key, arr in parts.items():
            np.save(part_dir / f'{part_key}.npy', arr)

    # Save task info.
    task_dir = tmp_dir / 'task'
    task_dir.mkdir(parents=True, exist_ok=True)
    for part_key, arr in dataset.task.labels.items():
        np.save(task_dir / f'{part_key}.npy', arr)

    # Save metadata.
    meta = {
        'type_': dataset.task.type_.value,
        'score': dataset.task.score.value,
    }
    (tmp_dir / 'meta.json').write_text(json.dumps(meta))

    # Atomic rename + completion marker.
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    os.rename(tmp_dir, cache_dir)
    (cache_dir / '_COMPLETE').touch()


def _load_dataset_cache(cache_dir: Path) -> Dataset[np.ndarray]:
    """Load a dataset from cache directory.

    Uses mmap_mode='r' so all DDP processes share the same OS page cache
    instead of each holding an independent copy.
    """
    data = {}
    data_root = cache_dir / 'data'
    for key_dir in data_root.iterdir():
        if key_dir.is_dir():
            data[key_dir.name] = {}
            for part_file in key_dir.iterdir():
                if part_file.suffix == '.npy':
                    part_key = part_file.stem
                    data[key_dir.name][part_key] = np.load(
                        part_file, mmap_mode='r'
                    )

    task_dir = cache_dir / 'task'
    labels = {}
    for part_file in task_dir.iterdir():
        if part_file.suffix == '.npy':
            part_key = part_file.stem
            labels[part_key] = np.load(part_file, mmap_mode='r')

    meta = json.loads((cache_dir / 'meta.json').read_text())
    task = Task(labels, TaskType(meta['type_']), Score(meta['score']))

    return Dataset(data, task)


def build_dataset(
    path: str | Path,
    split_id: SplitIDLike = DEFAULT_SPLIT_ID,
    *,
    extract_bin_from_num: bool = False,
    num_policy: None | str | NumPolicy = None,
    bin_policy: None | str | BinPolicy = None,
    cat_policy: None | str | CatPolicy = None,
    task_score: None | str | Score = None,
    seed: int = 0,
    cache: bool = False,
) -> Dataset[np.ndarray]:
    path = Path(path).resolve()
    cache_dir = None

    if cache:
        cache_dir = _cache_key(
            path, split_id, extract_bin_from_num, num_policy,
            bin_policy, cat_policy, task_score, seed,
        )
        distributed = util.is_distributed()
        local_rank = util.get_local_rank()

        if _cache_is_complete(cache_dir):
            # Cache is complete — all ranks read it directly.
            # No barrier needed (barrier is a collective op; if only some
            # ranks enter it, the others hang forever).
            logger.info(f'Using cached dataset: {cache_dir.name}')
            return _load_dataset_cache(cache_dir)
        elif distributed and local_rank != 0:
            # Rank 0 will build the cache; others poll for completion.
            # We cannot use dist.barrier() here because rank 0 does not
            # enter this branch — a barrier requires ALL ranks, so it
            # would deadlock.  Polling on the _COMPLETE marker is safe
            # since _save_dataset_cache creates it after os.rename.
            import time
            while not _cache_is_complete(cache_dir):
                time.sleep(0.1)
            logger.info(f'Using cached dataset (built by rank 0): {cache_dir.name}')
            return _load_dataset_cache(cache_dir)

    dataset = Dataset.from_dir(path, split_id)
    if task_score is not None:
        dataset = dataclasses.replace(
            dataset, task=dataclasses.replace(dataset.task, score=Score(task_score))
        )

    if 'x_num' in dataset.data and extract_bin_from_num:
        extracted_x_bin, remaining_x_num = _extract_bin_from_num(dataset.data['x_num'])
        if extracted_x_bin is not None:
            if remaining_x_num is None:
                del dataset.data['x_num']
            else:
                dataset.data['x_num'] = remaining_x_num
            x_bin = dataset.data.pop('x_bin', None)
            if x_bin is None:
                dataset.data['x_bin'] = extracted_x_bin
            else:
                assert extracted_x_bin.keys() == x_bin.keys()
                dataset.data['x_bin'] = {
                    k: np.concatenate([extracted_x_bin[k], x_bin[k]], axis=-1)
                    for k in extracted_x_bin.keys()
                }

    # The presence of "x_num" may change after the binary feature extraction,
    # so it must be checked again.
    if 'x_num' in dataset.data:
        dataset.data['x_num'] = transform_num(dataset.data['x_num'], num_policy, seed)

    if 'x_bin' in dataset.data:
        if bin_policy is not None:
            bin_policy = BinPolicy(bin_policy)
            if bin_policy == BinPolicy.CONVERT_TO_CAT:
                dataset.convert_bin_features_to_cat_()
            else:
                raise ValueError(f'Unknown {bin_policy=}')

    if 'x_cat' in dataset.data:
        dataset.data['x_cat'] = transform_cat(dataset.data['x_cat'], cat_policy)

    # Save cache (only rank 0 in DDP mode, or when not distributed).
    if cache_dir is not None:
        distributed = util.is_distributed()
        local_rank = util.get_local_rank()
        if not distributed or local_rank == 0:
            _save_dataset_cache(dataset, cache_dir)
    return dataset
