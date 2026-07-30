import dataclasses
import enum
import hashlib
import json
import os
import pickle
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import gc
import numpy as np
import sklearn.preprocessing
import torch
from loguru import logger
from torch import Tensor

from . import env
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

# Global variables to store feature indices after preprocessing
# These are populated by build_dataset() and can be saved to model.pt
FEATURE_INDICES_NUM: np.ndarray | None = None
FEATURE_INDICES_CAT: np.ndarray | None = None
FEATURE_INDICES_BIN: np.ndarray | None = None

# Global variables to store fitted transformers after preprocessing
# These are populated by transform_num() and transform_cat() and can be saved to model.pt
TRANSFORMER_NUM: sklearn.preprocessing.StandardScaler | sklearn.preprocessing.QuantileTransformer | None = None
TRANSFORMER_CAT_ORDINAL: sklearn.preprocessing.OrdinalEncoder | None = None
TRANSFORMER_CAT_ONEHOT: sklearn.preprocessing.OneHotEncoder | None = None


class DataPreprocessor:
    """
    Препроцессор данных с паттерном fit/transform.
    
    Обучает трансформеры на train данных и применяет их ко всем частям.
    Сохраняет индексы признаков и обученные трансформеры для последующего использования.
    """
    
    def __init__(self, config: dict):
        """
        Инициализация препроцессора.
        
        Args:
            config: Конфигурация с параметрами:
                - seed: случайное зерно для воспроизводимости
                - extract_bin_from_num: извлекать ли бинарные признаки из числовых
                - skip_bin_encoder: пропускать ли OrdinalEncoder для бинарных
                - num_policy: политика трансформации числовых признаков
                - num_memory_efficient: использовать ли memory-efficient трансформацию
                - bin_policy: политика обработки бинарных признаков
                - cat_policy: политика обработки категориальных признаков
        """
        self.config = config
        self.seed = config.get('seed', 0)
        
        # Индексы признаков
        self.feature_indices_num: np.ndarray | None = None
        self.feature_indices_bin: np.ndarray | None = None
        self.feature_indices_cat: np.ndarray | None = None
        
        # Трансформеры
        self.transformer_num: sklearn.preprocessing.StandardScaler | sklearn.preprocessing.QuantileTransformer | None = None
        self.transformer_bin_ordinal: sklearn.preprocessing.OrdinalEncoder | None = None
        self.transformer_cat_ordinal: sklearn.preprocessing.OrdinalEncoder | None = None
        self.transformer_cat_onehot: sklearn.preprocessing.OneHotEncoder | None = None
        
        # Флаг обучения
        self._fitted = False
        
        # Вспомогательные данные
        self._bin_mask: np.ndarray | None = None
        self._unique_values_bin: list | None = None
        self._constant_mask: np.ndarray | None = None  # Маска константных столбцов
    
    def fit(self, dataset: 'Dataset[np.ndarray]') -> 'DataPreprocessor':
        """
        Обучает трансформеры на train данных.
        
        - Бинарные индексы: вычисляются на ВСЕХ данных (train+val+test)
        - QuantileTransformer/StandardScaler: обучается на train
        - OrdinalEncoder для бинарных: обучается на train (если не skip_bin_encoder)
        - OrdinalEncoder для категориальных: обучается на train
        - OneHotEncoder: обучается на train
        
        Args:
            dataset: Датасет с train/val/test частями
            
        Returns:
            self для chaining
        """
        # 1. Извлечение бинарных признаков (индексы на всех данных)
        if 'x_num' in dataset.data and self.config.get('extract_bin_from_num'):
            x_num_all = np.concatenate(list(dataset.data['x_num'].values()))
            has_missing = np.any(np.isnan(x_num_all), 0)
            unique_values = [np.unique(x) for x in x_num_all.T]
            unique_counts = np.array([len(x) for x in unique_values])
            bin_mask = (unique_counts == 2) & ~has_missing
            self.feature_indices_bin = np.nonzero(bin_mask)[0]
            self.feature_indices_num = np.nonzero(~bin_mask)[0]
            self._bin_mask = bin_mask
            
            # Проверяем, все ли бинарные столбцы содержат только 0/1
            skip_encoder = self.config.get('skip_bin_encoder', False)
            is_01 = all(
                set(unique_values[i]) == {0.0, 1.0} for i in self.feature_indices_bin
            )
            
            # Если не skip_encoder или не все столбцы 0/1, создаём OrdinalEncoder
            if not (skip_encoder and is_01) and len(self.feature_indices_bin) > 0:
                self._unique_values_bin = [unique_values[i] for i in self.feature_indices_bin]
                self.transformer_bin_ordinal = sklearn.preprocessing.OrdinalEncoder(
                    categories=self._unique_values_bin
                )
                self.transformer_bin_ordinal.fit(dataset.data['x_num']['train'][:, self.feature_indices_bin])
        
        # 2. Обучение QuantileTransformer/StandardScaler (на train, ПОСЛЕ извлечения бинарных)
        if 'x_num' in dataset.data and self.config.get('num_policy'):
            num_policy = NumPolicy(self.config['num_policy'])
            # Если извлекали бинарные признаки, используем только оставшиеся числовые
            if self.feature_indices_num is not None:
                X_num_train = dataset.data['x_num']['train'][:, self.feature_indices_num]
            else:
                X_num_train = dataset.data['x_num']['train']
            
            if num_policy == NumPolicy.STANDARD:
                self.transformer_num = sklearn.preprocessing.StandardScaler()
                self.transformer_num.fit(X_num_train)
            elif num_policy == NumPolicy.NOISY_QUANTILE:
                # Для noisy-quantile используем columnwise трансформацию если memory_efficient
                if self.config.get('num_memory_efficient', False):
                    # Сохраняем параметры для columnwise трансформации
                    self._num_transformer_params = []
                    n_rows, n_cols = X_num_train.shape
                    for col_in in range(n_cols):
                        train_col = X_num_train[:, col_in]
                        fit_data = train_col.copy()
                        noise = np.random.RandomState(self.seed + col_in).normal(
                            0.0, 1e-5, fit_data.shape
                        ).astype(fit_data.dtype)
                        fit_data += noise
                        normalizer = sklearn.preprocessing.QuantileTransformer(
                            n_quantiles=max(min(n_rows // 30, 1000), 10),
                            output_distribution='normal',
                            random_state=self.seed,
                        )
                        normalizer.fit(fit_data.reshape(-1, 1))
                        self._num_transformer_params.append(normalizer)
                else:
                    self.transformer_num = sklearn.preprocessing.QuantileTransformer(
                        n_quantiles=max(min(X_num_train.shape[0] // 30, 1000), 10),
                        output_distribution='normal',
                        subsample=1_000_000_000,
                        random_state=self.seed,
                    )
                    # Добавляем шум для noisy-quantile
                    X_num_train_noisy = X_num_train + np.random.RandomState(self.seed).normal(
                        0.0, 1e-5, X_num_train.shape
                    ).astype(X_num_train.dtype)
                    self.transformer_num.fit(X_num_train_noisy)
                    
                    # Вычисляем маску константных столбцов ПОСЛЕ трансформации ОРИГИНАЛЬНЫХ данных
                    # (как в оригинальном transform_num)
                    X_num_train_transformed = self.transformer_num.transform(X_num_train)
                    X_num_train_transformed = np.nan_to_num(X_num_train_transformed, copy=False)
                    self._constant_mask = np.ptp(X_num_train_transformed, axis=0) == 0
        
        # 3. Обучение OrdinalEncoder для категориальных (на train)
        if 'x_cat' in dataset.data and self.config.get('cat_policy'):
            unknown_value = np.iinfo('int64').max - 3
            self.transformer_cat_ordinal = sklearn.preprocessing.OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=unknown_value,
                dtype='int64',
            ).fit(dataset.data['x_cat']['train'])
            
            # 4. Обучение OneHotEncoder (на train, если нужно)
            cat_policy = CatPolicy(self.config['cat_policy'])
            if cat_policy == CatPolicy.ONE_HOT:
                # Сначала кодируем train для OneHotEncoder
                X_cat_train_encoded = self.transformer_cat_ordinal.transform(
                    dataset.data['x_cat']['train']
                )
                self.transformer_cat_onehot = sklearn.preprocessing.OneHotEncoder(
                    handle_unknown='ignore',
                    sparse_output=False,
                    dtype=np.float32,
                ).fit(X_cat_train_encoded)
        
        self._fitted = True
        return self
    
    def transform(self, dataset: 'Dataset[np.ndarray]') -> 'Dataset[np.ndarray]':
        """
        Применяет обученные трансформеры ко всем частям.
        
        - Извлекает бинарные признаки по сохранённым индексам
        - Применяет QuantileTransformer/StandardScaler ко всем частям
        - Применяет OrdinalEncoder для бинарных (если использовался)
        - Применяет OrdinalEncoder для категориальных
        - Применяет OneHotEncoder (если нужно)
        
        Args:
            dataset: Датасет с train/val/test частями
            
        Returns:
            Новый датасет с трансформированными данными
        """
        assert self._fitted, "Must call fit() before transform()"
        
        # Создаём копию данных
        data = {k: dict(v) for k, v in dataset.data.items()}
        
        # 1. Извлечение бинарных признаков
        if self.feature_indices_bin is not None and 'x_num' in data:
            x_bin = {}
            x_num_remaining = {}
            
            # Проверяем, использовали ли мы OrdinalEncoder для бинарных
            if self.transformer_bin_ordinal is not None:
                # Используем OrdinalEncoder
                for k, v in data['x_num'].items():
                    x_bin[k] = self.transformer_bin_ordinal.transform(v[:, self.feature_indices_bin]).astype(bool)
                    x_num_remaining[k] = v[:, self.feature_indices_num]
            else:
                # Прямой каст (skip_bin_encoder=True и все столбцы 0/1)
                for k, v in data['x_num'].items():
                    x_bin[k] = v[:, self.feature_indices_bin].astype(_X_CAT_INT_DTYPE)
                    x_num_remaining[k] = v[:, self.feature_indices_num]
            
            # Объединяем с существующими x_bin
            if 'x_bin' in data:
                for k in x_bin:
                    x_bin[k] = np.concatenate([x_bin[k], data['x_bin'][k]], axis=-1)
            
            data['x_bin'] = x_bin
            data['x_num'] = x_num_remaining
        
        # 2. Применение QuantileTransformer/StandardScaler
        if self.transformer_num is not None and 'x_num' in data:
            x_num_transformed = {}
            for k, v in data['x_num'].items():
                # Шум НЕ добавляем при transform (он добавляется только при fit)
                x_num_transformed[k] = self.transformer_num.transform(v)
                # Заменяем NaN на 0
                x_num_transformed[k] = np.nan_to_num(x_num_transformed[k], copy=False)
            
            # Удаляем константные столбцы (используем сохранённую маску)
            if self._constant_mask is not None:
                active_mask = ~self._constant_mask
                for k in x_num_transformed:
                    x_num_transformed[k] = x_num_transformed[k][:, active_mask].astype(_X_NUM_DTYPE)
            else:
                for k in x_num_transformed:
                    x_num_transformed[k] = x_num_transformed[k].astype(_X_NUM_DTYPE)
            
            data['x_num'] = x_num_transformed
        elif hasattr(self, '_num_transformer_params') and 'x_num' in data:
            # Columnwise трансформация для memory_efficient
            x_num_transformed = {}
            n_cols = len(self._num_transformer_params)
            parts = list(data['x_num'].keys())
            
            # Определяем константные столбцы
            constant_mask = np.ptp(data['x_num']['train'], axis=0) == 0
            active_cols = np.nonzero(~constant_mask)[0]
            n_active = len(active_cols)
            
            for k in parts:
                x_num_transformed[k] = np.empty((len(data['x_num'][k]), n_active), dtype=_X_NUM_DTYPE)
            
            col_out = 0
            for col_in in range(n_cols):
                if constant_mask[col_in]:
                    continue
                
                normalizer = self._num_transformer_params[col_in]
                for k in parts:
                    part_col = data['x_num'][k][:, col_in]
                    transformed = normalizer.transform(part_col.reshape(-1, 1)).astype(_X_NUM_DTYPE).ravel()
                    transformed = np.nan_to_num(transformed, copy=False)
                    x_num_transformed[k][:, col_out] = transformed
                
                col_out += 1
            
            data['x_num'] = x_num_transformed
        
        # 3. Применение bin_policy (convert-to-cat)
        if 'x_bin' in data and self.config.get('bin_policy') == 'convert-to-cat':
            x_bin = data.pop('x_bin')
            if 'x_cat' not in data:
                data['x_cat'] = {k: np.where(np.isnan(v), 2.0, v).astype(_X_CAT_INT_DTYPE)
                                for k, v in x_bin.items()}
            else:
                for k in x_bin:
                    x_bin_as_cat = x_bin[k].astype(data['x_cat'][k].dtype)
                    data['x_cat'][k] = np.column_stack([data['x_cat'][k], x_bin_as_cat])
        
        # 4. Применение OrdinalEncoder для категориальных
        if self.transformer_cat_ordinal is not None and 'x_cat' in data:
            x_cat_encoded = {}
            for k, v in data['x_cat'].items():
                x_cat_encoded[k] = self.transformer_cat_ordinal.transform(v)
            
            # Обрабатываем неизвестные категории в val/test
            max_values = x_cat_encoded['train'].max(axis=0)
            unknown_value = np.iinfo('int64').max - 3
            for part in ['val', 'test']:
                if part in x_cat_encoded:
                    for column_idx in range(x_cat_encoded[part].shape[1]):
                        x_cat_encoded[part][x_cat_encoded[part][:, column_idx] == unknown_value, column_idx] = (
                            max_values[column_idx] + 1
                        )
            
            # 5. Применение OneHotEncoder (если нужно)
            if self.transformer_cat_onehot is not None:
                x_cat_onehot = {}
                for k, v in x_cat_encoded.items():
                    x_cat_onehot[k] = self.transformer_cat_onehot.transform(v)
                data['x_cat'] = x_cat_onehot
            else:
                data['x_cat'] = x_cat_encoded
        
        return dataclasses.replace(dataset, data=data)
    
    def fit_transform(self, dataset: 'Dataset[np.ndarray]') -> 'Dataset[np.ndarray]':
        """
        Обучает и применяет трансформеры.
        
        Args:
            dataset: Датасет с train/val/test частями
            
        Returns:
            Новый датасет с трансформированными данными
        """
        return self.fit(dataset).transform(dataset)

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
        return {k: data[v].copy() if copy else data[v] for k, v in split.items()}
    elif isinstance(data, Mapping):
        return type(data)(
            (k, apply_split(v, split, copy=copy))
            for k, v in data.items()  # type: ignore
        )
    else:
        raise ValueError(f'Unsupported type of `data`: {type(data)}')


def load_data(
    dataset_dir: str | Path, split_id: SplitIDLike
) -> dict[DataKey, dict[PartKey, np.ndarray]]:
    dataset_dir = _check_dataset_dir(dataset_dir)
    data = {x.stem: np.load(x) for x in dataset_dir.iterdir() if _is_npy_path(x)}
    split = load_split(dataset_dir, split_id)
    data = apply_split(data, split)

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
    X_num: dict[PartKey, np.ndarray], policy: None | str | NumPolicy, seed: None | int
) -> dict[PartKey, np.ndarray]:
    if policy is not None:
        policy = NumPolicy(policy)
        X_num_train = X_num['train']
        if policy == NumPolicy.STANDARD:
            normalizer = sklearn.preprocessing.StandardScaler()
        elif policy == NumPolicy.NOISY_QUANTILE:
            normalizer = sklearn.preprocessing.QuantileTransformer(
                n_quantiles=max(min(X_num['train'].shape[0] // 30, 1000), 10),
                output_distribution='normal',
                subsample=1_000_000_000,
                random_state=seed,
            )
            assert seed is not None
            X_num_train = X_num_train + np.random.RandomState(seed).normal(
                0.0, 1e-5, X_num_train.shape
            ).astype(X_num_train.dtype)
        else:
            raise ValueError(f'Unknown policy={policy}')

        normalizer.fit(X_num_train)
        del X_num_train

        # Transform one part at a time to reduce peak memory.
        X_num_transformed = {}
        for k, v in X_num.items():
            X_num_transformed[k] = normalizer.transform(v)
            del v
        X_num = X_num_transformed
        global TRANSFORMER_NUM
        TRANSFORMER_NUM = normalizer
        del normalizer
        gc.collect()

    # NOTE
    # (This is not a good way to process NaNs)
    # This is a quick hack to stop failing on some datasets because of NaNs.
    # NaNs are replaced with zeros (zero is the mean value for all features after
    # the conventional preprocessing techniques).
    X_num_nan = {k: np.nan_to_num(v) for k, v in X_num.items()}
    del X_num

    # Remove columns with one constant value.
    # np.ptp (peak-to-peak, max - min) is much faster than np.unique for this check.
    mask = np.ptp(X_num_nan['train'], axis=0) != 0
    X_num_masked = {k: v[:, mask] for k, v in X_num_nan.items()}
    del X_num_nan, mask

    X_num = {k: v.astype(_X_NUM_DTYPE) for k, v in X_num_masked.items()}
    del X_num_masked
    gc.collect()
    return X_num


def _transform_num_columnwise(
    X_num: dict[PartKey, np.ndarray],
    policy: NumPolicy,
    seed: int,
) -> dict[PartKey, np.ndarray]:
    """Column-wise transform to reduce peak RAM.

    Instead of fitting/transforming the entire matrix at once, process one column
    at a time.  This keeps peak memory proportional to a single column rather
    than the full dataset.
    """
    n_rows, n_cols = X_num['train'].shape
    parts = list(X_num.keys())

    # First pass: determine which columns are constant (to drop later).
    constant_mask = np.ptp(X_num['train'], axis=0) == 0  # (n_cols,)

    # Prepare output arrays (pre-allocated, no constant columns).
    active_cols = np.nonzero(~constant_mask)[0]
    n_active = len(active_cols)
    X_out = {k: np.empty((len(v), n_active), dtype=_X_NUM_DTYPE) for k, v in X_num.items()}

    col_out = 0
    for col_in in range(n_cols):
        if constant_mask[col_in]:
            continue

        # --- Fit on this single column (train) ---
        train_col = X_num['train'][:, col_in]  # view, 1-D

        if policy == NumPolicy.STANDARD:
            mean = train_col.mean()
            std = train_col.std()
            if std == 0:
                std = 1.0
        elif policy == NumPolicy.NOISY_QUANTILE:
            # Add noise to a copy of this one column only.
            fit_data = train_col.copy()
            noise = np.random.RandomState(seed + col_in).normal(
                0.0, 1e-5, fit_data.shape
            ).astype(fit_data.dtype)
            fit_data += noise
            normalizer = sklearn.preprocessing.QuantileTransformer(
                n_quantiles=max(min(n_rows // 30, 1000), 10),
                output_distribution='normal',
                random_state=seed,
            )
            normalizer.fit(fit_data.reshape(-1, 1))
            del fit_data, noise
        else:
            raise ValueError(f'Unknown policy={policy}')

        # --- Transform all parts for this column ---
        for k in parts:
            part_col = X_num[k][:, col_in]  # view, 1-D

            if policy == NumPolicy.STANDARD:
                transformed = ((part_col - mean) / std).astype(_X_NUM_DTYPE)
            else:
                transformed = normalizer.transform(part_col.reshape(-1, 1)).astype(_X_NUM_DTYPE).ravel()

            # Replace NaN with 0 (same as original behaviour).
            transformed = np.nan_to_num(transformed, copy=False)
            X_out[k][:, col_out] = transformed

        del train_col, part_col, transformed

        if policy == NumPolicy.NOISY_QUANTILE:
            del normalizer

        col_out += 1
        if col_in % 500 == 499:
            gc.collect()

    del X_num
    gc.collect()
    return X_out


def _extract_bin_from_num(
    X_num: dict[PartKey, np.ndarray],
    *,
    skip_encoder: bool = False,
) -> tuple[None | dict[PartKey, np.ndarray], None | dict[PartKey, np.ndarray]]:
    """Extract binary (0/1) features from numerical features.

    When `skip_encoder=True`, skips OrdinalEncoder and directly casts 0/1 columns
    to int64, avoiding large temporary allocations from sklearn's transformer.
    """
    X_num_all = np.concatenate(list(X_num.values()))
    has_missing_values = np.any(np.isnan(X_num_all), 0)
    unique_values = [np.unique(x) for x in X_num_all.T]
    unique_counts = np.array([len(x) for x in unique_values])
    del X_num_all  # Free the concatenated copy immediately

    bin_mask = (unique_counts == 2) & ~has_missing_values
    bin_idx = np.nonzero(bin_mask)[0]
    del unique_counts, has_missing_values

    if len(bin_idx) > 0:
        # Verify all detected "binary" columns are actually 0/1.
        is_01 = all(
            set(unique_values[i]) == {0.0, 1.0} for i in bin_idx
        )

        if skip_encoder and is_01:
            # Direct cast — no sklearn transformer, minimal copies.
            if len(bin_idx) == len(unique_values):
                # All features are binary.
                X_bin = {k: v.astype(_X_CAT_INT_DTYPE, copy=False) for k, v in X_num.items()}
                del X_num
                gc.collect()
                return X_bin, None
            else:
                # Some features are binary.
                X_bin = {k: v[:, bin_idx].astype(_X_CAT_INT_DTYPE, copy=False) for k, v in X_num.items()}
                X_num_remaining = {k: v[:, ~bin_mask] for k, v in X_num.items()}
                del X_num
                gc.collect()
                return X_bin, X_num_remaining
        else:
            transformer = sklearn.preprocessing.OrdinalEncoder(
                categories=[unique_values[i] for i in bin_idx]
            )
            transformer.fit(X_num['train'][:, bin_idx])
            if len(bin_idx) == len(unique_values):
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
        del unique_values
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

    X_cat_encoded = {}
    for k, v in X_cat.items():
        X_cat_encoded[k] = encoder.transform(v)
        del v
    global TRANSFORMER_CAT_ORDINAL
    TRANSFORMER_CAT_ORDINAL = encoder
    del X_cat, encoder

    max_values = X_cat_encoded['train'].max(axis=0)
    for part in ['val', 'test']:
        part = cast(PartKey, part)
        for column_idx in range(X_cat_encoded[part].shape[1]):
            X_cat_encoded[part][X_cat_encoded[part][:, column_idx] == unknown_value, column_idx] = (
                max_values[column_idx] + 1
            )
    del max_values

    if policy == CatPolicy.ORDINAL:
        return X_cat_encoded
    elif policy == CatPolicy.ONE_HOT:
        encoder = sklearn.preprocessing.OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False,
            dtype=np.float32,  # type: ignore
        )
        encoder.fit(X_cat_encoded['train'])
        X_cat_onehot = {}
        for k, v in X_cat_encoded.items():
            X_cat_onehot[k] = cast(np.ndarray, encoder.transform(v))
            del v
        global TRANSFORMER_CAT_ONEHOT
        TRANSFORMER_CAT_ONEHOT = encoder
        del X_cat_encoded, encoder
        gc.collect()
        return X_cat_onehot
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
        return Dataset(load_data(path, split_id), Task.from_dir(path, split_id))

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


def build_dataset(
    path: str | Path,
    split_id: SplitIDLike = DEFAULT_SPLIT_ID,
    *,
    extract_bin_from_num: bool = False,
    skip_bin_encoder: bool = False,
    num_policy: None | str | NumPolicy = None,
    num_memory_efficient: bool = False,
    bin_policy: None | str | BinPolicy = None,
    cat_policy: None | str | CatPolicy = None,
    task_score: None | str | Score = None,
    seed: int = 0,
    cache: bool = False,
) -> Dataset[np.ndarray]:
    path = Path(path).resolve()
    if cache:
        args = locals()
        args.pop('cache')
        args.pop('path')
        cache_path = env.get_cache_dir() / (
            f'build_dataset__{path.name}__{hashlib.md5(str(args).encode("utf-8")).hexdigest()}.pickle'
        )
        if cache_path.exists():
            cached_args, cached_value = pickle.loads(cache_path.read_bytes())
            assert args == cached_args, f'Hash collision for {cache_path}'
            logger.info(f'Using cached dataset: {cache_path.name}')
            return cached_value
    else:
        args = None
        cache_path = None

    print(f'Loading dataset from {path.name}...')
    dataset = Dataset.from_dir(path, split_id)
    if task_score is not None:
        dataset = dataclasses.replace(
            dataset, task=dataclasses.replace(dataset.task, score=Score(task_score))
        )

    # Reset global feature indices
    global FEATURE_INDICES_NUM, FEATURE_INDICES_CAT, FEATURE_INDICES_BIN
    FEATURE_INDICES_NUM = None
    FEATURE_INDICES_CAT = None
    FEATURE_INDICES_BIN = None

    if 'x_num' in dataset.data and extract_bin_from_num:
        print('Extracting binary features from numerical...')
        
        # Compute binary feature indices before extraction
        x_num_all = np.concatenate(list(dataset.data['x_num'].values()))
        has_missing = np.any(np.isnan(x_num_all), 0)
        unique_counts = np.array([len(np.unique(col)) for col in x_num_all.T])
        bin_mask = (unique_counts == 2) & ~has_missing
        bin_indices = np.nonzero(bin_mask)[0]
        num_indices = np.nonzero(~bin_mask)[0]
        del x_num_all, has_missing, unique_counts, bin_mask
        
        extracted_x_bin, remaining_x_num = _extract_bin_from_num(
            dataset.data['x_num'], skip_encoder=skip_bin_encoder
        )
        if extracted_x_bin is not None:
            # Store feature indices
            FEATURE_INDICES_BIN = bin_indices
            if remaining_x_num is None:
                del dataset.data['x_num']
            else:
                dataset.data['x_num'] = remaining_x_num
                FEATURE_INDICES_NUM = num_indices
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
        print(f'Transforming numerical features (policy={num_policy}, memory_efficient={num_memory_efficient})...')
        if num_memory_efficient and num_policy is not None:
            dataset.data['x_num'] = _transform_num_columnwise(
                dataset.data['x_num'], NumPolicy(num_policy), seed
            )
        else:
            dataset.data['x_num'] = transform_num(dataset.data['x_num'], num_policy, seed)

    if 'x_bin' in dataset.data:
        if bin_policy is not None:
            print(f'Transforming binary features (policy={bin_policy})...')
            bin_policy = BinPolicy(bin_policy)
            if bin_policy == BinPolicy.CONVERT_TO_CAT:
                dataset.convert_bin_features_to_cat_()
            else:
                raise ValueError(f'Unknown {bin_policy=}')

    if 'x_cat' in dataset.data:
        print(f'Transforming categorical features (policy={cat_policy})...')
        dataset.data['x_cat'] = transform_cat(dataset.data['x_cat'], cat_policy)

    if cache_path is not None:
        with tempfile.NamedTemporaryFile('wb') as tmp_cache_file:
            tmp_cache_file.write(pickle.dumps((args, dataset)))
            os.rename(tmp_cache_file.name, cache_path)
    return dataset
