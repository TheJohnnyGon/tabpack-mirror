# /// script
# requires-python = "==3.12.9"
# dependencies = [
#     "catboost==1.2.8",
#     "kaggle==1.7.4.5",
#     "loguru==0.7.3",
#     "numpy==2.3.1",
#     "openml==0.15.1",
#     "pandas==2.3.1",
#     "scikit-learn==1.7.0",
#     "tqdm==4.67.1",
# ]
# ///

"""Prepare datasets for doing tabular ML research.

The script produces *two* directories:

1. The first one is the "raw" data, which tries to stay close to the original data.
   It usually serves as a historical reference.
2. The second one is the preprocessed version of the first one,
   which is usually used for everyday work.

**Requirements**

Before running the script:

1. Get Kaggle API token and put it to `~/.kaggle/kaggle.json` as described here: https://www.kaggle.com/docs/api
2. Enter the following Kaggle competition: https://www.kaggle.com/competitions/otto-group-product-classification-challenge
3. Download TabReD: https://github.com/yandex-research/tabred

**Usage**

```
uv run <path to this script> --help
```

**Implementation notes**

An important requirement for this script is to produce exactly the same datasets
as in existing benchmarks and papers; this, in particular, applies to _splits_.
To meet this requirement, the script:

* Follows the official instructions (e.g. for existing benchmarks).
* Follows similar scripts. In particular, this script:
  https://github.com/yandex-research/rtdl-num-embeddings/blob/a8fc25025c83f2321c63ff127a3bcef83bb1bfb5/bin/datasets.py
  is used as a reference when preparing some of the datasets called "independent" in
  this script.
"""

import argparse
import enum
import json
import platform
import shutil
import sys
import tempfile
import time
import typing
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import catboost.datasets
import kaggle
import numpy as np
import openml
import pandas as pd
import sklearn.datasets
import sklearn.preprocessing
from loguru import logger
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from tqdm import tqdm

# This little fellow below is the cornerstone of split reproducibility.
# Never ever change it.
_SEED = 0

type DataKey = Literal['x_num', 'x_cat', 'x_bin', 'x_meta', 'groups', 'y']
type Splits = dict[str, 'np.ndarray | Splits']


@enum.unique
class TaskType(enum.Enum):
    REGRESSION = 'regression'
    BINCLASS = 'binclass'
    MULTICLASS = 'multiclass'


@enum.unique
class TaskScore(enum.Enum):
    ACCURACY = 'accuracy'
    ROC_AUC = 'roc-auc'
    CROSS_ENTROPY = 'cross-entropy'
    RMSE = 'rmse'
    R2 = 'r2'


_VALID_SCORES = {
    TaskType.REGRESSION: (TaskScore.RMSE, TaskScore.R2),
    TaskType.BINCLASS: (TaskScore.ACCURACY, TaskScore.ROC_AUC, TaskScore.CROSS_ENTROPY),
    TaskType.MULTICLASS: (TaskScore.ACCURACY, TaskScore.CROSS_ENTROPY),
}


@dataclass(frozen=True)
class Task:
    type: TaskType
    score: TaskScore

    def __post_init__(self) -> None:
        assert self.score in _VALID_SCORES[self.type]


def configure_logging():
    logger.remove()
    logger.add(sys.stderr, format='<level>{message}</level>')


def load_json(p: str | Path) -> dict[str, Any]:
    with open(p) as f:
        return json.load(f)


def dump_json(value, p: str | Path, **kwargs) -> None:
    with open(p, 'w') as f:
        return json.dump(value, f, **kwargs)


def unzip(path: Path, members: None | list[str] = None) -> None:
    with zipfile.ZipFile(path) as f:
        f.extractall(path.parent, members)


def _validate_data(data: dict[DataKey, np.ndarray]) -> None:
    assert data
    some_value = next(iter(data.values()))
    for key, value in data.items():
        assert isinstance(value, np.ndarray)
        assert len(value) == len(some_value)
        if key == 'y':
            # For now, only single-task labels are supported.
            assert value.ndim == 1
        elif key.startswith('x_'):
            assert value.ndim == 2


def _validate_splits(dataset_size: int, splits: Splits, *, full: bool) -> int:
    def impl(node: Splits, split: dict[str, np.ndarray], n_splits: int) -> int:
        part_keys = []
        node_keys = []
        for key, value in node.items():
            if isinstance(value, np.ndarray):
                assert key not in split, (
                    f'The splits contain the same part "{key}" on multiple levels'
                )
                assert value.size > 0
                assert value.ndim == 1
                assert np.isdtype(value.dtype, 'integral')
                assert np.all(value >= 0)
                assert np.all(value < dataset_size)
                part_keys.append(key)
                split[key] = value
            else:
                node_keys.append(key)

        # Now as split is filled with new part indices, recurse into inner nodes.
        for key in node_keys:
            n_splits = impl(node[key], split, n_splits)  # type: ignore

        if not node_keys:
            # The split is fully assembled and can be validated.
            all_idx = np.concatenate(list(split.values()))
            unique_idx = np.unique(all_idx)
            assert len(unique_idx) == len(all_idx), (
                'A split containing duplicated indices was found'
            )
            if full:
                assert np.all(unique_idx == np.arange(dataset_size))
            n_splits += 1

        # Remove the part indices added on the current level.
        for key in part_keys:
            del split[key]

        return n_splits

    return impl(splits, {}, 0)


def _save_splits(path: Path, splits: Splits) -> None:
    path.mkdir()

    for key, value in splits.items():
        if isinstance(value, np.ndarray):
            assert np.isdtype(value.dtype, 'integral')
            # The split data type is the only fixed data type for the _raw_ data.
            np.save(path / f'{key}.npy', value.astype(_SPLIT_DTYPE), allow_pickle=False)
        else:
            assert isinstance(value, dict)
            _save_splits(path / key, value)


def _save_default_split(path: Path, default_split_id: None | tuple[str, ...]) -> None:
    default_split_dir = path / 'default'

    if default_split_id is None:
        assert default_split_dir.exists()
        return

    assert default_split_id
    assert not default_split_dir.exists()
    default_split_dir.mkdir()

    subdir = path
    for dirname in default_split_id:
        subdir = subdir / dirname
        for path in subdir.iterdir():
            if path.suffix == '.npy':
                link_path = default_split_dir / path.name
                assert not link_path.exists()
                link_path.symlink_to(path.relative_to(default_split_dir, walk_up=True))


def save_dataset(
    dataset_dir: Path,
    *,
    task: Task,
    data: dict[DataKey, np.ndarray],
    splits: Splits,
    default_split_id: None | tuple[str, ...] = None,
    full_splits: bool = True,
):
    assert not dataset_dir.exists()
    _validate_data(data)
    dataset_size = len(next(iter(data.values())))
    n_splits = _validate_splits(dataset_size, splits, full=full_splits)

    info = {
        'task': {
            'type': task.type.value,
            'score': task.score.value,
        },
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        tmp_splits_dir = tmp_dir / 'splits'

        dump_json(info, tmp_dir / 'info.json', indent=4)
        for key, value in data.items():
            np.save(tmp_dir / f'{key}.npy', value)
        _save_splits(tmp_splits_dir, splits)
        _save_default_split(tmp_splits_dir, default_split_id)

        dataset_dir.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir.rename(dataset_dir)

    logger.info(f'Saved {dataset_dir}')
    print(f'Path:    {dataset_dir}')
    print(f'Info:    {info}')
    print(f'Data:    {sorted(data.keys())}')
    print(f'Splits:  {n_splits}')


def prepare_dataset(
    dataset_dir: Path, make_dataset_fn: Callable[..., dict[str, Any]], **kwargs
) -> None:
    if dataset_dir.exists():
        logger.info(f'Dataset already exists: {dataset_dir}')
        return

    print()
    logger.info(f'Preparing {dataset_dir} ...')
    save_dataset(dataset_dir, **make_dataset_fn(**kwargs))


# ======================================================================================
# Independent datasets
# ======================================================================================
_INDEPENDENT_RELATIVE_PART_SIZE = 0.2


@enum.unique
class _SplitType(enum.Enum):
    TRAIN_VAL = enum.auto()
    TRAIN_VAL_TEST = enum.auto()


def _make_split(size: int, stratify: None | np.ndarray, type: _SplitType) -> Splits:
    """
    NOTE
    This function closely follows the following code to produce the same splits:
    https://github.com/yandex-research/rtdl-num-embeddings/blob/a8fc25025c83f2321c63ff127a3bcef83bb1bfb5/bin/datasets.py#L136
    """
    all_idx = np.arange(size, dtype=np.int64)
    a_idx, b_idx = train_test_split(
        all_idx,
        test_size=_INDEPENDENT_RELATIVE_PART_SIZE,
        stratify=stratify,
        random_state=_SEED + (1 if type == _SplitType.TRAIN_VAL else 0),
    )

    if type == _SplitType.TRAIN_VAL:
        return {'train': a_idx, 'val': b_idx}

    a1_idx, a2_idx = train_test_split(
        a_idx,
        test_size=_INDEPENDENT_RELATIVE_PART_SIZE,
        stratify=None if stratify is None else stratify[a_idx],
        random_state=_SEED + 1,
    )
    return {'train': a1_idx, 'val': a2_idx, 'test': b_idx}


def _cast_X_y(X_y):
    return cast(tuple[np.ndarray, np.ndarray], X_y)


def _make_churn_modelling() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        kaggle.api.dataset_download_files(
            'shrutimechlearn/churn-modelling', path=tmp, unzip=True
        )
        df = pd.read_csv(Path(tmp) / 'Churn_Modelling.csv')

    df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])
    df['Gender'] = df['Gender'].astype('category').cat.codes.values.astype(np.int64)
    y = df.pop('Exited').values.astype(np.int64)

    num_columns = [
        'CreditScore',
        'Gender',
        'Age',
        'Tenure',
        'Balance',
        'NumOfProducts',
        'EstimatedSalary',
        'HasCrCard',
        'IsActiveMember',
        'EstimatedSalary',
    ]
    cat_columns = ['Geography']
    num_columns_set = frozenset(num_columns)
    cat_columns_set = frozenset(cat_columns)
    assert not (num_columns_set & cat_columns_set)
    assert num_columns_set | cat_columns_set == frozenset(df.columns.tolist())

    return {
        'task': Task(TaskType.BINCLASS, TaskScore.ACCURACY),
        'data': {
            'x_num': df[num_columns].values,
            # TODO: is `.astype(str)` really needed?
            'x_cat': df[cat_columns].astype(str).values,
            'y': y,
        },
        'splits': {'default': _make_split(len(y), y, _SplitType.TRAIN_VAL_TEST)},
    }


def _make_california_housing() -> dict[str, Any]:
    x_num, y = _cast_X_y(sklearn.datasets.fetch_california_housing(return_X_y=True))
    return {
        'task': Task(TaskType.REGRESSION, TaskScore.RMSE),
        'data': {'x_num': x_num, 'y': y},
        'splits': {'default': _make_split(len(y), None, _SplitType.TRAIN_VAL_TEST)},
    }


def _make_house_16h() -> dict[str, Any]:
    bunch = sklearn.datasets.fetch_openml(data_id=574, as_frame=True)
    return {
        'task': Task(TaskType.REGRESSION, TaskScore.RMSE),
        'data': {'x_num': bunch['data'].values, 'y': bunch['target'].values},
        'splits': {
            'default': _make_split(
                len(bunch['target']), None, _SplitType.TRAIN_VAL_TEST
            )
        },
    }


def _make_adult() -> dict[str, Any]:
    # NOTE
    # As with some other independent datasets, the way this dataset
    # is downloaded and processed should be treated as a historical
    # artifact inherited from the script mentioned in the docstring
    # at the beginning of this file.

    df_trainval, df_test = catboost.datasets.adult()
    assert (df_trainval.dtypes == df_test.dtypes).all()
    assert (df_trainval.columns == df_test.columns).all()
    trainval_size = len(df_trainval)

    df = pd.concat([df_trainval, df_test], ignore_index=True)
    y = (df.pop('income') == '>50K').values.astype('int64')
    categorical_mask = df.dtypes != np.float64
    data = {
        'x_num': df.loc[:, ~categorical_mask].values,
        'x_cat': df.loc[:, categorical_mask].values,
        'y': y,
    }
    data['x_cat'][data['x_cat'] == 'nan'] = '__nan__'

    return {
        'task': Task(TaskType.BINCLASS, TaskScore.ACCURACY),
        'data': data,
        'splits': {
            'default': {
                **_make_split(trainval_size, y[:trainval_size], _SplitType.TRAIN_VAL),
                'test': np.arange(
                    trainval_size, trainval_size + len(df_test), dtype=np.int64
                ),
            }
        },
    }


def _make_otto_group_products() -> dict[str, Any]:
    competition_name = 'otto-group-product-classification-challenge'
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        kaggle.api.competition_download_files(competition_name, path=tmp)
        with zipfile.ZipFile(tmp / f'{competition_name}.zip') as f:
            f.extractall(tmp)
            del f
        df = pd.read_csv(tmp / 'train.csv')

    df.drop(columns=['id'], inplace=True)
    y = _label_encoding(
        df.pop('target').map(lambda x: int(x.split('_')[-1]) - 1).values  # type: ignore[code]
    )

    return {
        'task': Task(TaskType.MULTICLASS, TaskScore.ACCURACY),
        'data': {'x_num': df.values, 'y': y},
        'splits': {'default': _make_split(len(y), y, _SplitType.TRAIN_VAL_TEST)},
    }


def _make_diamonds() -> dict[str, Any]:
    bunch = sklearn.datasets.fetch_openml(data_id=42225, as_frame=True)
    cat_mask = bunch['data'].dtypes == 'category'

    return {
        'task': Task(TaskType.REGRESSION, TaskScore.RMSE),
        'data': {
            'x_num': bunch['data'].loc[:, ~cat_mask].values,
            'x_cat': bunch['data'].loc[:, cat_mask].values,
            'y': bunch['target'].values,
        },
        'splits': {
            'default': _make_split(
                len(bunch['target']), None, _SplitType.TRAIN_VAL_TEST
            )
        },
    }


def _make_higgs_small() -> dict[str, Any]:
    bunch = sklearn.datasets.fetch_openml(data_id=23512, as_frame=True)

    x_num = bunch['data'].values
    # NOTE
    # `.values` does _not_ return `np.ndarray` for categorical data,
    # so `.to_numpy` is used.
    y = bunch['target'].to_numpy()

    nan_mask = np.isnan(x_num)
    invalid_object_mask = nan_mask.any(1)
    # There is exactly one object with missing values ...
    assert invalid_object_mask.sum() == 1
    # ... and it has nine missing values ...
    assert nan_mask[invalid_object_mask].sum() == 9
    # ... so let's drop it:
    valid_object_mask = ~invalid_object_mask
    x_num = x_num[valid_object_mask]
    y = y[valid_object_mask]

    return {
        'task': Task(TaskType.BINCLASS, TaskScore.ACCURACY),
        'data': {'x_num': x_num, 'y': y},
        'splits': {'default': _make_split(len(y), y, _SplitType.TRAIN_VAL_TEST)},
    }


def _make_black_friday() -> dict[str, Any]:
    bunch = sklearn.datasets.fetch_openml(data_id=41540, as_frame=True)
    cat_mask = bunch['data'].dtypes == 'category'

    return {
        'task': Task(TaskType.REGRESSION, TaskScore.RMSE),
        'data': {
            'x_num': bunch['data'].loc[:, ~cat_mask].values,
            'x_cat': bunch['data'].loc[:, cat_mask].values,
            'y': bunch['target'].values,
        },
        'splits': {
            'default': _make_split(
                len(bunch['target']), None, _SplitType.TRAIN_VAL_TEST
            )
        },
    }


def _make_covtype2() -> dict[str, Any]:
    # NOTE
    # The way this dataset is downloaded and split follows the RTDL script
    # referenced at the beginning of this file.
    #
    # Also, an additional preprocessing is performed. Most of the binary features in
    # this dataset look like a one-hot-encoding representation of one categorical
    # feature. These binary features are converted back to the hypothetical categorical
    # feature, which gives the "Covtype2" from the TabM paper:
    # https://arxiv.org/abs/2410.24210

    x_num, y = _cast_X_y(sklearn.datasets.fetch_covtype(return_X_y=True))

    # The cardinality of the one categorical feature discussed above.
    cat_cardinality = 40
    x_num, x_cat_one_hot = np.split(x_num, [x_num.shape[1] - cat_cardinality], axis=1)  # type: ignore
    # Validate the one-hot-encoding representation.
    assert np.all(x_cat_one_hot.sum(1) == 1)
    # Also, check that the cardinality is chosen correctly by testing a greater value:
    assert not np.all(x_num[:, -(cat_cardinality + 1) :].sum(1) == 1)

    # Now, restore the categorical features from its OHE representation.
    ohe_nonzero_rows, ohe_nonzero_columns = np.nonzero(x_cat_one_hot)
    assert np.all(ohe_nonzero_rows == np.arange(len(x_cat_one_hot)))
    x_cat = ohe_nonzero_columns[:, None]

    return {
        'task': Task(TaskType.MULTICLASS, TaskScore.ACCURACY),
        'data': {'x_num': x_num, 'x_cat': x_cat, 'y': y},
        'splits': {'default': _make_split(len(y), y, _SplitType.TRAIN_VAL_TEST)},
    }


def _make_mslr_web10k() -> dict[str, Any]:
    # The official MSLR-WEB10K data reuploaded to Dropbox.
    # For the official description, references and license, see:
    # https://www.microsoft.com/en-us/research/project/mslr
    data_url = 'https://www.dropbox.com/s/572rj8m5f9l2nz5/MSLR-WEB10K.zip?dl=1'

    def parse_file(path):
        with open(path) as f:
            rows = []
            for line in f:
                line = line.split()
                rows.append(
                    np.fromiter(
                        (float(item.split(':', 1)[-1]) for item in line),
                        np.float32,
                        len(line),
                    )
                )
            rows = np.array(rows)
        return {
            'x_num': rows[:, 2:],
            'y': rows[:, 0],
            'groups': rows[:, 1].astype(np.int64),
        }

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        archive_path = tmp / 'data.zip'
        logger.info(f'Downloading {data_url}')
        urllib.request.urlretrieve(data_url, archive_path)
        unzip(archive_path, ['Fold1/test.txt', 'Fold1/train.txt', 'Fold1/vali.txt'])

        stem_to_part = {'train': 'train', 'vali': 'val', 'test': 'test'}
        fold_dir = tmp / 'Fold1'
        parts = []
        parts_data = []
        for path in fold_dir.iterdir():
            logger.info(f'Parsing {path.name} ...')
            parts.append(stem_to_part[path.stem])
            parts_data.append(parse_file(path))

    data = dict(zip(parts, parts_data))
    train_size = len(data['train']['y'])
    val_size = len(data['val']['y'])
    test_size = len(data['test']['y'])
    return {
        'task': Task(TaskType.REGRESSION, TaskScore.RMSE),
        'data': {
            key: np.concatenate([data[part][key] for part in ['train', 'val', 'test']])
            for key in data['train'].keys()
        },
        'splits': {
            'default': {
                'train': np.arange(train_size),
                'val': np.arange(train_size, train_size + val_size),
                'test': np.arange(
                    train_size + val_size, train_size + val_size + test_size
                ),
            }
        },
    }


def prepare_independent_datasets(data_dir: Path) -> None:
    prepare_dataset(data_dir / 'churn', _make_churn_modelling)
    prepare_dataset(data_dir / 'california', _make_california_housing)
    prepare_dataset(data_dir / 'house', _make_house_16h)
    prepare_dataset(data_dir / 'adult', _make_adult)
    prepare_dataset(data_dir / 'otto', _make_otto_group_products)
    prepare_dataset(data_dir / 'diamond', _make_diamonds)
    prepare_dataset(data_dir / 'higgs-small', _make_higgs_small)
    prepare_dataset(data_dir / 'black-friday', _make_black_friday)
    prepare_dataset(data_dir / 'covtype2', _make_covtype2)
    prepare_dataset(data_dir / 'microsoft', _make_mslr_web10k)


# ======================================================================================
# TabReD
# ======================================================================================
_TABRED_BENCHMARK_NAME = 'TabReD'
_TABRED_BENCHMARK_DIRNAME = _TABRED_BENCHMARK_NAME.lower()


def _make_tabred_dataset(tabred_dataset_dir: Path) -> dict[str, Any]:
    info = load_json(tabred_dataset_dir / 'info.json')
    # In TabReD, the score is not specified in regression tasks
    # and is assumed to be RMSE.
    task_type = TaskType(info['task_type'])
    if task_type == TaskType.REGRESSION:
        assert 'score' not in info
        score = TaskScore.RMSE
    else:
        score = TaskScore(info['score'])
    task = Task(task_type, score)

    x_paths = [
        tabred_dataset_dir / f'X_{x_type}.npy'
        for x_type in ['num', 'cat', 'bin', 'meta']
    ]
    data = {
        **{path.stem.lower(): np.load(path) for path in x_paths if path.exists()},
        'y': np.load(tabred_dataset_dir / 'Y.npy'),
    }

    split_prefix = 'split-'
    splits = {
        split_dir.name.removeprefix(split_prefix): {
            'train': np.load(split_dir / 'train_idx.npy'),
            'val': np.load(split_dir / 'val_idx.npy'),
            'test': np.load(split_dir / 'test_idx.npy'),
        }
        for split_dir in tabred_dataset_dir.iterdir()
        if split_dir.name.startswith(split_prefix)
    }

    return {
        'task': task,
        'data': data,
        'splits': splits,
        # In TabReD, there are splits that do not fully utilize the data
        # (i.e. some of the dataset objects remain unused).
        'full_splits': False,
    }


def prepare_tabred_benchmark(benchmark_dir: Path, *, tabred_official_dir: Path) -> None:
    # Sort the directory content to make the order deterministic.
    for tabred_dataset_dir in sorted(tabred_official_dir.iterdir()):
        if (
            platform.system() == 'Darwin'
            and not tabred_dataset_dir.is_dir()
            and tabred_dataset_dir.name == '.DS_Store'
        ):
            continue
        prepare_dataset(
            benchmark_dir / tabred_dataset_dir.name,
            _make_tabred_dataset,
            tabred_dataset_dir=tabred_dataset_dir,
        )


# ======================================================================================
# TabArena
# ======================================================================================
_TABARENA_BENCHMARK_NAME = 'TabArena'
_TABARENA_BENCHMARK_DIRNAME = _TABARENA_BENCHMARK_NAME.lower()
_TABARENA_OPENML_SUITE = 'tabarena-v0.1'
_TABARENA_TASK_TO_SCORE: dict[TaskType, TaskScore] = {
    TaskType.REGRESSION: TaskScore.RMSE,
    TaskType.BINCLASS: TaskScore.ROC_AUC,
    TaskType.MULTICLASS: TaskScore.CROSS_ENTROPY,
}
_TABARENA_N_INNER_SPLITS = 8
# The following mappings are based on this post by TabArena authors:
# https://github.com/autogluon/tabarena/issues/238#issuecomment-3491609117
_TABARENA_ADDITIONAL_CAT_COLUMNS = {
    363618: ['poutcome'],
    363624: ['avgAge', 'contributionPrivateThirdPartyInsurance'],
    363679: ['experience'],
    363681: ['coupon'],
    363700: ['ghazard'],
}
_TABARENA_ADDITIONAL_TIMESTAMP_COLUMNS = {
    363684: [('Dt_Customer', '%Y-%m-%d')],
}


def _get_tabarena_task_type(
    openml_task_type: openml.tasks.TaskType, y: pd.Series
) -> TaskType:
    if openml_task_type == openml.tasks.TaskType.SUPERVISED_REGRESSION:
        return TaskType.REGRESSION

    elif openml_task_type == openml.tasks.TaskType.SUPERVISED_CLASSIFICATION:
        n_unique_labels = y.nunique()
        assert n_unique_labels >= 2
        return TaskType.BINCLASS if n_unique_labels == 2 else TaskType.MULTICLASS

    else:
        raise ValueError(f'Unsupported OpenMLtask type: {openml_task_type}')


def _make_tabarena_repeat_splits(
    task: openml.tasks.OpenMLTask,
    task_type: TaskType,
    y: np.ndarray,
    *,
    repeat: int,
    n_folds: int,
) -> Splits:
    is_regression = task_type == TaskType.REGRESSION
    fold_splits = {}
    for fold in range(n_folds):
        trainval_idx, test_idx = task.get_train_test_split_indices(
            fold=fold, repeat=repeat
        )
        inner_splits = {
            str(i): {'train': trainval_idx[train_idx], 'val': trainval_idx[val_idx]}
            for i, (train_idx, val_idx) in enumerate(
                (KFold if is_regression else StratifiedKFold)(
                    n_splits=_TABARENA_N_INNER_SPLITS,
                    shuffle=True,
                    random_state=repeat * 1000 + fold,
                ).split(trainval_idx, *(() if is_regression else (y[trainval_idx],)))
            )
        }
        fold_splits[str(fold)] = {'test': test_idx, **inner_splits}
    return fold_splits


def _get_openml_dataset_name(task_id: int) -> str:
    return openml.datasets.get_dataset(openml.tasks.get_task(task_id).dataset_id).name


def _make_tabarena_dataset(task_id: int) -> dict[str, Any]:
    openml_task = openml.tasks.get_task(task_id, download_splits=True)
    dataset = openml.datasets.get_dataset(openml_task.dataset_id, download_data=True)

    # Fetch the data.
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=openml_task.target_name,  # type: ignore
        dataset_format='dataframe',
    )
    X = cast(pd.DataFrame, X)
    y = cast(pd.Series, y)

    # Prepare the properties.
    task_type = _get_tabarena_task_type(openml_task.task_type_id, y)
    task = Task(task_type, _TABARENA_TASK_TO_SCORE[task_type])

    # Prepare the data dictionary.
    cat_columns = []
    num_columns = []
    for attribute, is_cat in zip(attribute_names, categorical_indicator):
        if is_cat or attribute in _TABARENA_ADDITIONAL_CAT_COLUMNS.get(task_id, ()):
            cat_columns.append(attribute)
        else:
            num_columns.append(attribute)
    bin_columns = X[num_columns].columns[X[num_columns].dtypes == bool].tolist()  # noqa: E721
    if bin_columns:
        num_columns = [x for x in num_columns if x not in bin_columns]

    timestamp_columns = _TABARENA_ADDITIONAL_TIMESTAMP_COLUMNS.get(task_id, [])
    if timestamp_columns:
        assert all(column in num_columns for column, _ in timestamp_columns)
        for column, format_ in timestamp_columns:
            X[column] = (
                # `.astype(int)` produces nanoseconds.
                pd.to_datetime(X[column], format=format_).astype(int) / 10**9
            )

    data = {}
    if num_columns:
        data['x_num'] = X.loc[:, num_columns].to_numpy()
    if cat_columns:
        data['x_cat'] = X.loc[:, cat_columns].to_numpy()
    if bin_columns:
        data['x_bin'] = X.loc[:, bin_columns].to_numpy()
    data['y'] = y.to_numpy()

    # Prepare the splits.

    # NOTE
    # TabArena uses different number of repeats depending on the dataset size:
    #
    # dataset_size: int = dataset.qualities['NumberOfInstances']  # type: ignore
    # if dataset_size < 2_500:
    #     n_repeats = 10
    # elif dataset_size <= 250_000:  # TODO: strict or non-strict?
    #     n_repeats = 3
    # else:
    #     n_repeats = 1
    #
    # However, TabArena actually provides 10 repeats for all datasets,
    # so let's save them all just in case (at the cost of ~0.5GB extra disk space).
    n_repeats = 10

    _, n_folds, _ = openml_task.get_split_dimensions()
    # NOTE
    # The structure of TabArena splits is as follows
    # ("Outer" is used as a synonym of "Fold"):
    # {
    #     <Repeat>: {
    #         <Outer split>: {
    #             "test": <NumPy Array>,
    #             <Inner split>: {
    #                 "train": <NumPy Array>,
    #                 "val": <NumPy Array>,
    #             }
    #         }
    #     }
    # }
    splits: Splits = {
        str(repeat): _make_tabarena_repeat_splits(
            openml_task, task_type, data['y'], repeat=repeat, n_folds=n_folds
        )
        for repeat in range(n_repeats)
    }

    return {
        'task': task,
        'data': data,
        'splits': splits,
        # The default split corresponds to
        # the first inner split of the first fold of the first repeat.
        'default_split_id': ('0', '0', '0'),
    }


def prepare_tabarena_benchmark(
    benchmark_dir: Path, *, pause_duration: None | float = None
) -> None:
    benchmark_suite = openml.study.get_suite(_TABARENA_OPENML_SUITE)
    assert benchmark_suite.tasks is not None

    for i, task_id in enumerate(benchmark_suite.tasks):
        if i > 0 and pause_duration is not None:
            time.sleep(pause_duration)
        prepare_dataset(
            benchmark_dir / _get_openml_dataset_name(task_id),
            _make_tabarena_dataset,
            task_id=task_id,
        )


# ======================================================================================
# Preprocessing
# ======================================================================================
_X_NUM_DTYPE = np.float32
_X_BIN_DTYPE = np.float32
_X_CAT_INT_DTYPE = np.int64
_X_CAT_STR_DTYPE = np.str_
_Y_REG_DTYPE = np.float32
_Y_CLF_DTYPE = np.int64
_SPLIT_DTYPE = np.int32


def _numpy_save(path: Path, *args, **kwargs):
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.save(path, *args, **kwargs)


def _label_encoding(values: np.ndarray) -> np.ndarray:
    return sklearn.preprocessing.LabelEncoder().fit_transform(values)  # type: ignore


def preprocess_dataset(*, src_dataset_dir: Path, dst_dataset_dir: Path) -> None:
    if dst_dataset_dir.exists():
        logger.info(f'The dataset is already preprocessed: {dst_dataset_dir}')
        return

    task_type = TaskType(load_json(src_dataset_dir / 'info.json')['task']['type'])

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # Preprocess the NumPy data.
        for src_path in src_dataset_dir.glob('*.npy'):
            stem = src_path.stem
            value = np.load(src_path, allow_pickle=True)

            if stem == 'x_num':
                assert np.isdtype(value.dtype, 'numeric')
                dtype = _X_NUM_DTYPE

            elif stem == 'x_cat':
                dtype = (
                    _X_CAT_INT_DTYPE
                    if value.dtype == np.bool or np.isdtype(value.dtype, 'integral')
                    else _X_CAT_STR_DTYPE
                )

            elif stem == 'x_bin':
                assert value.dtype == np.bool or np.isdtype(value.dtype, 'numeric')
                assert np.unique(value[~np.isnan(value)]).tolist() == [0, 1]
                dtype = _X_BIN_DTYPE

            elif stem == 'y':
                if task_type == TaskType.REGRESSION:
                    assert np.isdtype(value.dtype, 'numeric')
                    dtype = _Y_REG_DTYPE
                else:
                    assert (
                        np.isdtype(value.dtype, 'integral')
                        or value.dtype == np.bool
                        or value.dtype.kind in ('U', 'S', 'T', 'O')  # Strings
                    )
                    value = _label_encoding(value)
                    dtype = _Y_CLF_DTYPE

            elif stem in typing.get_args(DataKey.__value__):
                # Preserve the original data type.
                dtype = value.dtype

            else:
                raise ValueError(f'Unsupported data key: {stem}')

            _numpy_save(
                tmp / src_path.relative_to(src_dataset_dir), value.astype(dtype)
            )

        # Preprocess the splits.
        for src_path in src_dataset_dir.joinpath('splits').glob('**/*.npy'):
            dst_path = tmp / src_path.relative_to(src_dataset_dir)
            if src_path.is_symlink():
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src_path, dst_path, follow_symlinks=False)
            else:
                value = np.load(src_path)
                assert np.isdtype(value.dtype, 'integral'), (
                    'Raw splits are expected to have an integer data type'
                )
                _numpy_save(dst_path, value.astype(_SPLIT_DTYPE))

        # Copy everything else.
        for src_path in src_dataset_dir.glob('**/*'):
            tmp_path = tmp / src_path.relative_to(src_dataset_dir)
            if not tmp_path.exists(follow_symlinks=False):
                tmp_path.parent.mkdir(parents=True, exist_ok=True)
                if src_path.is_dir():
                    tmp_path.mkdir()
                else:
                    shutil.copyfile(src_path, tmp_path, follow_symlinks=False)

        dst_dataset_dir.parent.mkdir(parents=True, exist_ok=True)
        tmp.rename(dst_dataset_dir)


def preprocess_all_datasets(*, src: Path, dst: Path) -> None:
    src_dataset_dirs = [x.parent for x in src.glob('**/info.json')]
    src_dataset_dirs.sort()

    with tqdm(total=len(src_dataset_dirs)) as pbar:
        for src_dataset_dir in src_dataset_dirs:
            pbar.set_description(f'Preprocessing ({src_dataset_dir.relative_to(src)})')
            preprocess_dataset(
                src_dataset_dir=src_dataset_dir,
                dst_dataset_dir=dst / src_dataset_dir.relative_to(src),
            )
            pbar.update()


# ======================================================================================
# Main
# ======================================================================================
def _check_data_dir(path: str | Path, *, force: bool) -> Path:
    path = Path(path)
    if path.exists():
        if force:
            logger.warning(f'Removing the existing directory: {path}')
            shutil.rmtree(path)
    return path


def main(
    path: str | Path,
    *,
    independent: bool,
    tabred: bool,
    tabred_official_dir: None | str | Path,
    tabarena: bool,
    tabarena_pause_duration: None | float,
    debug: bool,
    force: bool,
) -> None:
    # Check the directories in the inverse order to better handle force=True.
    preprocessed_data_dir = _check_data_dir(path, force=force)
    raw_data_dir = _check_data_dir(f'{path}-raw', force=force)

    # >>> Prepare raw data for the reference. At this stage, functions preparing
    #     datasets may contain dataset-specific code to handle dataset-specific quirks.

    if raw_data_dir.exists():
        logger.info('Resuming')
    else:
        raw_data_dir.mkdir(parents=True)

    if tabred_official_dir is not None:
        tabred_official_dir = Path(tabred_official_dir)

    if debug:
        assert tabred_official_dir is not None
        prepare_dataset(raw_data_dir / 'a', _make_california_housing)
        prepare_dataset(
            raw_data_dir / 'b',
            _make_tabred_dataset,
            tabred_dataset_dir=tabred_official_dir / 'sberbank-housing',
        )
        prepare_dataset(raw_data_dir / 'c', _make_tabarena_dataset, task_id=363678)

    else:
        if independent:
            prepare_independent_datasets(raw_data_dir)

        if tabred:
            assert tabred_official_dir is not None, (
                'To prepare TabReD, the path to the official TabReD data directory'
                ' must be provided'
            )
            prepare_tabred_benchmark(
                raw_data_dir / _TABRED_BENCHMARK_DIRNAME,
                tabred_official_dir=tabred_official_dir,
            )

        if tabarena:
            prepare_tabarena_benchmark(
                raw_data_dir / _TABARENA_BENCHMARK_DIRNAME,
                pause_duration=tabarena_pause_duration,
            )

    # >>> Prepare data for everyday use by preprocessing the raw data.
    #     This stage is dataset-agnostic.
    print()
    preprocessed_data_dir.mkdir(exist_ok=True)
    preprocess_all_datasets(src=raw_data_dir, dst=preprocessed_data_dir)


if __name__ == '__main__':
    configure_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path',
        default='data',
        help=(
            'Where to put the preprocessed data.'
            ' The "raw" data will be placed next to the preprocessed data'
            ' with the "-raw" suffix.'
        ),
    )
    parser.add_argument(
        '--independent', action='store_true', help='Prepare independent datasets.'
    )
    parser.add_argument(
        '--tabred', action='store_true', help='Prepare the TabReD benchmark.'
    )
    parser.add_argument(
        '--tabred-official-dir', help='The path to the official TabReD directory.'
    )
    parser.add_argument(
        '--tabarena', action='store_true', help='Prepare the TabArena benchmark.'
    )
    parser.add_argument(
        '--tabarena-pause-duration',
        type=float,
        help=(
            'The pause (in seconds) between dataset downloads'
            ' to avoid the OpenML rate limit'
        ),
    )
    parser.add_argument('--debug', action='store_true')
    parser.add_argument(
        '--force',
        action='store_true',
        help='Remove existing data directories on the start.',
    )
    args = parser.parse_args()

    main(**vars(args))
