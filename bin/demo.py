import pickle
import time
from pathlib import Path
from typing import NotRequired, TypedDict

import delu
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import lib.experiment
import lib.util
from lib.types import KWArgs


class Config(TypedDict):
    seed: int
    dataset: KWArgs
    model: KWArgs
    test_size: NotRequired[float]


def main(config: Config, exp: str | Path) -> lib.experiment.Report:
    exp = Path(exp)
    report = lib.experiment.create_report(main, add_gpu_info=True)

    delu.random.seed(config['seed'])

    X, y = make_classification(**config['dataset'])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.get('test_size')
    )

    model = LogisticRegression(**config['model'])

    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    report['time'] = time.perf_counter() - start_time

    y_pred = model.predict_proba(X_test)
    report['score'] = accuracy_score(y_test, np.argmax(y_pred, axis=1))

    exp.joinpath('model.pickle').write_bytes(pickle.dumps(model))
    np.save(exp / 'y_pred.npy', y_pred)

    lib.experiment.finish(exp, report)
    return report


if __name__ == '__main__':
    lib.util.init()
    lib.experiment.run_cli(main)
