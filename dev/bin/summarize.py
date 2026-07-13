import argparse
import sys
from pathlib import Path

from loguru import logger

import lib.experiment
import lib.util


def main(path: str | Path, *, force: bool = False) -> None:
    path = Path(path)
    if not path.is_dir():
        raise ValueError(f'path must be an existing directory ({path=})')

    experiments = [x.parent for x in path.glob('**/report.json')]
    if not force:
        experiments = [
            x for x in experiments if not lib.experiment.get_summary_path(x).exists()
        ]
    n_experiments = len(experiments)

    count = 0
    print(
        f'Will write {n_experiments} {"summary" if n_experiments == 1 else "summaries"}'
    )
    for exp in experiments:
        prefix = f'[{count + 1}/{n_experiments}]'
        if not lib.experiment.is_done(exp):
            logger.warning(f'{prefix} Skipping an unfinished experiment: {exp!s}')
            continue

        if count == 0:
            print()
        print(f'{prefix} {exp!s}')
        lib.experiment.dump_summary(
            exp, lib.experiment.summarize(lib.experiment.load_report(exp))
        )
        count += 1

    print(f'\nHas written {count}/{n_experiments} summaries')


if __name__ == '__main__':
    lib.util.init()

    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--force', action='store_true')

    sys.exit(main(**vars(parser.parse_args())))
