"""Run the full TabPack pipeline for one dataset.

Usage:

    uv run bin/tabpack/run.py exp/tabpack/0/<dataset>/evaluation [--n-seeds N]

The pipeline consists of two experiments located next to each other:

* `<prefix>/main` trains the pack (with sampled hyperparameters) and selects
  the online ensemble. Its config must already exist
  (see `bin/tabpack/make_configs.py`).
* `<prefix>/evaluation` retrains only the selected ensemble members over
  `--n-seeds` random seeds. Its config is derived from the main run's report.
"""

import argparse
import json
import shutil
from copy import deepcopy
from pathlib import Path

from loguru import logger

import bin.evaluate
import bin.tabpack.tabpack
import lib.env
import lib.experiment
import lib.util


def _check_source_exp(source_exp: str | Path) -> Path:
    assert lib.experiment.is_experiment(source_exp)
    assert lib.experiment.is_done(source_exp)
    return Path(source_exp)


def _evaluate_ensemble(
    source_exp: str | Path,
    *,
    name: str,
    n_seeds: int,
    force: bool = False,
) -> None:
    """
    Retrieve ids from the final online ensemble (from the main exp),
    and rerun the method only with the selected configs.
    """

    assert n_seeds > 0

    source_exp = _check_source_exp(source_exp)
    source_config = lib.experiment.load_config(source_exp)
    source_report = lib.experiment.load_report(source_exp)

    ensemble_report = source_report['online_ensembles'][name]['report']
    ensemble_unique_ids = sorted(set(ensemble_report['ids']))

    pack_experiments = json.loads(source_exp.joinpath('experiments.json').read_text())
    # NOTE: At this moment, experiments.json and report['experiments'] are not identical
    # * experiments.json contains all configs (with maybe reports) in sorted way
    # * report['experiments'] contains only stopped (in order of stopping) models

    # Adjust the config for the ensemble evaluation.
    base_config = deepcopy(source_config)
    base_config['n_models'] = len(ensemble_unique_ids)
    base_config['configs'] = [
        pack_experiments[x]['config'] for x in ensemble_unique_ids
    ]
    if base_config['optimizer']['type'] in ('AdamWPack', 'MuonAdamWPack'):
        base_config['optimizer']['shared_step'] = True

    # Remove irrelevant config fields.
    for key in list(source_config['online_ensembles'].keys()):
        if key != name:
            del base_config['online_ensembles'][key]
    for key in (
        'seed',
        'pack_size',
        'share_training_batches',
        'share_training_batch_sequence',
        'object_bagging',
        'sampler',
    ):
        base_config.pop(key, None)
        del key

    exp = source_exp.parent / 'evaluation'
    config = {
        'function': source_report['function'],
        'n_seeds': n_seeds,
        'base_config': base_config,
    }
    if lib.experiment.get_config_path(exp).exists():
        assert lib.experiment.load_config(exp) == config
    else:
        lib.experiment.create(exp, config=config, parents=True, force=True)
    lib.experiment.run(bin.evaluate.main, None, exp, force=force, resume=True)


def main(
    exp: str | Path,
    *,
    n_seeds: int = 5,
    ensemble: None | str = None,
    clean: bool = False,
    force: bool = False,
) -> None:
    # The experiment prefix (`exp/tabpack/<version>/<dataset>`) and the
    # `evaluation` experiment within it are both accepted.
    exp = Path(exp)
    exp_prefix = exp.parent if exp.name == 'evaluation' else exp
    main_exp = exp_prefix / 'main'
    assert lib.experiment.is_experiment(main_exp)

    # Resolve the online ensemble to evaluate.
    main_config = lib.experiment.load_config(main_exp)
    online_ensemble_names = list(main_config.get('online_ensembles', {}))
    if ensemble is None:
        assert len(online_ensemble_names) == 1, (
            'The main config defines multiple online ensembles:'
            f' {online_ensemble_names}. Choose one with the --ensemble option.'
        )
        ensemble = online_ensemble_names[0]
    else:
        assert ensemble in online_ensemble_names

    # If the main experiment is not done,
    # then all secondary experiments are invalid and must be removed.
    if not lib.experiment.is_done(main_exp) or force:
        for path in exp_prefix.iterdir():
            if path.is_dir() and path.name != main_exp.name:
                logger.warning(f'Removing {path}')
                shutil.rmtree(path)
            del path

    # Run the main experiment.
    lib.experiment.run(bin.tabpack.tabpack.main, None, main_exp, force=force)

    # Run the evaluation.
    _evaluate_ensemble(main_exp, name=ensemble, n_seeds=n_seeds, force=force)

    # Finish.
    if clean:
        for directory in [
            lib.env.get_project_dir(),
            lib.env.get_snapshot_dir(),
            lib.env.get_tmp_output_dir(),
        ]:
            if directory is None:
                continue
            for path in directory.joinpath(
                exp_prefix.resolve().relative_to(lib.env.get_project_dir())
            ).rglob('*'):
                if not path.is_dir() and path.suffix in ('.pt', '.npz'):
                    path.unlink()


if __name__ == '__main__':
    lib.util.init()

    parser = argparse.ArgumentParser()
    parser.add_argument('exp')
    parser.add_argument('--n-seeds', type=int, default=5)
    parser.add_argument('--ensemble')
    parser.add_argument('--clean', action='store_true')
    parser.add_argument('--force', action='store_true')

    main(**vars(parser.parse_args()))
