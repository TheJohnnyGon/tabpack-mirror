"""Create the evaluation config from a finished TabPack `main` experiment.

Usage:

    uv run bin/tabpack/make_evaluation_config.py exp/tabpack/0/<dataset>/main \
        [--ensemble NAME] [--n-seeds N] [--run] [--force]

This script reproduces the config-creation step that used to live inside
`bin/tabpack/run.py` (see `_evaluate_ensemble`). Given a *finished* `main`
experiment, it:

* resolves which online ensemble to evaluate (like `run.py:main` did),
* derives the `evaluation` config from the main run's report and
  `experiments.json`,
* writes it as `config.toml` into the sibling `<prefix>/evaluation` directory.

By default the evaluation itself is NOT executed. Pass `--run` to additionally
launch `bin.evaluate.main` (equivalent to the original pipeline).
"""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import bin.evaluate
import lib.experiment
import lib.util


def _check_source_exp(source_exp: str | Path) -> Path:
    assert lib.experiment.is_experiment(source_exp)
    assert lib.experiment.is_done(source_exp)
    return Path(source_exp)


def _resolve_ensemble(source_exp: Path, ensemble: None | str) -> str:
    """Resolve which online ensemble to evaluate (see `run.py:main`)."""
    source_config = lib.experiment.load_config(source_exp)
    online_ensemble_names = list(source_config.get('online_ensembles', {}))
    if ensemble is None:
        assert len(online_ensemble_names) == 1, (
            'The main config defines multiple online ensembles:'
            f' {online_ensemble_names}. Choose one with the --ensemble option.'
        )
        return online_ensemble_names[0]
    assert ensemble in online_ensemble_names
    return ensemble


def make_evaluation_config(
    source_exp: str | Path,
    *,
    name: str,
    n_seeds: int,
) -> dict:
    """Build the evaluation config from the main run's report and configs.

    This mirrors the config-building part of `run.py:_evaluate_ensemble`.
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

    config = {
        'function': source_report['function'],
        'n_seeds': n_seeds,
        'base_config': base_config,
    }
    return config


def main(
    source_exp: str | Path,
    *,
    n_seeds: int = 5,
    ensemble: None | str = None,
    run: bool = False,
    force: bool = False,
) -> None:
    # Accept either the `main` experiment or its prefix (`.../<dataset>`).
    source_exp = Path(source_exp)
    if source_exp.name == 'evaluation':
        raise ValueError(
            'Pass the path to the `main` experiment (or its prefix),'
            ' not the `evaluation` directory.'
        )
    main_exp = source_exp if source_exp.name == 'main' else source_exp / 'main'
    main_exp = _check_source_exp(main_exp)

    name = _resolve_ensemble(main_exp, ensemble)

    config = make_evaluation_config(main_exp, name=name, n_seeds=n_seeds)

    exp = main_exp.parent / 'evaluation'
    if lib.experiment.get_config_path(exp).exists():
        assert lib.experiment.load_config(exp) == config
    else:
        lib.experiment.create(exp, config=config, parents=True, force=True)

    if run:
        lib.experiment.run(bin.evaluate.main, None, exp, force=force, resume=True)


if __name__ == '__main__':
    lib.util.init()

    parser = argparse.ArgumentParser()
    parser.add_argument('source_exp')
    parser.add_argument('--n-seeds', type=int, default=5)
    parser.add_argument('--ensemble')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--force', action='store_true')

    main(**vars(parser.parse_args()))
