"""Create the TabPack configs for all datasets — both plain and cosine variants.

Usage:

    uv run bin/tabpack/make_configs_new.py

Creates two sets of experiment configs:

* ``exp/tabpack/0/<dataset>/main/config.toml``
  Plain TabPack — no numerical embeddings (MuonAdamW pack optimizer only).

* ``exp/tabpack-cosine/0/<dataset>/main/config.toml``
  TabPack-cosine — cosine embeddings + MuonAdamW pack optimizer
  (the reference variant from the paper).

Commands for running each set are saved to:
  <local dir>/commands/tabpack.sh
  <local dir>/commands/tabpack-cosine.sh

See ``bin/tabpack/run.py`` for how to execute them.
"""

from pathlib import Path
from typing import Any

import dev.lib.config as devconfig
import dev.lib.datasets as devdatasets
import lib.env
import lib.experiment
import lib.util

lib.util.init(torch_=False)

N_SEEDS = 5

# The datasets used for most of the experiments in the TabPack paper.
MAIN_DATASETS = [
    devdatasets.CHURN,
    devdatasets.CALIFORNIA,
    devdatasets.HOUSE,
    devdatasets.ADULT,
    devdatasets.DIAMOND,
    devdatasets.OTTO,
    devdatasets.HIGGS_SMALL,
    devdatasets.BLACK_FRIDAY,
    devdatasets.MICROSOFT,
    *devdatasets.TABRED_DATASETS,
]

# ── shared base ────────────────────────────────────────────────────────────────

def _base_config(dataset: str, *, seed: int = 0) -> dict[str, Any]:
    """Fields common to both plain and cosine variants."""
    return {
        'seed': seed,
        'data': devconfig.make_data_config(dataset, cache=True),
        'n_models': 64,
        'optimizer': {'type': 'MuonAdamWPack', 'shared_step': True},
        'batch_size': devconfig.get_nn_batch_size(dataset),
        'n_epochs': -1,
        'patience': 16,
        'online_ensembles': {
            'greedy': {
                'type': 'greedy',
                'update_type': 'latest',
                'include_current_ensemble_in_pool': True,
                'patience': 32,
                'options': {'max_ensemble_size': 32},
            }
        },
        'amp_dtype': 'bfloat16',
        'save_all_predictions': True,
        'track_online_ensemble_history': True,
        'track_experiments': True,
    }


# ── variant: plain (no embeddings) ────────────────────────────────────────────

def make_config_plain(dataset: str, *, seed: int = 0) -> dict[str, Any]:
    """TabPack without numerical embeddings.

    Corresponds to the ``tabpack`` ablation in the paper
    (``tabpack/experiments/tabpack/make.py`` in the original repo).
    """
    config = _base_config(dataset, seed=seed)
    config['model'] = {
        'activation': 'ReLU',
        'd_block': 384,
    }
    config['sampler'] = {
        'type': 'RandomSampler',
        'space': {
            'model': {
                'n_blocks': ['_tune_', 'int', 1, 3],
                'dropout': ['_tune_', '?uniform', 0.0, 0.0, 0.5],
            },
            'optimizer': {
                'lr': ['_tune_', 'loguniform', 0.0001, 0.005],
                'weight_decay': ['_tune_', 'loguniform', 0.001, 1.0],
                'muon_lr': ['_tune_', 'loguniform', 0.001, 0.1],
            },
        },
    }
    return config


# ── variant: cosine embeddings ─────────────────────────────────────────────────

def make_config_cosine(dataset: str, *, seed: int = 0) -> dict[str, Any]:
    """TabPack with CosineEmbeddingsPack (reference variant from the paper).

    Identical to the config produced by the original ``make_configs.py``.
    """
    config = _base_config(dataset, seed=seed)
    config['model'] = {
        'num_embeddings': {'type': 'CosineEmbeddingsPack'},
        'activation': 'ReLU',
        'd_block': 384,
    }
    config['sampler'] = {
        'type': 'RandomSampler',
        'space': {
            'model': {
                'num_embeddings': {
                    'd_embedding': [
                        '_tune_',
                        'int',
                        8,
                        (
                            20
                            if dataset in (devdatasets.TABRED_MAPS_ROUTING,)
                            else 32
                        ),
                        4,
                    ],
                    'init_scale': ['_tune_', 'loguniform', 0.01, 10.0],
                },
                'n_blocks': ['_tune_', 'int', 1, 3],
                'dropout': ['_tune_', '?uniform', 0.0, 0.0, 0.5],
            },
            'optimizer': {
                'lr': ['_tune_', 'loguniform', 0.0001, 0.005],
                'weight_decay': ['_tune_', 'loguniform', 0.001, 1.0],
                'muon_lr': ['_tune_', 'loguniform', 0.001, 0.1],
            },
        },
    }
    return config


# ── config generation ──────────────────────────────────────────────────────────

VARIANTS: list[tuple[str, str, Any]] = [
    # (exp_template,              commands_filename,    make_config_fn)
    ('exp/tabpack/0/{}',         'tabpack.sh',         make_config_plain),
    ('exp/tabpack-cosine/0/{}',  'tabpack-cosine.sh',  make_config_cosine),
]

commands_dir = lib.env.get_local_dir() / 'commands'
commands_dir.mkdir(parents=True, exist_ok=True)

for exp_template, commands_filename, make_config_fn in VARIANTS:
    commands = []

    for dataset in MAIN_DATASETS:
        exp_prefix = Path(exp_template.format(dataset))
        main_exp = exp_prefix / 'main'
        evaluation_exp = exp_prefix / 'evaluation'

        if lib.experiment.is_experiment(evaluation_exp) and lib.experiment.is_done(
            evaluation_exp
        ):
            # Skip fully completed experiments.
            continue

        # Do not rewrite configs of in-progress experiments.
        if not lib.experiment.is_experiment(main_exp) or lib.experiment.is_fresh(main_exp):
            lib.experiment.create(
                main_exp,
                config=make_config_fn(dataset),
                parents=True,
                force=True,
            )

        # Save the command to run the experiment.
        commands.append(
            f'uv run bin/tabpack/run.py {evaluation_exp} --n-seeds {N_SEEDS} --clean'
        )

    commands.append('')
    commands_dir.joinpath(commands_filename).write_text('\n'.join(commands))

# ── post-run hints ─────────────────────────────────────────────────────────────

print("""\
Configs written. To add them to Git, run:

git add 'exp/tabpack/**/config.toml'
git add 'exp/tabpack-cosine/**/config.toml'
git commit -m 'Add TabPack configs (plain + cosine variants)'
""")
