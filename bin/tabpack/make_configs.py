"""Create the TabPack configs for all datasets.

Usage:

    uv run bin/tabpack/make_configs.py

For each dataset, the script creates `exp/tabpack/0/<dataset>/main/config.toml`
(the reference config is the "tabpack-cosine" variant from the paper:
cosine embeddings + the Muon-AdamW pack optimizer)
and saves the commands for running the experiments
(see `bin/tabpack/run.py`) to `<local dir>/commands/tabpack.sh`.
"""

from pathlib import Path
from typing import Any

import dev.lib.config as devconfig
import dev.lib.datasets as devdatasets
import lib.env
import lib.experiment
import lib.util

lib.util.init(torch_=False)

EXP_TEMPLATE = 'exp/tabpack/0/{}'
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


def make_config(dataset: str, *, seed: int = 0) -> dict[str, Any]:
    return {
        'seed': seed,
        'data': devconfig.make_data_config(dataset, cache=True),
        'n_models': 64,
        'model': {
            'num_embeddings': {'type': 'CosineEmbeddingsPack'},
            'activation': 'ReLU',
            'd_block': 384,
        },
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
        'sampler': {
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
        },
        'amp_dtype': 'bfloat16',
        'save_all_predictions': True,
        'track_online_ensemble_history': True,
        'track_experiments': True,
    }


commands = []
for dataset in MAIN_DATASETS:
    exp_prefix = Path(EXP_TEMPLATE.format(dataset))
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
            main_exp, config=make_config(dataset), parents=True, force=True
        )

    # Save the command to run the experiment.
    commands.append(
        f'uv run bin/tabpack/run.py {evaluation_exp} --n-seeds {N_SEEDS} --clean'
    )
commands.append('')

commands_dir = lib.env.get_local_dir() / 'commands'
commands_dir.mkdir(parents=True, exist_ok=True)
commands_dir.joinpath('tabpack.sh').write_text('\n'.join(commands))

exp_prefix = EXP_TEMPLATE.split('/{}', 1)[0]
print(f"""\
To add the configs to Git, run:

git add '{exp_prefix}/**/config.toml'
git commit -m 'Add TabPack configs in {exp_prefix}'
""")
