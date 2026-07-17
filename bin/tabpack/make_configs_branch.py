"""Create one or several TabPack `main` configs for a single dataset.

Usage:

    # Reproduce the default peru config (plain variant, d_block=384).
    uv run bin/tabpack/make_configs_branch.py peru

    # Branch over several d_block values, save the trained weights, and also
    # generate the cosine variant.
    uv run bin/tabpack/make_configs_branch.py peru \
        --variant plain cosine \
        --d-block 256 384 512 \
        --save-model

This is a lightweight, self-contained alternative to
`bin/tabpack/make_configs_new.py` for a *single* dataset with explicit
branching. Unlike `make_configs_new.py`, it does not depend on
`dev.lib.datasets`, so it also works for datasets that are referenced only by
their data path (such as `peru`).

The Cartesian product of `--variant` and `--d-block` defines the branches.
Each branch is written to its own experiment directory:

    <exp-root>/<dataset>[__<variant>][__d<d_block>]/main/config.toml

The suffixes are omitted when there is only a single value for the
corresponding option, so a single plain config with the default `d_block`
lands at the canonical `<exp-root>/<dataset>/main/config.toml`.
"""

import argparse
import itertools
from pathlib import Path
from typing import Any

import lib.experiment
import lib.util

DEFAULT_EXP_ROOT = 'exp/tabpack/0'
DEFAULT_D_BLOCK = 384


# ── shared base ────────────────────────────────────────────────────────────────
def _base_config(
    *,
    data_path: str,
    seed: int,
    n_models: int,
    batch_size: int,
    num_policy: None | str,
    save_model: bool,
) -> dict[str, Any]:
    """Fields common to all branches (modeled on the peru config)."""
    data_config: dict[str, Any] = {'path': data_path}
    if num_policy is not None:
        data_config['num_policy'] = num_policy
    data_config['extract_bin_from_num'] = True
    data_config['bin_policy'] = 'convert-to-cat'
    data_config['cache'] = True

    config: dict[str, Any] = {
        'seed': seed,
        'n_models': n_models,
        'batch_size': batch_size,
        'n_epochs': -1,
        'patience': 16,
        'amp_dtype': 'bfloat16',
        'save_all_predictions': True,
        'track_online_ensemble_history': True,
        'track_experiments': True,
        'data': data_config,
        'optimizer': {'type': 'MuonAdamWPack', 'shared_step': True},
        'online_ensembles': {
            'greedy': {
                'type': 'greedy',
                'update_type': 'latest',
                'include_current_ensemble_in_pool': True,
                'patience': 32,
                'options': {'max_ensemble_size': 32},
            }
        },
    }
    if save_model:
        config['save_model'] = True
    return config


def _sampler(*, cosine: bool, d_embedding_max: int) -> dict[str, Any]:
    model_space: dict[str, Any] = {
        'n_blocks': ['_tune_', 'int', 1, 3],
        'dropout': ['_tune_', '?uniform', 0.0, 0.0, 0.5],
    }
    if cosine:
        model_space = {
            'num_embeddings': {
                'd_embedding': ['_tune_', 'int', 8, d_embedding_max, 4],
                'init_scale': ['_tune_', 'loguniform', 0.01, 10.0],
            },
            **model_space,
        }
    return {
        'type': 'RandomSampler',
        'space': {
            'model': model_space,
            'optimizer': {
                'lr': ['_tune_', 'loguniform', 0.0001, 0.005],
                'weight_decay': ['_tune_', 'loguniform', 0.001, 1.0],
                'muon_lr': ['_tune_', 'loguniform', 0.001, 0.1],
            },
        },
    }


def make_config(
    *,
    data_path: str,
    variant: str,
    d_block: int,
    seed: int,
    n_models: int,
    batch_size: int,
    num_policy: None | str,
    d_embedding_max: int,
    save_model: bool,
) -> dict[str, Any]:
    assert variant in ('plain', 'cosine')
    cosine = variant == 'cosine'

    config = _base_config(
        data_path=data_path,
        seed=seed,
        n_models=n_models,
        batch_size=batch_size,
        num_policy=num_policy,
        save_model=save_model,
    )
    model: dict[str, Any] = {'activation': 'ReLU', 'd_block': d_block}
    if cosine:
        model = {'num_embeddings': {'type': 'CosineEmbeddingsPack'}, **model}
    config['model'] = model
    config['sampler'] = _sampler(cosine=cosine, d_embedding_max=d_embedding_max)
    return config


def _branch_dirname(
    dataset: str, *, variant: str, d_block: int, multi_variant: bool, multi_d_block: bool
) -> str:
    name = dataset
    if multi_variant:
        name += f'__{variant}'
    if multi_d_block:
        name += f'__d{d_block}'
    return name


def main(
    dataset: str,
    *,
    data_path: None | str = None,
    exp_root: str = DEFAULT_EXP_ROOT,
    variant: None | list[str] = None,
    d_block: None | list[int] = None,
    seed: int = 0,
    n_models: int = 64,
    batch_size: int = 512,
    num_policy: None | str = 'noisy-quantile',
    d_embedding_max: int = 32,
    save_model: bool = False,
    force: bool = False,
) -> None:
    data_path = data_path if data_path is not None else f'data/{dataset}'
    variants = variant if variant else ['plain']
    d_blocks = d_block if d_block else [DEFAULT_D_BLOCK]

    multi_variant = len(variants) > 1
    multi_d_block = len(d_blocks) > 1

    created = []
    for variant_, d_block_ in itertools.product(variants, d_blocks):
        config = make_config(
            data_path=data_path,
            variant=variant_,
            d_block=d_block_,
            seed=seed,
            n_models=n_models,
            batch_size=batch_size,
            num_policy=num_policy,
            d_embedding_max=d_embedding_max,
            save_model=save_model,
        )
        dirname = _branch_dirname(
            dataset,
            variant=variant_,
            d_block=d_block_,
            multi_variant=multi_variant,
            multi_d_block=multi_d_block,
        )
        main_exp = Path(exp_root) / dirname / 'main'

        if lib.experiment.is_experiment(main_exp) and not (
            lib.experiment.is_fresh(main_exp) or force
        ):
            print(f'Skipping existing experiment: {main_exp}')
            continue

        lib.experiment.create(main_exp, config=config, parents=True, force=True)
        created.append(main_exp)
        print(f'Created {main_exp}')

    if created:
        print('\nRun the created experiments with:')
        for main_exp in created:
            print(f'  uv run bin/tabpack/tabpack.py {main_exp}')


if __name__ == '__main__':
    lib.util.init(torch_=False)

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='Dataset name, e.g. peru')
    parser.add_argument(
        '--data-path', help='Data path (default: data/<dataset>)'
    )
    parser.add_argument('--exp-root', default=DEFAULT_EXP_ROOT)
    parser.add_argument(
        '--variant',
        nargs='+',
        choices=['plain', 'cosine'],
        help='Which variants to generate (default: plain). Multiple allowed.',
    )
    parser.add_argument(
        '--d-block',
        nargs='+',
        type=int,
        help=f'One or more d_block values to branch over (default: {DEFAULT_D_BLOCK}).',
    )
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n-models', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument(
        '--num-policy',
        default='noisy-quantile',
        help="Numerical policy (use 'none' to disable).",
    )
    parser.add_argument('--d-embedding-max', type=int, default=32)
    parser.add_argument(
        '--save-model',
        action='store_true',
        help='Set save_model = true in the config (dumps model.pt for inference).',
    )
    parser.add_argument('--force', action='store_true')

    args = vars(parser.parse_args())
    if args['num_policy'] == 'none':
        args['num_policy'] = None
    main(**args)
