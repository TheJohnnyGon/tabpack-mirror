"""Standalone inference from a TabPack `model.pt` artifact.

Usage:

    uv run bin/tabpack/predict.py exp/tabpack/0/<dataset>/main \
        [--part test] [--output PATH] [--batch-size N] [--ensemble]

`model.pt` is produced by `bin/tabpack/tabpack.py` when `save_model = true`
is set in the experiment config. It bundles the best weights of every finished
pack member together with everything needed to rebuild `ModelPack` and reproduce
the data preprocessing, so inference can be run entirely outside of the training
script.

This script:

* rebuilds the full-size `ModelPack` with the exact resolved model config,
* loads each saved member's best checkpoint into its own pack slot,
* keeps only the finished members and runs a forward pass,
* applies the same post-processing as training (regression de-standardization
  / sigmoid / softmax),
* optionally averages the per-member predictions into a single ensemble
  prediction (the arithmetic mean, matching the aggregation-friendly units in
  which predictions are stored).

By default, the dataset referenced by the artifact's `data_config` is rebuilt
(deterministically, using the stored preprocessing) and the requested `--part`
is used as inference input. This guarantees the preprocessing is identical to
training, because `transform_num` / `transform_cat` are fit on the train split.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from loguru import logger

import bin.tabpack.nn
import lib.data
import lib.util
from bin.tabpack.nn import BATCH_DIM
from bin.tabpack.tabpack import ModelPack


def load_model_artifact(path: str | Path) -> dict:
    """Load a `model.pt` artifact (accepts the file or its experiment dir)."""
    path = Path(path)
    if path.is_dir():
        path = path / 'model.pt'
    if not path.exists():
        raise RuntimeError(
            f'The model artifact does not exist: {path}.'
            ' Make sure the experiment was run with `save_model = true`.'
        )
    return torch.load(path, weights_only=False)


def build_model(artifact: dict, device: torch.device) -> tuple[ModelPack, list[int]]:
    """Rebuild the pack and load the saved members' best weights.

    Returns the model (already restricted to the finished members) and the list
    of member ids in the order of the pack dimension.
    """
    state_dicts: dict[int, dict[str, torch.Tensor]] = artifact['state_dicts']
    member_ids = sorted(state_dicts)
    if not member_ids:
        raise RuntimeError('The artifact does not contain any saved members.')

    # Rebuild the full-size pack with the exact resolved model config, so that
    # each member's slot has the same shapes as during training.
    model = ModelPack(
        n_num_features=artifact['n_num_features'],
        cat_cardinalities=artifact['cat_cardinalities'],
        n_classes=artifact['n_classes'],
        pack_size=artifact['n_models'],
        **artifact['model_config'],
    )
    model.to(device)
    model.eval()

    # Load each saved member's best checkpoint into its own pack slot.
    for member_id in member_ids:
        member_state_dict = {
            name: value.to(device)
            for name, value in state_dicts[member_id].items()
        }
        bin.tabpack.nn.module_pack_load_state_dict(
            model,
            member_state_dict,
            pack_idx=torch.tensor([member_id], device=device),
            state_dict_idx=torch.tensor([0], device=device),
        )

    return model, member_ids


def _postprocess(
    y_pred: torch.Tensor, artifact: dict, *, is_regression: bool, n_classes: None | int
) -> torch.Tensor:
    """Reproduce the post-processing applied in `tabpack._evaluate`."""
    if is_regression:
        stats = artifact['regression_label_stats']
        assert stats is not None
        y_pred = y_pred * stats['std'] + stats['mean']
    elif n_classes is None or n_classes == 2:
        y_pred = torch.special.expit(y_pred)
    else:
        y_pred = torch.special.softmax(y_pred, dim=-1)
    return y_pred


@torch.inference_mode()
def predict(
    artifact: dict,
    x_num: None | torch.Tensor,
    x_cat: None | torch.Tensor,
    *,
    device: torch.device,
    batch_size: int = 32768,
    ensemble: bool = False,
) -> np.ndarray:
    """Run inference and return predictions.

    The output shape is:
    * `(n_members, n_objects[, n_classes])` for per-member predictions, or
    * `(n_objects[, n_classes])` for the averaged ensemble prediction.
    """
    model, member_ids = build_model(artifact, device)
    member_idx = torch.tensor(member_ids, device=device)

    n_classes = artifact['n_classes']
    is_regression = artifact['regression_label_stats'] is not None

    n_objects = (x_num if x_num is not None else x_cat).shape[0]

    with bin.tabpack.nn.module_pack_select(model, member_idx):
        chunks = []
        for batch_idx in torch.arange(n_objects, device=device).split(batch_size):
            y_pred = (
                model(
                    None if x_num is None else x_num[batch_idx],
                    None if x_cat is None else x_cat[batch_idx],
                )
                .squeeze(-1)  # Remove the last dimension for regression predictions.
                .float()
            )
            chunks.append(y_pred)
        y_pred = torch.cat(chunks, dim=BATCH_DIM)

    y_pred = _postprocess(
        y_pred, artifact, is_regression=is_regression, n_classes=n_classes
    )

    predictions = y_pred.cpu().numpy()
    assert np.isfinite(predictions).all()

    if ensemble:
        predictions = predictions.mean(axis=0)

    return predictions


def main(
    exp: str | Path,
    *,
    part: str = 'test',
    output: None | str | Path = None,
    batch_size: int = 32768,
    ensemble: bool = False,
) -> None:
    exp = Path(exp)
    artifact = load_model_artifact(exp)
    device = lib.util.get_device()
    logger.info(f'Device: {device}')

    member_ids = sorted(artifact['state_dicts'])
    logger.info(f'Loaded {len(member_ids)} saved members: {member_ids}')

    # Rebuild the dataset exactly as during training to reproduce the
    # preprocessing (fit on the train split).
    dataset = lib.data.build_dataset(**artifact['data_config'])
    dataset = dataset.to_torch(device)

    x_num = dataset.data['x_num'][part] if 'x_num' in dataset.data else None
    x_cat = dataset.data['x_cat'][part] if 'x_cat' in dataset.data else None

    predictions = predict(
        artifact,
        x_num,
        x_cat,
        device=device,
        batch_size=batch_size,
        ensemble=ensemble,
    )
    logger.info(f'Predictions shape: {predictions.shape}')

    output_path = (
        exp.joinpath(f'predictions_{part}.npy')
        if output is None
        else Path(output)
    )
    if output_path.is_dir():
        output_path = output_path / f'predictions_{part}.npy'
    np.save(output_path, predictions)
    logger.info(f'Saved predictions to {output_path}')


if __name__ == '__main__':
    lib.util.init()

    parser = argparse.ArgumentParser()
    parser.add_argument('exp', help='Experiment dir with model.pt (or path to it)')
    parser.add_argument('--part', default='test')
    parser.add_argument('--output')
    parser.add_argument('--batch-size', type=int, default=32768)
    parser.add_argument(
        '--ensemble',
        action='store_true',
        help='Average per-member predictions into one ensemble prediction',
    )

    main(**vars(parser.parse_args()))
