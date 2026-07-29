"""Inference script for TabPack models saved in model.pt.

Evaluates the full ensemble (all saved models) with averaged predictions.

Usage:
    uv run python infer/infer.py [--device DEVICE] [--parts val test]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure project root is importable
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import lib.data
import lib.util
import bin.tabpack.nn
import bin.tabpack.tabpack
import bin.tabpack.metrics
from bin.tabpack.tabpack import ModelPack, apply_model_impl, _evaluate
from bin.tabpack.nn import BATCH_DIM
from lib.types import PredictionType


def load_model_artifact(path: str | Path) -> dict:
    """Load the model.pt artifact."""
    return torch.load(path, map_location='cpu', weights_only=False)


def build_dataset_with_indices(
    data_config: dict,
    feature_indices: dict,
) -> lib.data.Dataset:
    """Build dataset using stored feature indices for consistent preprocessing.
    
    This function replicates the preprocessing done during training,
    using the stored feature indices to ensure the same features are selected.
    """
    path = Path(data_config['path']).resolve()
    split_id = data_config.get('split_id', ('default',))
    if isinstance(split_id, str):
        split_id = (split_id,)
    
    # Load raw data
    print(f'Loading dataset from {path.name}...')
    dataset = lib.data.Dataset.from_dir(path, split_id)
    
    # Get feature indices
    num_indices = feature_indices.get('num')
    bin_indices = feature_indices.get('bin')
    cat_indices = feature_indices.get('cat')
    
    # Extract binary features from numerical using stored indices
    if 'x_num' in dataset.data and bin_indices is not None:
        print(f'Extracting {len(bin_indices)} binary features using stored indices...')
        x_num = dataset.data['x_num']
        
        # Create binary features from stored indices
        x_bin = {
            k: v[:, bin_indices].astype(lib.data._X_CAT_INT_DTYPE, copy=False)
            for k, v in x_num.items()
        }
        
        # Keep remaining numerical features
        if num_indices is not None and len(num_indices) > 0:
            x_num_remaining = {
                k: v[:, num_indices] for k, v in x_num.items()
            }
            dataset.data['x_num'] = x_num_remaining
        else:
            del dataset.data['x_num']
        
        # Merge with existing x_bin if any
        existing_x_bin = dataset.data.pop('x_bin', None)
        if existing_x_bin is None:
            dataset.data['x_bin'] = x_bin
        else:
            dataset.data['x_bin'] = {
                k: np.concatenate([x_bin[k], existing_x_bin[k]], axis=-1)
                for k in x_bin.keys()
            }
    
    # Apply numerical transformation
    num_policy = data_config.get('num_policy')
    if 'x_num' in dataset.data and num_policy is not None:
        print(f'Transforming numerical features (policy={num_policy})...')
        seed = data_config.get('seed', 0)
        dataset.data['x_num'] = lib.data.transform_num(
            dataset.data['x_num'], num_policy, seed
        )
    
    # Apply binary policy (convert to categorical)
    bin_policy = data_config.get('bin_policy')
    if 'x_bin' in dataset.data and bin_policy == 'convert-to-cat':
        print('Converting binary features to categorical...')
        dataset.convert_bin_features_to_cat_()
    
    # Apply categorical transformation
    cat_policy = data_config.get('cat_policy')
    if 'x_cat' in dataset.data and cat_policy is not None:
        print(f'Transforming categorical features (policy={cat_policy})...')
        dataset.data['x_cat'] = lib.data.transform_cat(dataset.data['x_cat'], cat_policy)
    
    return dataset


def build_model_for_ensemble(
    artifact: dict,
    model_ids: list[int],
) -> ModelPack:
    """Rebuild ModelPack for the ensemble of saved models.

    The model_config contains lists of per-model hyperparameters (n_blocks, dropout)
    for ALL n_models. We need to extract only the entries for the saved model IDs
    and reorder them to match the state_dicts order.
    """
    pack_size = len(model_ids)
    model_config = artifact['model_config']

    # Identify which keys in model_config are per-model lists
    # (e.g. n_blocks, dropout) vs scalar values (e.g. d_block, activation)
    ensemble_model_config = {}
    for key, value in model_config.items():
        if isinstance(value, list) and len(value) == artifact['n_models']:
            # Per-model list: extract entries for saved model IDs
            ensemble_model_config[key] = [value[mid] for mid in model_ids]
        else:
            # Scalar or already-correct list (e.g. max_n_blocks)
            ensemble_model_config[key] = value

    model = ModelPack(
        n_num_features=artifact['n_num_features'],
        cat_cardinalities=artifact['cat_cardinalities'],
        n_classes=artifact['n_classes'],
        pack_size=pack_size,
        **ensemble_model_config,
    )
    return model


def load_all_weights(
    model: ModelPack,
    artifact: dict,
    model_ids: list[int],
) -> None:
    """Load all saved model weights into the pack."""
    # Each state_dict has pack dim=1, so state_dict_idx is always [0]
    state_dict_idx = torch.tensor([0])
    for pack_idx, model_id in enumerate(model_ids):
        state_dict = artifact['state_dicts'][model_id]
        pack_idx_tensor = torch.tensor([pack_idx])
        bin.tabpack.nn.module_pack_load_state_dict(
            model, state_dict, pack_idx=pack_idx_tensor, state_dict_idx=state_dict_idx
        )


def evaluate_ensemble(
    model: ModelPack,
    dataset: lib.data.Dataset,
    artifact: dict,
    device: torch.device,
    parts: list[str] | None = None,
    batch_size: int = 32768,
) -> dict:
    """Evaluate the ensemble (all pack members) and average predictions."""
    if parts is None:
        parts = ['test']

    prediction_type = PredictionType(artifact['prediction_type'])
    regression_label_stats = artifact.get('regression_label_stats')

    # Reconstruct RegressionLabelStats if needed
    reg_stats = None
    if regression_label_stats is not None:
        reg_stats = lib.data.RegressionLabelStats(
            mean=regression_label_stats['mean'],
            std=regression_label_stats['std'],
        )

    # Move dataset to torch
    dataset_torch = dataset.to_torch(device)
    model.to(device)
    model.eval()

    pack_size = model.pack_size

    # Evaluate all pack members
    with torch.inference_mode():
        result = _evaluate(
            apply_model_impl,
            model,
            optimizer=None,  # type: ignore
            dataset=dataset_torch,
            parts=parts,
            regression_label_stats=reg_stats,
            prediction_type=prediction_type,
            batch_size=batch_size,
            device=device,
        )

    # Average predictions across pack members (ensemble)
    ensemble_predictions = {}
    ensemble_metrics = {}
    for part in parts:
        # result.predictions[part] has shape (pack_size, n_samples, ...)
        # Average over pack dimension
        avg_pred = result.predictions[part].mean(axis=0)
        ensemble_predictions[part] = avg_pred

        # Calculate metrics for the averaged prediction
        ensemble_metrics[part] = bin.tabpack.metrics.calculate_metrics_pack(
            y_true=dataset.task.labels[part],
            y_pred=avg_pred[None],  # Add pack dim for the function
            task_type=dataset.task.type_,
            prediction_type=prediction_type,
            score=dataset.task.score,
        )

    return {
        'metrics': ensemble_metrics,
        'predictions': ensemble_predictions,
        'per_model_predictions': result.predictions,
    }


def main():
    parser = argparse.ArgumentParser(description='TabPack ensemble inference')
    parser.add_argument(
        '--model-path',
        type=str,
        default='infer/model.pt',
        help='Path to model.pt',
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (default: auto-detect)',
    )
    parser.add_argument(
        '--parts',
        type=str,
        nargs='+',
        default=['val', 'test'],
        help='Dataset parts to evaluate on',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32768,
        help='Evaluation batch size',
    )
    args = parser.parse_args()

    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = lib.util.get_device()
    print(f'Device: {device}')

    # Load artifact
    print(f'Loading model from {args.model_path}...')
    artifact = load_model_artifact(args.model_path)
    model_ids = sorted(artifact['state_dicts'].keys())
    print(f'  n_models (total): {artifact["n_models"]}')
    print(f'  n_models (saved): {len(model_ids)}')
    print(f'  n_num_features: {artifact["n_num_features"]}')
    print(f'  n_cat_features: {len(artifact["cat_cardinalities"])}')
    print(f'  n_classes: {artifact["n_classes"]}')
    print(f'  prediction_type: {artifact["prediction_type"]}')
    
    # Check for feature indices
    feature_indices = artifact.get('feature_indices', {})
    if feature_indices:
        print(f'  feature_indices:')
        for key, indices in feature_indices.items():
            if indices is not None:
                print(f'    {key}: {len(indices)} features')
            else:
                print(f'    {key}: None')
    else:
        print('  WARNING: No feature_indices in model.pt, using build_dataset directly')

    # Build dataset
    print(f'\nBuilding dataset...')
    if feature_indices and any(v is not None for v in feature_indices.values()):
        dataset = build_dataset_with_indices(artifact['data_config'], feature_indices)
    else:
        # Fallback to original build_dataset
        dataset = lib.data.build_dataset(**artifact['data_config'])
    
    print(f'  train: {dataset.size("train")}, val: {dataset.size("val")}, test: {dataset.size("test")}')
    print(f'  n_num_features: {dataset.n_num_features}')
    print(f'  n_cat_features: {dataset.n_cat_features}')
    print(f'  n_bin_features: {dataset.n_bin_features}')

    # Build model for ensemble
    print(f'\nBuilding model (pack_size={len(model_ids)})...')
    model = build_model_for_ensemble(artifact, model_ids)

    # Load all weights
    print('Loading weights...')
    load_all_weights(model, artifact, model_ids)

    # Evaluate ensemble
    print(f'\nEvaluating ensemble on {args.parts}...')
    result = evaluate_ensemble(
        model,
        dataset,
        artifact,
        device,
        parts=args.parts,
        batch_size=args.batch_size,
    )

    # Print results
    print('\n=== Ensemble Results ===')
    for part in args.parts:
        metrics = result['metrics'][part]
        print(f'  [{part}]')
        for key, values in metrics.items():
            print(f'    {key}: {values[0]:.4f}')

    return result


if __name__ == '__main__':
    main()
