"""Inference script for TabPack models saved in model.pt.

Evaluates the full ensemble (all saved models) with averaged predictions.

Usage:
    uv run python infer/infer.py [--device DEVICE] [--parts val test]
    uv run python infer/infer.py --compare-nirvana  # compare with training metrics
"""

import argparse
import json
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


def print_nirvana_comparison(model_path: str | Path, parts: list[str]) -> None:
    """Read and print metrics from report.json for comparison with inference results.
    
    Args:
        model_path: Path to model.pt file
        parts: List of parts to show metrics for (e.g., ['val', 'test'])
    """
    model_dir = Path(model_path).parent
    report_path = model_dir / 'report.json'
    
    if not report_path.exists():
        print(f'\n=== Nirvana Results ===')
        print(f'  report.json not found at {report_path}')
        return
    
    print(f'\n=== Nirvana Results ===')
    
    # Load report.json
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    # Extract ensemble metrics
    metrics_data = {}
    if 'online_ensembles' in report:
        for ens_name, ens_data in report['online_ensembles'].items():
            if 'report' in ens_data and 'metrics' in ens_data['report']:
                metrics_data = ens_data['report']['metrics']
                break
    
    # Print extracted metrics
    for part in parts:
        if part in metrics_data:
            print(f'  [{part}]')
            metrics = metrics_data[part]
            if 'accuracy' in metrics:
                print(f'    accuracy:  {metrics["accuracy"]:.4f}')
            if 'roc-auc' in metrics:
                print(f'    roc-auc:   {metrics["roc-auc"]:.4f}')
            if 'macro avg' in metrics and 'f1-score' in metrics['macro avg']:
                print(f'    f1-macro:  {metrics["macro avg"]["f1-score"]:.4f}')
            if 'weighted avg' in metrics and 'f1-score' in metrics['weighted avg']:
                print(f'    f1-micro:  {metrics["weighted avg"]["f1-score"]:.4f}')
        else:
            print(f'  [{part}] - no metrics found')


def load_model_artifact(path: str | Path) -> dict:
    """Load the model.pt artifact."""
    return torch.load(path, map_location='cpu', weights_only=False)


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
    model_keys: list[int | tuple[int, int]],
) -> None:
    """Load all saved model weights into the pack.
    
    Args:
        model_keys: List of keys to look up in state_dicts.
                   Can be int (legacy format, model_id only) or
                   tuple[int, int] (new format, (model_id, step)).
    """
    # Each state_dict has pack dim=1, so state_dict_idx is always [0]
    state_dict_idx = torch.tensor([0])
    for pack_idx, key in enumerate(model_keys):
        state_dict = artifact['state_dicts'][key]
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
    weights: list[float] | None = None,
) -> dict:
    """Evaluate the ensemble (all pack members) and average predictions.
    
    Args:
        weights: Optional list of weights for each model in the ensemble.
                 If None, simple averaging is used.
    """
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
        eval_result = _evaluate(
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
        # _evaluate returns tuple (result, batch_size) due to decorator
        if isinstance(eval_result, tuple):
            result = eval_result[0]
        else:
            result = eval_result

    # Average predictions across pack members (ensemble)
    ensemble_predictions = {}
    ensemble_metrics = {}
    for part in parts:
        # result.predictions[part] has shape (pack_size, n_samples, ...)
        if weights is not None:
            # Weighted average
            weights_array = np.array(weights)
            weights_normalized = weights_array / weights_array.sum()
            # Reshape weights for broadcasting: (pack_size, 1, ...)
            weights_shape = (pack_size,) + (1,) * (result.predictions[part].ndim - 1)
            avg_pred = (result.predictions[part] * weights_normalized.reshape(weights_shape)).sum(axis=0)
        else:
            # Simple average
            avg_pred = result.predictions[part].mean(axis=0)
        
        ensemble_predictions[part] = avg_pred

        # Calculate metrics for the averaged prediction using lib.metrics
        # which provides full classification_report (f1, precision, recall, roc-auc)
        import lib.metrics
        ensemble_metrics[part] = lib.metrics.calculate_metrics(
            y_true=dataset.task.labels[part],
            y_pred=avg_pred,
            task_type=dataset.task.type_,
            prediction_type=prediction_type,
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
    parser.add_argument(
        '--compare-nirvana',
        action='store_true',
        help='Compare inference results with training metrics from summary.txt',
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
    
    # Detect state_dicts key format: legacy (int) vs new (tuple[int, int])
    state_dict_keys = list(artifact['state_dicts'].keys())
    if state_dict_keys and isinstance(state_dict_keys[0], tuple):
        key_format = 'new'  # (model_id, step)
        print(f'  state_dicts format: new (model_id, step) tuples')
    else:
        key_format = 'legacy'  # int model_id
        print(f'  state_dicts format: legacy (model_id only)')
    
    # Get unique model IDs from state_dicts
    if key_format == 'new':
        unique_model_ids = sorted({k[0] for k in state_dict_keys})
    else:
        unique_model_ids = sorted(state_dict_keys)
    
    print(f'  n_models (total): {artifact["n_models"]}')
    print(f'  n_models (saved): {len(unique_model_ids)}')
    print(f'  n_checkpoints (saved): {len(state_dict_keys)}')
    print(f'  n_num_features: {artifact["n_num_features"]}')
    print(f'  n_cat_features: {len(artifact["cat_cardinalities"])}')
    print(f'  n_classes: {artifact["n_classes"]}')
    print(f'  prediction_type: {artifact["prediction_type"]}')
    
    # Check for ensemble info
    ensemble_info = artifact.get('ensemble')
    ensemble_weights = None
    model_keys: list[int | tuple[int, int]] = []
    
    if ensemble_info and ensemble_info.get('ids'):
        ensemble_ids = ensemble_info['ids']
        ensemble_steps = ensemble_info.get('steps', [])
        ensemble_weights = ensemble_info.get('weights')
        print(f'  ensemble: greedy with {len(ensemble_ids)} models')
        print(f'    IDs: {ensemble_ids}')
        print(f'    Steps: {ensemble_steps}')
        if ensemble_weights:
            print(f'    Weights: {ensemble_weights}')
        
        # Build model_keys based on format
        if key_format == 'new':
            # New format: use (id, step) tuples
            for idx, eid in enumerate(ensemble_ids):
                estep = ensemble_steps[idx] if idx < len(ensemble_steps) else None
                key = (eid, estep) if estep is not None else None
                if key is not None and key in artifact['state_dicts']:
                    model_keys.append(key)
                else:
                    print(f'  WARNING: checkpoint not found for model {eid} at step {estep}')
        else:
            # Legacy format: use model_id only
            available_ensemble_ids = [mid for mid in ensemble_ids if mid in artifact['state_dicts']]
            if len(available_ensemble_ids) < len(ensemble_ids):
                missing = set(ensemble_ids) - set(available_ensemble_ids)
                print(f'  WARNING: {len(missing)} ensemble models not in state_dicts: {missing}')
                print(f'  Using {len(available_ensemble_ids)} available ensemble models')
            model_keys = available_ensemble_ids
            # Filter weights if they exist
            if ensemble_weights:
                ensemble_weights = [
                    w for mid, w in zip(ensemble_ids, ensemble_weights)
                    if mid in artifact['state_dicts']
                ]
    else:
        print('  ensemble: None (using all saved models with simple averaging)')
        ensemble_weights = None
        model_keys = state_dict_keys if key_format == 'new' else unique_model_ids

    # Build dataset
    print(f'\nBuilding dataset...')
    preprocessor = artifact.get('preprocessor')
    if preprocessor is not None:
        print('Using saved DataPreprocessor from model.pt')
        data_config = artifact['data_config']
        dataset = lib.data.Dataset.from_dir(
            Path(data_config['path']).resolve(),
            data_config.get('split_id', lib.data.DEFAULT_SPLIT_ID)
        )
        dataset = preprocessor.transform(dataset)
    else:
        # Fallback to original build_dataset
        print('WARNING: No preprocessor found, using build_dataset (legacy)')
        dataset = lib.data.build_dataset(**artifact['data_config'])
    
    print(f'  train: {dataset.size("train")}, val: {dataset.size("val")}, test: {dataset.size("test")}')
    print(f'  n_num_features: {dataset.n_num_features}')
    print(f'  n_cat_features: {dataset.n_cat_features}')
    print(f'  n_bin_features: {dataset.n_bin_features}')

    # Extract model_ids from model_keys for build_model_for_ensemble
    if key_format == 'new':
        model_ids_for_build = [k[0] for k in model_keys]
    else:
        model_ids_for_build = model_keys
    
    # Build model for ensemble
    print(f'\nBuilding model (pack_size={len(model_keys)})...')
    model = build_model_for_ensemble(artifact, model_ids_for_build)

    # Load all weights
    print('Loading weights...')
    load_all_weights(model, artifact, model_keys)

    # Evaluate ensemble
    print(f'\nEvaluating ensemble on {args.parts}...')
    result = evaluate_ensemble(
        model,
        dataset,
        artifact,
        device,
        parts=args.parts,
        batch_size=args.batch_size,
        weights=ensemble_weights,
    )

    # Print results
    print('\n=== Ensemble Results ===')
    for part in args.parts:
        metrics = result['metrics'][part]
        print(f'  [{part}]')
        # Show key metrics: accuracy, roc-auc, f1-macro, f1-micro
        if 'accuracy' in metrics:
            acc = metrics['accuracy']
            print(f'    accuracy:  {acc:.4f}' if isinstance(acc, float) else f'    accuracy:  {acc[0]:.4f}')
        if 'roc-auc' in metrics:
            roc = metrics['roc-auc']
            print(f'    roc-auc:   {roc:.4f}' if isinstance(roc, float) else f'    roc-auc:   {roc[0]:.4f}')
        if 'macro avg' in metrics and 'f1-score' in metrics['macro avg']:
            f1 = metrics['macro avg']['f1-score']
            print(f'    f1-macro:  {f1:.4f}' if isinstance(f1, float) else f'    f1-macro:  {f1[0]:.4f}')
        if 'weighted avg' in metrics and 'f1-score' in metrics['weighted avg']:
            f1_micro = metrics['weighted avg']['f1-score']
            print(f'    f1-micro:  {f1_micro:.4f}' if isinstance(f1_micro, float) else f'    f1-micro:  {f1_micro[0]:.4f}')

    # Compare with Nirvana results if requested
    if args.compare_nirvana:
        print_nirvana_comparison(args.model_path, args.parts)

    return result


if __name__ == '__main__':
    main()
