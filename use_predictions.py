#!/usr/bin/env python3
"""Use predictions from pair logit training for LTR."""

import argparse
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Use pair logit predictions')
    parser.add_argument('exp_dir', type=Path, help='Experiment directory')
    parser.add_argument('--part', default='test', help='Part to use (train/val/test)')
    parser.add_argument('--ensemble', action='store_true', help='Average predictions across models')
    args = parser.parse_args()
    
    # Load predictions
    predictions_path = args.exp_dir / 'predictions.npz'
    predictions = np.load(predictions_path)
    
    preds = predictions[args.part]  # shape: (n_models, n_samples)
    print(f'Loaded {args.part} predictions: {preds.shape}')
    
    if args.ensemble:
        # Average across models
        preds_ensemble = preds.mean(axis=0)  # shape: (n_samples,)
        print(f'Ensembled predictions: {preds_ensemble.shape}')
        print(f'  min={preds_ensemble.min():.4f}, max={preds_ensemble.max():.4f}')
        print(f'  mean={preds_ensemble.mean():.4f}, std={preds_ensemble.std():.4f}')
        
        # Save ensembled predictions
        output_path = args.exp_dir / f'predictions_{args.part}_ensemble.npy'
        np.save(output_path, preds_ensemble)
        print(f'Saved to {output_path}')
    else:
        # Save all predictions
        output_path = args.exp_dir / f'predictions_{args.part}_all.npy'
        np.save(output_path, preds)
        print(f'Saved to {output_path}')


if __name__ == '__main__':
    main()
