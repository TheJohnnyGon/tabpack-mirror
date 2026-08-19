#!/usr/bin/env python3
"""Analyze predictions from pair logit training."""

import argparse
import json
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Analyze pair logit predictions')
    parser.add_argument('exp_dir', type=Path, help='Experiment directory')
    args = parser.parse_args()
    
    exp_dir = args.exp_dir
    
    # Load predictions
    predictions_path = exp_dir / 'predictions.npz'
    if not predictions_path.exists():
        print(f'ERROR: {predictions_path} not found')
        return
    
    predictions = np.load(predictions_path)
    print('=== Predictions ===')
    for key in predictions.keys():
        arr = predictions[key]
        print(f'{key}: shape={arr.shape}, dtype={arr.dtype}')
        print(f'  min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}, std={arr.std():.4f}')
    print()
    
    # Load report
    report_path = exp_dir / 'report.json'
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)
        
        print('=== Report ===')
        print(f'prediction_type: {report.get("prediction_type")}')
        print(f'n_models: {report.get("n_models")}')
        print(f'step: {report.get("step")}')
        print(f'time: {report.get("time"):.1f}s')
        print()
        
        if 'pair_accuracy' in report:
            print('=== Pair Accuracy ===')
            for part, acc in report['pair_accuracy'].items():
                print(f'{part}: {acc:.4f}')
            print()
        
        if 'best' in report:
            best = report['best']
            print('=== Best Model ===')
            print(f'id: {best.get("id")}')
            print(f'best_step: {best.get("best_step")}')
            if 'metrics' in best:
                for part, metrics in best['metrics'].items():
                    print(f'{part}: {metrics.get("score", metrics):.4f}')
            print()
        
        if 'experiments' in report:
            print(f'=== Experiments ({len(report["experiments"])}) ===')
            # Show first 5
            for i, exp in enumerate(report['experiments'][:5]):
                metrics = exp.get('report', {}).get('metrics', {})
                val_score = metrics.get('val', {}).get('score', 0)
                test_score = metrics.get('test', {}).get('score', 0)
                print(f'  {i}: val={val_score:.4f}, test={test_score:.4f}')
            if len(report['experiments']) > 5:
                print(f'  ... and {len(report["experiments"]) - 5} more')
            print()
    
    # Load summary
    summary_path = exp_dir / 'summary.txt'
    if summary_path.exists():
        print('=== Summary ===')
        print(summary_path.read_text())


if __name__ == '__main__':
    main()
