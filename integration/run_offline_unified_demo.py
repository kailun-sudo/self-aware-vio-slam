#!/usr/bin/env python3
"""
Run the minimum offline unified ORB-SLAM + self-aware SLAM demo.
"""

import argparse
import os
import shutil
import sys
from typing import Optional

import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SELF_AWARE_ROOT = os.path.join(ROOT_DIR, 'self_aware_slam')
sys.path.insert(0, SELF_AWARE_ROOT)

from src.ground_truth_alignment import (  # noqa: E402
    associate_trajectories,
    compute_pose_errors,
    load_euroc_groundtruth,
    load_tum_trajectory,
)
from src.models.inference import run_inference  # noqa: E402


def _load_trajectory(path: str, fmt: str) -> pd.DataFrame:
    if fmt == 'euroc':
        return load_euroc_groundtruth(path)
    if fmt == 'tum':
        return load_tum_trajectory(path)
    raise ValueError(f'Unsupported trajectory format: {fmt}')


def _attach_actual_pose_errors(predictions: pd.DataFrame,
                               pose_errors: pd.DataFrame,
                               max_diff: float) -> pd.DataFrame:
    """Attach nearest actual pose errors to prediction rows by timestamp."""
    if predictions.empty or pose_errors.empty:
        return predictions

    left = predictions.sort_values('timestamp').reset_index(drop=True)
    right = pose_errors[['timestamp', 'pose_error', 'rotation_error_deg']].sort_values('timestamp').reset_index(drop=True)
    merged = pd.merge_asof(
        left,
        right,
        on='timestamp',
        direction='nearest',
        tolerance=max_diff,
    )
    merged = merged.rename(columns={
        'pose_error': 'actual_pose_error',
        'rotation_error_deg': 'actual_rotation_error_deg',
    })
    return merged


def run_demo(metrics_path: str,
             estimated_path: str,
             groundtruth_path: str,
             output_dir: str,
             config_path: Optional[str] = None,
             checkpoint_path: Optional[str] = None,
             dataset_stats_path: Optional[str] = None,
             gt_format: str = 'euroc',
             est_format: str = 'tum',
             max_diff: float = 0.01):
    """Prepare pose errors and run reliability inference."""
    os.makedirs(output_dir, exist_ok=True)

    packaged_metrics = os.path.join(output_dir, 'slam_metrics.csv')
    packaged_estimated = os.path.join(output_dir, 'estimated.txt')
    packaged_groundtruth = os.path.join(output_dir, 'groundtruth.csv')
    pose_errors_path = os.path.join(output_dir, 'pose_errors.csv')
    predictions_path = os.path.join(output_dir, 'reliability_predictions.csv')

    shutil.copyfile(metrics_path, packaged_metrics)
    shutil.copyfile(estimated_path, packaged_estimated)
    shutil.copyfile(groundtruth_path, packaged_groundtruth)

    gt = _load_trajectory(groundtruth_path, gt_format)
    est = _load_trajectory(estimated_path, est_format)
    matched = associate_trajectories(gt, est, max_diff=max_diff)
    if matched.empty:
        raise RuntimeError('No matched trajectory pairs found. Check timestamps and formats.')

    pose_errors = compute_pose_errors(matched, align=True)
    pose_errors.to_csv(pose_errors_path, index=False)

    predictions = run_inference(
        metrics_path=packaged_metrics,
        output_path=predictions_path,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        dataset_path=dataset_stats_path,
        pose_errors_path=pose_errors_path,
    )
    predictions = _attach_actual_pose_errors(predictions, pose_errors, max_diff=max_diff)
    predictions.to_csv(predictions_path, index=False)

    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w') as handle:
        handle.write('Offline unified demo summary\n')
        handle.write(f'matched_poses: {len(matched)}\n')
        handle.write(f'pose_error_mean: {pose_errors["pose_error"].mean():.6f}\n')
        handle.write(f'confidence_mean: {predictions["confidence_score"].mean():.6f}\n')
        handle.write(f'failure_probability_mean: {predictions["failure_probability"].mean():.6f}\n')

    return {
        'metrics_path': packaged_metrics,
        'pose_errors_path': pose_errors_path,
        'predictions_path': predictions_path,
        'summary_path': summary_path,
    }


def main():
    parser = argparse.ArgumentParser(description='Run offline unified ORB-SLAM self-awareness demo')
    parser.add_argument('--metrics', type=str, required=True, help='Path to exported slam_metrics.csv')
    parser.add_argument('--estimated', type=str, required=True, help='Path to exported estimated TUM trajectory')
    parser.add_argument('--groundtruth', type=str, required=True, help='Path to ground-truth trajectory file')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory for packaged outputs')
    parser.add_argument('--config', type=str, default=None, help='Path to self-aware config.yaml')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to trained model checkpoint')
    parser.add_argument('--dataset-stats', type=str, default=None,
                        help='Path to train_dataset.pkl with normalization stats')
    parser.add_argument('--gt-format', type=str, choices=['euroc', 'tum'], default='euroc')
    parser.add_argument('--est-format', type=str, choices=['euroc', 'tum'], default='tum')
    parser.add_argument('--max-diff', type=float, default=0.01,
                        help='Max timestamp difference for trajectory association (seconds)')
    args = parser.parse_args()

    outputs = run_demo(
        metrics_path=args.metrics,
        estimated_path=args.estimated,
        groundtruth_path=args.groundtruth,
        output_dir=args.output_dir,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        dataset_stats_path=args.dataset_stats,
        gt_format=args.gt_format,
        est_format=args.est_format,
        max_diff=args.max_diff,
    )

    for key, value in outputs.items():
        print(f'{key}: {value}')


if __name__ == '__main__':
    main()
