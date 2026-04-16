#!/usr/bin/env python3
"""
Package current unified outputs into a self-aware training-ready sequence folder.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.ground_truth_alignment import (  # noqa: E402
    associate_trajectories,
    compute_pose_errors,
    load_euroc_groundtruth,
    load_tum_trajectory,
)


def align_metrics_to_pose_errors(metrics: pd.DataFrame,
                                 pose_errors: pd.DataFrame,
                                 max_diff: float) -> pd.DataFrame:
    """Align SLAM metrics rows to pose-error rows by nearest timestamp."""
    aligned = pd.merge_asof(
        pose_errors.sort_values('timestamp'),
        metrics.sort_values('timestamp'),
        on='timestamp',
        direction='nearest',
        tolerance=max_diff,
    )
    aligned = aligned.dropna().reset_index(drop=True)
    return aligned


def package_sequence(metrics_path: str,
                     estimated_path: str,
                     groundtruth_path: str,
                     output_dir: str,
                     gt_format: str = 'euroc',
                     est_format: str = 'tum',
                     max_diff: float = 0.01) -> str:
    os.makedirs(output_dir, exist_ok=True)

    gt = load_euroc_groundtruth(groundtruth_path) if gt_format == 'euroc' else load_tum_trajectory(groundtruth_path)
    est = load_tum_trajectory(estimated_path) if est_format == 'tum' else load_euroc_groundtruth(estimated_path)
    metrics = pd.read_csv(metrics_path)

    matched = associate_trajectories(gt, est, max_diff=max_diff)
    if matched.empty:
        raise RuntimeError('No matched trajectory pairs found while packaging sequence.')

    pose_errors = compute_pose_errors(matched, align=True)
    aligned_metrics = align_metrics_to_pose_errors(metrics, pose_errors, max_diff=max_diff)
    if len(aligned_metrics) != len(pose_errors):
        min_len = min(len(aligned_metrics), len(pose_errors))
        aligned_metrics = aligned_metrics.iloc[:min_len].reset_index(drop=True)
        pose_errors = pose_errors.iloc[:min_len].reset_index(drop=True)

    if len(aligned_metrics) != len(pose_errors):
        raise RuntimeError(
            f'Row mismatch after alignment: metrics={len(aligned_metrics)}, pose_errors={len(pose_errors)}'
        )

    shutil.copyfile(groundtruth_path, os.path.join(output_dir, 'groundtruth.csv'))
    shutil.copyfile(estimated_path, os.path.join(output_dir, 'estimated.csv'))
    aligned_metrics.to_csv(os.path.join(output_dir, 'slam_metrics.csv'), index=False)
    pose_errors.to_csv(os.path.join(output_dir, 'pose_errors.csv'), index=False)

    summary_path = os.path.join(output_dir, 'packaging_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as handle:
        handle.write('Packaged unified training sequence\n')
        handle.write(f'rows: {len(aligned_metrics)}\n')
        handle.write(f'pose_error_mean: {pose_errors["pose_error"].mean():.6f}\n')
        handle.write(f'output_dir: {output_dir}\n')

    return output_dir


def main():
    parser = argparse.ArgumentParser(description='Package unified outputs into training-ready self-aware sequence format')
    parser.add_argument('--metrics', required=True, help='Path to unified slam_metrics.csv')
    parser.add_argument('--estimated', required=True, help='Path to unified estimated trajectory (TUM)')
    parser.add_argument('--groundtruth', required=True, help='Path to EuRoC groundtruth CSV')
    parser.add_argument('--sequence-name', required=True, help='Output sequence directory name, e.g. MH_01_unified')
    parser.add_argument('--dataset-root', default=os.path.join(PROJECT_ROOT, 'slam_metrics_dataset'),
                        help='Root directory for packaged training sequences')
    parser.add_argument('--max-diff', type=float, default=0.01, help='Max timestamp association tolerance')
    args = parser.parse_args()

    output_dir = os.path.join(args.dataset_root, args.sequence_name)
    package_sequence(
        metrics_path=args.metrics,
        estimated_path=args.estimated,
        groundtruth_path=args.groundtruth,
        output_dir=output_dir,
        max_diff=args.max_diff,
    )
    print(output_dir)


if __name__ == '__main__':
    main()
