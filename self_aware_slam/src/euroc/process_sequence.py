"""
End-to-end processing of a single EuRoC sequence with VINS-Fusion output.

Takes:
  - EuRoC sequence directory (contains ground truth)
  - VINS-Fusion output file (trajectory in TUM format)
  - Optional VINS log file

Produces the 4 CSVs expected by dataset_builder:
  - slam_metrics.csv
  - pose_errors.csv
  - groundtruth.csv
  - estimated.csv

All saved under slam_metrics_dataset/{sequence_name}/
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.euroc.euroc_gt_parser import parse_euroc_gt, save_groundtruth
from src.euroc.vins_output_parser import (
    parse_vins_trajectory,
    parse_vins_log,
    build_slam_metrics_from_trajectory,
    save_estimated,
    save_slam_metrics,
)
from src.ground_truth_alignment import (
    associate_trajectories,
    compute_pose_errors,
)


# EuRoC sequence name mapping (folder name -> short name)
EUROC_SEQUENCE_MAP = {
    'MH_01_easy': 'MH_01',
    'MH_02_easy': 'MH_02',
    'MH_03_medium': 'MH_03',
    'MH_04_difficult': 'MH_04',
    'MH_05_difficult': 'MH_05',
    'V1_01_easy': 'V1_01',
    'V1_02_medium': 'V1_02',
    'V1_03_difficult': 'V1_03',
    'V2_01_easy': 'V2_01',
    'V2_02_medium': 'V2_02',
    'V2_03_difficult': 'V2_03',
    # Also accept short names directly
    'MH_01': 'MH_01',
    'MH_02': 'MH_02',
    'MH_03': 'MH_03',
    'MH_04': 'MH_04',
    'MH_05': 'MH_05',
}


def process_sequence(
    euroc_seq_dir: str,
    vins_trajectory_path: str,
    sequence_name: str,
    output_base_dir: str = 'slam_metrics_dataset',
    vins_log_path: str = None,
    max_assoc_diff: float = 0.02,
    align: bool = True,
) -> str:
    """Process a single EuRoC sequence into the project dataset format.

    Args:
        euroc_seq_dir: Path to EuRoC sequence root
            (e.g., /data/euroc/MH_01_easy/)
        vins_trajectory_path: Path to VINS-Fusion trajectory output
            (e.g., vins_result_no_loop.csv)
        sequence_name: Short sequence name (e.g., 'MH_01')
        output_base_dir: Base directory for output
        vins_log_path: Optional path to VINS stdout log
        max_assoc_diff: Max timestamp diff for GT-est association (seconds)
        align: Whether to apply Umeyama alignment

    Returns:
        Path to the output sequence directory
    """
    out_dir = os.path.join(output_base_dir, sequence_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nProcessing {sequence_name}")
    print(f"  EuRoC dir:  {euroc_seq_dir}")
    print(f"  VINS traj:  {vins_trajectory_path}")

    # 1. Parse ground truth
    print("  [1/4] Parsing EuRoC ground truth...")
    gt = parse_euroc_gt(euroc_seq_dir)
    gt_path = os.path.join(out_dir, 'groundtruth.csv')
    save_groundtruth(gt, gt_path)
    print(f"    -> {len(gt)} GT poses")

    # 2. Parse VINS-Fusion trajectory
    print("  [2/4] Parsing VINS-Fusion trajectory...")
    est = parse_vins_trajectory(vins_trajectory_path)
    est_path = os.path.join(out_dir, 'estimated.csv')
    save_estimated(est, est_path)
    print(f"    -> {len(est)} estimated poses")

    # 3. Compute pose errors (align + associate)
    print("  [3/4] Computing pose errors (Umeyama alignment)...")
    print(f"    GT  timestamp range: [{gt['timestamp'].iloc[0]:.6f}, "
          f"{gt['timestamp'].iloc[-1]:.6f}] ({len(gt)} poses)")
    print(f"    Est timestamp range: [{est['timestamp'].iloc[0]:.6f}, "
          f"{est['timestamp'].iloc[-1]:.6f}] ({len(est)} poses)")

    matched = associate_trajectories(gt, est, max_diff=max_assoc_diff)

    if len(matched) == 0:
        raise RuntimeError(
            f"Zero matched poses for {sequence_name}. "
            f"GT range [{gt['timestamp'].iloc[0]:.6f}, {gt['timestamp'].iloc[-1]:.6f}], "
            f"Est range [{est['timestamp'].iloc[0]:.6f}, {est['timestamp'].iloc[-1]:.6f}], "
            f"max_diff={max_assoc_diff}s. "
            f"Check that both are in seconds and cover overlapping time."
        )

    match_ratio = len(matched) / len(est)
    errors = compute_pose_errors(matched, align=align)
    print(f"    -> {len(errors)} matched frames ({match_ratio:.0%} of estimated), "
          f"mean ATE={errors['pose_error'].mean():.4f}m, "
          f"RMSE={np.sqrt((errors['pose_error']**2).mean()):.4f}m")

    if match_ratio < 0.5:
        print(f"    WARNING: only {match_ratio:.0%} of estimated poses matched GT. "
              f"Consider increasing --max-diff (current: {max_assoc_diff}s).")

    errors_path = os.path.join(out_dir, 'pose_errors.csv')
    errors.to_csv(errors_path, index=False)

    # 4. Build SLAM metrics from matched poses (same rows as pose_errors)
    print("  [4/4] Building SLAM metrics...")
    log_metrics = None
    if vins_log_path:
        log_metrics = parse_vins_log(vins_log_path)
        if log_metrics is not None:
            print(f"    Enriched from log: {len(log_metrics)} entries")

    matched_est = pd.DataFrame({
        'timestamp': matched['timestamp'].values,
        'px': matched['est_px'].values,
        'py': matched['est_py'].values,
        'pz': matched['est_pz'].values,
        'qw': matched['est_qw'].values,
        'qx': matched['est_qx'].values,
        'qy': matched['est_qy'].values,
        'qz': matched['est_qz'].values,
    })
    metrics = build_slam_metrics_from_trajectory(matched_est, log_metrics)

    # Assert row-count alignment: slam_metrics and pose_errors must match
    assert len(metrics) == len(errors), (
        f"Row count mismatch: slam_metrics={len(metrics)}, "
        f"pose_errors={len(errors)}. This is a bug."
    )

    metrics_path = os.path.join(out_dir, 'slam_metrics.csv')
    save_slam_metrics(metrics, metrics_path)

    # --- Summary diagnostic ---
    gt_duration = gt['timestamp'].iloc[-1] - gt['timestamp'].iloc[0]
    est_duration = est['timestamp'].iloc[-1] - est['timestamp'].iloc[0]
    print(f"\n  {'─' * 46}")
    print(f"  Summary: {sequence_name}")
    print(f"  {'─' * 46}")
    print(f"    Ground truth rows:    {len(gt):>6}  ({gt_duration:.1f}s)")
    print(f"    Estimated traj rows:  {len(est):>6}  ({est_duration:.1f}s)")
    print(f"    Matched rows:         {len(matched):>6}  ({match_ratio:.1%} of est)")
    print(f"    pose_errors.csv rows: {len(errors):>6}")
    print(f"    slam_metrics.csv rows:{len(metrics):>6}")
    print(f"    ATE mean/RMSE:        {errors['pose_error'].mean():.4f} / "
          f"{np.sqrt((errors['pose_error']**2).mean()):.4f} m")
    n_lost = (metrics['tracking_state'] == 0).sum()
    print(f"    Tracking lost frames: {n_lost:>6}  "
          f"({100*n_lost/max(len(metrics),1):.1f}%)")
    print(f"    Output: {out_dir}/")
    print(f"  {'─' * 46}")
    return out_dir


def auto_detect_sequence_name(euroc_dir: str) -> str:
    """Try to detect the sequence short name from the directory name."""
    basename = os.path.basename(os.path.normpath(euroc_dir))
    if basename in EUROC_SEQUENCE_MAP:
        return EUROC_SEQUENCE_MAP[basename]
    # Try partial match
    for full_name, short_name in EUROC_SEQUENCE_MAP.items():
        if full_name in basename:
            return short_name
    return basename


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Process EuRoC + VINS-Fusion -> dataset format')
    parser.add_argument('euroc_dir',
                        help='EuRoC sequence directory')
    parser.add_argument('vins_trajectory',
                        help='VINS-Fusion trajectory output (TUM format)')
    parser.add_argument('--name', default=None,
                        help='Sequence name (auto-detected if omitted)')
    parser.add_argument('--output-dir', default='slam_metrics_dataset',
                        help='Base output directory')
    parser.add_argument('--vins-log', default=None,
                        help='VINS-Fusion stdout log file')
    parser.add_argument('--max-diff', type=float, default=0.02,
                        help='Max timestamp diff for association (seconds)')
    parser.add_argument('--no-align', action='store_true',
                        help='Skip Umeyama alignment')
    args = parser.parse_args()

    seq_name = args.name or auto_detect_sequence_name(args.euroc_dir)
    process_sequence(
        euroc_seq_dir=args.euroc_dir,
        vins_trajectory_path=args.vins_trajectory,
        sequence_name=seq_name,
        output_base_dir=args.output_dir,
        vins_log_path=args.vins_log,
        max_assoc_diff=args.max_diff,
        align=not args.no_align,
    )
