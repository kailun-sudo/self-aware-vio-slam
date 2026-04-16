"""
Parse VINS-Fusion output into project-standard CSV formats.

VINS-Fusion produces:
  1. Trajectory file (TUM format): timestamp tx ty tz qx qy qz qw
     Typically: vins_result_loop.csv or vins_result_no_loop.csv
  2. Stdout/log output containing feature counts, residuals, etc.

This module handles both sources:
  - Trajectory -> estimated.csv
  - Trajectory -> approximate slam_metrics.csv (motion, tracking state)
  - Log file -> enriched slam_metrics.csv (feature counts, reproj errors)
"""

import numpy as np
import pandas as pd
import re
import os
from typing import Optional, Tuple


def parse_vins_trajectory(traj_path: str) -> pd.DataFrame:
    """Parse VINS-Fusion trajectory output (TUM format).

    TUM format: timestamp tx ty tz qx qy qz qw (space-separated)
    Some VINS versions use comma separation.

    Returns:
        DataFrame with columns [timestamp, px, py, pz, qw, qx, qy, qz]
        (reordered to project-standard wxyz quaternion order)
    """
    # Detect separator
    with open(traj_path, 'r') as f:
        first_data_line = ''
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('%'):
                first_data_line = line
                break

    sep = ',' if ',' in first_data_line else r'\s+'

    df = pd.read_csv(traj_path, sep=sep, comment='#', header=None,
                     engine='python')

    if df.shape[1] < 8:
        raise ValueError(
            f"Expected 8 columns (TUM format), got {df.shape[1]} in {traj_path}")

    # TUM order: timestamp tx ty tz qx qy qz qw
    df = df.iloc[:, :8]
    df.columns = ['timestamp', 'px', 'py', 'pz', 'qx', 'qy', 'qz', 'qw']

    # Convert ns to seconds if needed
    if df['timestamp'].iloc[0] > 1e15:
        df['timestamp'] = df['timestamp'].astype(np.float64) / 1e9

    # Sanity check: timestamps should be plausible Unix seconds
    t0 = df['timestamp'].iloc[0]
    if not (1e8 < t0 < 2e10):
        raise ValueError(
            f"Timestamp {t0} is outside plausible range after conversion. "
            f"Expected Unix seconds (~1.4e9). Check format in {traj_path}."
        )

    # Reorder to project standard: wxyz
    df = df[['timestamp', 'px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz']]
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def compute_motion_from_trajectory(traj: pd.DataFrame) -> np.ndarray:
    """Compute per-frame camera motion magnitude from consecutive poses.

    Returns:
        Array of shape (N,) with Euclidean displacement between frames.
        First frame gets 0.
    """
    pos = traj[['px', 'py', 'pz']].values
    motion = np.zeros(len(pos))
    motion[1:] = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    return motion


def compute_tracking_state_from_gaps(timestamps: np.ndarray,
                                     max_gap_s: float = 0.15) -> np.ndarray:
    """Infer tracking state from timestamp gaps.

    If the gap between consecutive frames exceeds max_gap_s, mark the frame
    after the gap as tracking_state=0 (lost).

    Args:
        timestamps: sorted timestamp array
        max_gap_s: maximum expected inter-frame gap (default 0.15s for ~10Hz)

    Returns:
        Array of shape (N,) with 1=tracking OK, 0=tracking lost
    """
    state = np.ones(len(timestamps), dtype=int)
    if len(timestamps) < 2:
        return state
    dt = np.diff(timestamps)
    median_dt = np.median(dt)
    threshold = max(max_gap_s, median_dt * 3.0)
    gaps = dt > threshold
    state[1:][gaps] = 0
    return state


def parse_vins_log(log_path: str) -> Optional[pd.DataFrame]:
    """Parse VINS-Fusion stdout log for per-frame metrics.

    Extracts lines matching common VINS-Fusion debug patterns:
      - "feature_tracker: ... tracked N features"
      - "whole feature num: N"
      - "solver costs: ... iterations"
      - "average reprojection error: X"

    Returns:
        DataFrame with available columns, or None if log is not parseable.
        Columns may include: timestamp, feature_count, reprojection_error_mean,
        optimization_iterations
    """
    if not os.path.isfile(log_path):
        return None

    records = []
    current = {}

    # Patterns observed in VINS-Fusion output
    pat_features = re.compile(
        r'(?:feature|point)\s*(?:num|count)[:\s]*(\d+)', re.IGNORECASE)
    pat_reproj = re.compile(
        r'(?:reprojection|reproj)\s*(?:error|err)[:\s]*([\d.]+)', re.IGNORECASE)
    pat_iter = re.compile(
        r'(?:iteration|iter)[:\s]*(\d+)', re.IGNORECASE)
    pat_time = re.compile(
        r'\b(?:timestamp|time)[:\s]*([\d.]+)', re.IGNORECASE)

    with open(log_path, 'r') as f:
        for line in f:
            m_feat = pat_features.search(line)
            m_reproj = pat_reproj.search(line)
            m_iter = pat_iter.search(line)
            m_time = pat_time.search(line)

            if m_time:
                if current:
                    records.append(current)
                current = {'timestamp': float(m_time.group(1))}

            if m_feat:
                current['feature_count'] = int(m_feat.group(1))
            if m_reproj:
                current['reprojection_error_mean'] = float(m_reproj.group(1))
            if m_iter:
                current['optimization_iterations'] = int(m_iter.group(1))

    if current:
        records.append(current)

    # Drop records without a valid timestamp
    records = [r for r in records if 'timestamp' in r]

    if not records:
        return None

    return pd.DataFrame(records)


def build_slam_metrics_from_trajectory(
        traj: pd.DataFrame,
        log_metrics: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Build slam_metrics.csv from trajectory (and optional log data).

    Computes what we can from the trajectory:
      - camera_motion_magnitude: from consecutive poses
      - tracking_state: from timestamp gaps
      - tracking_length: cumulative tracking time

    Enriches from log data if available:
      - feature_count
      - reprojection_error_mean/std
      - optimization_iterations

    Missing fields get reasonable defaults so that the 11-column contract
    expected by dataset_builder is satisfied.

    Returns:
        DataFrame with all 11 slam_metrics columns.
    """
    n = len(traj)
    timestamps = traj['timestamp'].values

    motion = compute_motion_from_trajectory(traj)
    tracking_state = compute_tracking_state_from_gaps(timestamps)

    # Tracking length: cumulative time since start, reset on tracking loss
    tracking_length = np.zeros(n)
    for i in range(1, n):
        if tracking_state[i] == 1:
            tracking_length[i] = tracking_length[i - 1] + (
                timestamps[i] - timestamps[i - 1])
        else:
            tracking_length[i] = 0.0

    metrics = pd.DataFrame({
        'timestamp': timestamps,
        'feature_count': 0,
        'feature_tracking_ratio': 0.0,
        'reprojection_error_mean': 0.0,
        'reprojection_error_std': 0.0,
        'imu_residual_norm': 0.0,
        'optimization_iterations': 0,
        'tracking_time_ms': 0.0,
        'tracking_state': tracking_state,
        'camera_motion_magnitude': motion,
        'tracking_length': tracking_length,
    })

    # Enrich from log data if available
    if log_metrics is not None and len(log_metrics) > 0:
        metrics = _merge_log_metrics(metrics, log_metrics)

    return metrics


def _merge_log_metrics(metrics: pd.DataFrame,
                       log_metrics: pd.DataFrame) -> pd.DataFrame:
    """Merge parsed log metrics into the slam_metrics dataframe by nearest timestamp."""
    if 'timestamp' not in log_metrics.columns:
        return metrics

    log_ts = log_metrics['timestamp'].values
    met_ts = metrics['timestamp'].values

    enrichable = [c for c in log_metrics.columns
                  if c != 'timestamp' and c in metrics.columns]

    if not enrichable:
        return metrics

    for col in enrichable:
        log_vals = log_metrics[col].values
        for i, lt in enumerate(log_ts):
            idx = np.argmin(np.abs(met_ts - lt))
            if np.abs(met_ts[idx] - lt) < 0.1:
                metrics.at[idx, col] = log_vals[i]

    # Forward-fill zero columns that were partially enriched
    for col in enrichable:
        vals = metrics[col].values.copy()
        if np.any(vals != 0):
            for i in range(1, len(vals)):
                if vals[i] == 0 and vals[i - 1] != 0:
                    vals[i] = vals[i - 1]
            metrics[col] = vals

    return metrics


def save_estimated(traj: pd.DataFrame, output_path: str):
    """Save estimated trajectory in project CSV format."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    traj.to_csv(output_path, index=False,
                columns=['timestamp', 'px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz'])


def save_slam_metrics(metrics: pd.DataFrame, output_path: str):
    """Save slam_metrics in project CSV format (11 columns)."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    cols = [
        'timestamp', 'feature_count', 'feature_tracking_ratio',
        'reprojection_error_mean', 'reprojection_error_std',
        'imu_residual_norm', 'optimization_iterations',
        'tracking_time_ms', 'tracking_state',
        'camera_motion_magnitude', 'tracking_length'
    ]
    metrics.to_csv(output_path, index=False, columns=cols)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Parse VINS-Fusion output into project format')
    parser.add_argument('trajectory', help='VINS trajectory file (TUM format)')
    parser.add_argument('--log', help='VINS stdout log file (optional)')
    parser.add_argument('--out-estimated', default='estimated.csv')
    parser.add_argument('--out-metrics', default='slam_metrics.csv')
    args = parser.parse_args()

    traj = parse_vins_trajectory(args.trajectory)
    print(f"Parsed {len(traj)} poses from {args.trajectory}")

    log_metrics = parse_vins_log(args.log) if args.log else None
    if log_metrics is not None:
        print(f"Parsed {len(log_metrics)} log entries with columns: "
              f"{list(log_metrics.columns)}")

    metrics = build_slam_metrics_from_trajectory(traj, log_metrics)

    save_estimated(traj, args.out_estimated)
    save_slam_metrics(metrics, args.out_metrics)
    print(f"Saved: {args.out_estimated}, {args.out_metrics}")
