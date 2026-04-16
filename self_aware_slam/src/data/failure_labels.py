"""
Task 2.3: Failure Label Definition

Define failure events for SLAM reliability learning.

Failure if:
  - pose_error > 0.3 m
  OR
  - tracking_lost = True

Creates binary failure_label column.
"""

import numpy as np
import pandas as pd


def create_failure_labels(pose_errors: pd.DataFrame,
                          slam_metrics: pd.DataFrame,
                          error_threshold: float = 0.3,
                          use_tracking_state: bool = True) -> np.ndarray:
    """Create binary failure labels.

    Args:
        pose_errors: DataFrame with 'pose_error' column
        slam_metrics: DataFrame with 'tracking_state' column
        error_threshold: Pose error threshold in meters
        use_tracking_state: Whether to also use tracking_state for labeling

    Returns:
        Binary array (1 = failure, 0 = ok)
    """
    failure = (pose_errors['pose_error'].values > error_threshold).astype(int)

    if use_tracking_state and 'tracking_state' in slam_metrics.columns:
        tracking_lost = (slam_metrics['tracking_state'].values == 0).astype(int)
        if len(failure) != len(tracking_lost):
            raise ValueError(
                f"Row count mismatch: pose_errors has {len(failure)} rows, "
                f"slam_metrics has {len(tracking_lost)} rows. "
                f"These must be aligned row-by-row."
            )
        failure = np.maximum(failure, tracking_lost)

    return failure


def create_predictive_failure_labels(failure_labels: np.ndarray,
                                     prediction_horizon: int = 20) -> np.ndarray:
    """Create labels for predicting future failure.

    For each frame t, label is 1 if any frame in [t+1, t+horizon] is a failure.
    This enables predictive failure detection (Δt seconds ahead).

    Args:
        failure_labels: (N,) binary failure labels
        prediction_horizon: Number of future frames to check

    Returns:
        (N,) predictive failure labels
    """
    n = len(failure_labels)
    predictive = np.zeros(n, dtype=int)

    for i in range(n):
        end = min(i + prediction_horizon + 1, n)
        if np.any(failure_labels[i + 1:end]):
            predictive[i] = 1

    return predictive
