"""
Task 2.3: Failure/Target Definition

Legacy helpers are kept for backward compatibility.
V2 training uses unified future targets:

- regression target: future max pose error over the next H frames
- classification target: future max pose error > threshold OR future tracking lost
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def create_failure_labels(
    pose_errors: pd.DataFrame,
    slam_metrics: pd.DataFrame,
    error_threshold: float = 0.3,
    use_tracking_state: bool = True,
) -> np.ndarray:
    """Legacy current-frame failure labels."""
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


def create_predictive_failure_labels(
    failure_labels: np.ndarray,
    prediction_horizon: int = 20,
) -> np.ndarray:
    """Legacy future-failure labels from binary current-failure labels."""
    n = len(failure_labels)
    predictive = np.zeros(n, dtype=int)

    for i in range(n):
        end = min(i + prediction_horizon + 1, n)
        if np.any(failure_labels[i + 1:end]):
            predictive[i] = 1

    return predictive


def create_future_error_targets(
    pose_errors: pd.DataFrame | np.ndarray,
    prediction_horizon: int = 10,
    aggregation: str = 'max',
) -> np.ndarray:
    """Create regression targets from future pose error over a fixed horizon.

    The final `prediction_horizon` rows are left as NaN and should be trimmed
    before temporal window construction.
    """
    if isinstance(pose_errors, pd.DataFrame):
        values = pose_errors['pose_error'].values.astype(np.float32)
    else:
        values = np.asarray(pose_errors, dtype=np.float32)

    targets = np.full(len(values), np.nan, dtype=np.float32)
    for idx in range(len(values)):
        start = idx + 1
        end = idx + prediction_horizon + 1
        if end > len(values):
            break
        future_slice = values[start:end]
        if aggregation == 'max':
            targets[idx] = float(np.max(future_slice))
        elif aggregation == 'mean':
            targets[idx] = float(np.mean(future_slice))
        else:
            raise ValueError(f"Unsupported future target aggregation: {aggregation}")

    return targets


def create_future_failure_labels(
    pose_errors: pd.DataFrame | np.ndarray,
    slam_metrics: pd.DataFrame,
    error_threshold: float = 1.0,
    prediction_horizon: int = 10,
    use_tracking_state: bool = True,
    aggregation: str = 'max',
) -> np.ndarray:
    """Create classification labels aligned with the future regression target."""
    future_error = create_future_error_targets(
        pose_errors=pose_errors,
        prediction_horizon=prediction_horizon,
        aggregation=aggregation,
    )
    failure = (future_error > error_threshold).astype(np.float32)

    if use_tracking_state and 'tracking_state' in slam_metrics.columns:
        tracking_lost = (slam_metrics['tracking_state'].fillna(0).astype(int).values == 0).astype(np.float32)
        tracking_failure = np.zeros(len(tracking_lost), dtype=np.float32)
        for idx in range(len(tracking_lost)):
            start = idx + 1
            end = idx + prediction_horizon + 1
            if end > len(tracking_lost):
                break
            if np.any(tracking_lost[start:end]):
                tracking_failure[idx] = 1.0
        failure = np.maximum(failure, tracking_failure)

    failure[np.isnan(future_error)] = np.nan
    return failure
