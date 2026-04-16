"""
Task 2.2: Temporal Window Construction

Build temporal context windows from SLAM feature sequences.

Input tensor: X = [features_{t-k}, ..., features_t]
Output: y = pose_error_t
"""

import numpy as np
from typing import Tuple


def create_temporal_windows(features: np.ndarray, targets: np.ndarray,
                            window_size: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding temporal windows from feature sequences.

    Args:
        features: (N, D) feature array
        targets: (N,) target array (e.g., pose_error)
        window_size: Number of frames in each window

    Returns:
        X: (N - window_size + 1, window_size, D) windowed features
        y: (N - window_size + 1,) corresponding targets
    """
    n_samples = len(features) - window_size + 1
    if n_samples <= 0:
        raise ValueError(f"Sequence length {len(features)} too short for window_size {window_size}")

    n_features = features.shape[1]
    X = np.zeros((n_samples, window_size, n_features), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.float32)

    for i in range(n_samples):
        X[i] = features[i:i + window_size]
        y[i] = targets[i + window_size - 1]

    return X, y


def create_temporal_windows_with_labels(
        features: np.ndarray,
        pose_errors: np.ndarray,
        failure_labels: np.ndarray,
        window_size: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create temporal windows with both regression and classification targets.

    Args:
        features: (N, D)
        pose_errors: (N,)
        failure_labels: (N,) binary

    Returns:
        X: (M, window_size, D)
        y_error: (M,) pose error targets
        y_failure: (M,) failure labels
    """
    n_samples = len(features) - window_size + 1
    if n_samples <= 0:
        raise ValueError(f"Sequence too short for window_size {window_size}")

    n_features = features.shape[1]
    X = np.zeros((n_samples, window_size, n_features), dtype=np.float32)
    y_error = np.zeros(n_samples, dtype=np.float32)
    y_failure = np.zeros(n_samples, dtype=np.float32)

    for i in range(n_samples):
        X[i] = features[i:i + window_size]
        y_error[i] = pose_errors[i + window_size - 1]
        y_failure[i] = failure_labels[i + window_size - 1]

    return X, y_error, y_failure
