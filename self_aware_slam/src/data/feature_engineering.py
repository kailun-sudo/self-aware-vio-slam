"""
Task 2.1: Feature Engineering

Extract and normalize features from SLAM logs for reliability learning.

Features:
  - feature_count
  - feature_tracking_ratio
  - reprojection_error_mean
  - reprojection_error_std
  - imu_residual_norm
  - camera_motion_magnitude
  - tracking_length

Normalization: z-score (mean=0, std=1)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


FEATURE_COLUMNS = [
    'feature_count',
    'feature_tracking_ratio',
    'reprojection_error_mean',
    'reprojection_error_std',
    'imu_residual_norm',
    'camera_motion_magnitude',
    'tracking_length',
]

FEATURE_ALIASES = {
    'feature_count': ['feature_count', 'num_matches', 'num_keypoints'],
    'feature_tracking_ratio': ['feature_tracking_ratio', 'inlier_ratio'],
    'reprojection_error_mean': ['reprojection_error_mean', 'pose_optimization_residual'],
    'reprojection_error_std': ['reprojection_error_std', 'mean_epipolar_error'],
    'imu_residual_norm': ['imu_residual_norm', 'pose_optimization_residual'],
    'camera_motion_magnitude': ['camera_motion_magnitude', 'trajectory_increment_norm', 'imu_delta_translation'],
    'tracking_length': ['tracking_length'],
}


def prepare_feature_dataframe(slam_metrics: pd.DataFrame,
                              feature_columns=None) -> pd.DataFrame:
    """Fill canonical feature columns from aliases when needed."""
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS

    prepared = slam_metrics.copy()

    if 'tracking_length' not in prepared.columns and 'tracking_state' in prepared.columns:
        tracking_lengths = []
        current = 0
        for state in prepared['tracking_state'].fillna(0).astype(int):
            current = current + 1 if state else 0
            tracking_lengths.append(current)
        prepared['tracking_length'] = tracking_lengths

    for column in feature_columns:
        if column in prepared.columns:
            continue

        aliases = FEATURE_ALIASES.get(column, [column])
        source_column = next((alias for alias in aliases if alias in prepared.columns), None)
        if source_column is None:
            if column == 'reprojection_error_std':
                prepared[column] = 0.0
                continue
            if column == 'camera_motion_magnitude':
                prepared[column] = 0.0
                continue
            if column == 'tracking_length':
                prepared[column] = np.arange(1, len(prepared) + 1, dtype=np.float32)
                continue
            raise KeyError(
                f"Required feature column '{column}' not found. "
                f"Available columns: {list(prepared.columns)}"
            )

        if column == 'reprojection_error_std' and source_column == 'mean_epipolar_error':
            prepared[column] = 0.0
            continue

        prepared[column] = prepared[source_column]

    return prepared


def extract_features(slam_metrics: pd.DataFrame,
                     feature_columns=None) -> np.ndarray:
    """Extract feature columns from SLAM metrics DataFrame.

    Args:
        slam_metrics: DataFrame with SLAM internal metrics

    Returns:
        numpy array of shape (N, num_features)
    """
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS

    prepared = prepare_feature_dataframe(slam_metrics, feature_columns=feature_columns)
    features = prepared[feature_columns].values.astype(np.float32)
    # Replace NaN/inf with 0
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features


def compute_normalization_stats(features: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute z-score normalization statistics.

    Args:
        features: (N, D) feature array

    Returns:
        Dict with 'mean' and 'std' arrays of shape (D,)
    """
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    # Prevent division by zero
    std[std < 1e-8] = 1.0
    return {'mean': mean, 'std': std}


def normalize_features(features: np.ndarray,
                       stats: Dict[str, np.ndarray]) -> np.ndarray:
    """Apply z-score normalization.

    Args:
        features: (N, D) feature array
        stats: Dict with 'mean' and 'std'

    Returns:
        Normalized feature array
    """
    return (features - stats['mean']) / stats['std']


def denormalize_features(features: np.ndarray,
                         stats: Dict[str, np.ndarray]) -> np.ndarray:
    """Reverse z-score normalization."""
    return features * stats['std'] + stats['mean']
