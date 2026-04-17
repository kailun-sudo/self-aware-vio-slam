"""
Task 2.1: Feature Engineering

Two feature sets are supported:

1. Runtime features (7-D)
   Used by the currently deployed online/offline inference pipeline.

2. Learning features (trend-aware)
   Used by the v2 training/benchmark pipeline to make the predictive
   task better aligned with future-error prediction.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    'feature_count',
    'feature_tracking_ratio',
    'reprojection_error_mean',
    'reprojection_error_std',
    'imu_residual_norm',
    'camera_motion_magnitude',
    'tracking_length',
]


LEARNING_FEATURE_COLUMNS = FEATURE_COLUMNS + [
    'delta_feature_tracking_ratio',
    'delta_reprojection_error_mean',
    'delta_reprojection_error_std',
    'delta_imu_residual_norm',
    'delta_camera_motion_magnitude',
    'delta_tracking_length',
    'rolling_mean_feature_tracking_ratio',
    'rolling_std_feature_tracking_ratio',
    'rolling_mean_reprojection_error_mean',
    'rolling_std_reprojection_error_mean',
    'rolling_mean_camera_motion_magnitude',
    'rolling_std_camera_motion_magnitude',
    'slope_feature_tracking_ratio',
    'slope_reprojection_error_mean',
    'slope_camera_motion_magnitude',
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


def _compute_tracking_length(tracking_state: pd.Series) -> np.ndarray:
    tracking_lengths = []
    current = 0
    for state in tracking_state.fillna(0).astype(int):
        current = current + 1 if state else 0
        tracking_lengths.append(current)
    return np.asarray(tracking_lengths, dtype=np.float32)


def _compute_delta(series: pd.Series) -> pd.Series:
    return series.diff().fillna(0.0)


def _compute_rolling_mean(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def _compute_rolling_std(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).std().fillna(0.0)


def _compute_window_slope(series: pd.Series, window: int) -> pd.Series:
    slope = (series - series.shift(window - 1)) / max(window - 1, 1)
    return slope.fillna(0.0)


def prepare_feature_dataframe(
    slam_metrics: pd.DataFrame,
    feature_columns: List[str] | None = None,
) -> pd.DataFrame:
    """Fill canonical runtime feature columns from available aliases."""
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS

    prepared = slam_metrics.copy()

    if 'tracking_length' not in prepared.columns and 'tracking_state' in prepared.columns:
        prepared['tracking_length'] = _compute_tracking_length(prepared['tracking_state'])

    for column in feature_columns:
        if column in prepared.columns:
            continue

        aliases = FEATURE_ALIASES.get(column, [column])
        source_column = next((alias for alias in aliases if alias in prepared.columns), None)
        if source_column is None:
            if column in {'reprojection_error_std', 'camera_motion_magnitude'}:
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
            # We only have a scalar geometric proxy, not a true std estimate.
            prepared[column] = 0.0
            continue

        prepared[column] = prepared[source_column]

    return prepared


def prepare_learning_feature_dataframe(
    slam_metrics: pd.DataFrame,
    feature_columns: List[str] | None = None,
    rolling_window: int = 5,
) -> pd.DataFrame:
    """Build trend-aware features for v2 training/benchmarking."""
    if feature_columns is None:
        feature_columns = LEARNING_FEATURE_COLUMNS

    prepared = prepare_feature_dataframe(slam_metrics, feature_columns=FEATURE_COLUMNS)

    prepared['delta_feature_tracking_ratio'] = _compute_delta(prepared['feature_tracking_ratio'])
    prepared['delta_reprojection_error_mean'] = _compute_delta(prepared['reprojection_error_mean'])
    prepared['delta_reprojection_error_std'] = _compute_delta(prepared['reprojection_error_std'])
    prepared['delta_imu_residual_norm'] = _compute_delta(prepared['imu_residual_norm'])
    prepared['delta_camera_motion_magnitude'] = _compute_delta(prepared['camera_motion_magnitude'])
    prepared['delta_tracking_length'] = _compute_delta(prepared['tracking_length'])

    prepared['rolling_mean_feature_tracking_ratio'] = _compute_rolling_mean(
        prepared['feature_tracking_ratio'], rolling_window
    )
    prepared['rolling_std_feature_tracking_ratio'] = _compute_rolling_std(
        prepared['feature_tracking_ratio'], rolling_window
    )
    prepared['rolling_mean_reprojection_error_mean'] = _compute_rolling_mean(
        prepared['reprojection_error_mean'], rolling_window
    )
    prepared['rolling_std_reprojection_error_mean'] = _compute_rolling_std(
        prepared['reprojection_error_mean'], rolling_window
    )
    prepared['rolling_mean_camera_motion_magnitude'] = _compute_rolling_mean(
        prepared['camera_motion_magnitude'], rolling_window
    )
    prepared['rolling_std_camera_motion_magnitude'] = _compute_rolling_std(
        prepared['camera_motion_magnitude'], rolling_window
    )

    prepared['slope_feature_tracking_ratio'] = _compute_window_slope(
        prepared['feature_tracking_ratio'], rolling_window
    )
    prepared['slope_reprojection_error_mean'] = _compute_window_slope(
        prepared['reprojection_error_mean'], rolling_window
    )
    prepared['slope_camera_motion_magnitude'] = _compute_window_slope(
        prepared['camera_motion_magnitude'], rolling_window
    )

    missing = [column for column in feature_columns if column not in prepared.columns]
    if missing:
        raise KeyError(f"Learning feature columns missing after preparation: {missing}")

    return prepared


def extract_features(
    slam_metrics: pd.DataFrame,
    feature_columns: List[str] | None = None,
) -> np.ndarray:
    """Extract runtime features for inference compatibility."""
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS

    prepared = prepare_feature_dataframe(slam_metrics, feature_columns=feature_columns)
    features = prepared[feature_columns].values.astype(np.float32)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def extract_learning_features(
    slam_metrics: pd.DataFrame,
    feature_columns: List[str] | None = None,
    rolling_window: int = 5,
) -> np.ndarray:
    """Extract trend-aware features for the v2 training task."""
    if feature_columns is None:
        feature_columns = LEARNING_FEATURE_COLUMNS

    prepared = prepare_learning_feature_dataframe(
        slam_metrics,
        feature_columns=feature_columns,
        rolling_window=rolling_window,
    )
    features = prepared[feature_columns].values.astype(np.float32)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def compute_normalization_stats(features: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute z-score normalization statistics."""
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-8] = 1.0
    return {'mean': mean, 'std': std}


def normalize_features(features: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    """Apply z-score normalization."""
    return (features - stats['mean']) / stats['std']


def denormalize_features(features: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    """Reverse z-score normalization."""
    return features * stats['std'] + stats['mean']
