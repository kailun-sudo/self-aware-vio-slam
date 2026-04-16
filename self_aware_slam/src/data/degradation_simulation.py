"""
Task 2.4: Sensor Degradation Simulation

Simulate degraded sensor observations to augment the training dataset.

Camera degradation (applied to images via OpenCV):
  - motion blur (GaussianBlur)
  - gaussian noise
  - brightness change (addWeighted)
  - image dropout

IMU degradation (applied to IMU readings):
  - bias drift
  - noise amplification

The effect on SLAM metrics is simulated by perturbing the feature vectors
directly (since we may not have raw images in the ML pipeline).
"""

import numpy as np
from typing import Optional


def simulate_camera_degradation_on_features(
        features: np.ndarray,
        degradation_type: str,
        severity: float = 0.5,
        rng: Optional[np.random.RandomState] = None
) -> np.ndarray:
    """Simulate effect of camera degradation on SLAM features.

    Instead of modifying raw images, we directly perturb the extracted
    SLAM features to simulate what would happen under degraded conditions.

    Feature columns (by index):
        0: feature_count
        1: feature_tracking_ratio
        2: reprojection_error_mean
        3: reprojection_error_std
        4: imu_residual_norm
        5: camera_motion_magnitude
        6: tracking_length

    Args:
        features: (N, 7) feature array
        degradation_type: 'motion_blur', 'gaussian_noise', 'brightness_change', 'image_dropout'
        severity: 0.0 (none) to 1.0 (severe)
        rng: Random state

    Returns:
        Degraded feature array
    """
    if rng is None:
        rng = np.random.RandomState()

    degraded = features.copy()
    n = len(features)

    if degradation_type == 'motion_blur':
        # Reduces feature count and tracking ratio, increases reprojection error
        degraded[:, 0] *= (1 - 0.5 * severity)  # feature_count drops
        degraded[:, 1] *= (1 - 0.4 * severity)  # tracking ratio drops
        degraded[:, 2] += severity * 0.5 * np.abs(rng.normal(0, 1, n))  # reproj error up
        degraded[:, 3] += severity * 0.3 * np.abs(rng.normal(0, 1, n))

    elif degradation_type == 'gaussian_noise':
        # Increases reprojection error, slightly reduces features
        degraded[:, 0] *= (1 - 0.3 * severity)
        degraded[:, 2] += severity * 0.3 * np.abs(rng.normal(0, 1, n))
        degraded[:, 3] += severity * 0.2 * np.abs(rng.normal(0, 1, n))

    elif degradation_type == 'brightness_change':
        # Major impact on feature detection
        degraded[:, 0] *= (1 - 0.6 * severity)
        degraded[:, 1] *= (1 - 0.5 * severity)
        degraded[:, 2] += severity * 0.4 * np.abs(rng.normal(0, 1, n))

    elif degradation_type == 'image_dropout':
        # Random frames lose all features
        dropout_mask = rng.random(n) < (severity * 0.3)
        degraded[dropout_mask, 0] = rng.randint(0, 10, dropout_mask.sum())
        degraded[dropout_mask, 1] = rng.uniform(0, 0.1, dropout_mask.sum())
        degraded[dropout_mask, 2] += 1.0

    # Clamp values
    degraded[:, 0] = np.clip(degraded[:, 0], 0, 500)
    degraded[:, 1] = np.clip(degraded[:, 1], 0, 1)
    degraded[:, 2] = np.clip(degraded[:, 2], 0, 10)
    degraded[:, 3] = np.clip(degraded[:, 3], 0, 5)

    return degraded


def simulate_imu_degradation_on_features(
        features: np.ndarray,
        degradation_type: str,
        severity: float = 0.5,
        rng: Optional[np.random.RandomState] = None
) -> np.ndarray:
    """Simulate effect of IMU degradation on SLAM features.

    Args:
        features: (N, 7) feature array
        degradation_type: 'bias_drift', 'noise_amplification'
        severity: 0.0 to 1.0
        rng: Random state

    Returns:
        Degraded feature array
    """
    if rng is None:
        rng = np.random.RandomState()

    degraded = features.copy()
    n = len(features)

    if degradation_type == 'bias_drift':
        # Gradually increasing IMU residual
        drift = severity * 0.1 * np.cumsum(rng.normal(0, 0.01, n))
        degraded[:, 4] += np.abs(drift)

    elif degradation_type == 'noise_amplification':
        # Sudden increase in IMU noise
        degraded[:, 4] *= (1 + severity * 5)
        degraded[:, 4] += severity * 0.1 * np.abs(rng.normal(0, 1, n))

    return degraded


def augment_dataset_with_degradation(
        features: np.ndarray,
        pose_errors: np.ndarray,
        n_augmented: int = 3,
        seed: int = 42
) -> tuple:
    """Augment dataset by applying random degradations.

    Args:
        features: (N, 7) original features
        pose_errors: (N,) original pose errors
        n_augmented: Number of augmented copies per degradation type
        seed: Random seed

    Returns:
        (augmented_features, augmented_errors, degradation_types)
        All including originals.
    """
    rng = np.random.RandomState(seed)

    all_features = [features]
    all_errors = [pose_errors]
    all_types = ['none'] * len(features)

    camera_types = ['motion_blur', 'gaussian_noise', 'brightness_change', 'image_dropout']
    imu_types = ['bias_drift', 'noise_amplification']

    for _ in range(n_augmented):
        for deg_type in camera_types:
            severity = rng.uniform(0.3, 0.9)
            deg_features = simulate_camera_degradation_on_features(
                features, deg_type, severity, rng)
            # Degradation increases pose error
            error_scale = 1 + severity * 2
            deg_errors = pose_errors * error_scale + severity * 0.05 * np.abs(rng.normal(0, 1, len(pose_errors)))
            all_features.append(deg_features)
            all_errors.append(deg_errors)
            all_types.extend([deg_type] * len(features))

        for deg_type in imu_types:
            severity = rng.uniform(0.3, 0.9)
            deg_features = simulate_imu_degradation_on_features(
                features, deg_type, severity, rng)
            error_scale = 1 + severity * 1.5
            deg_errors = pose_errors * error_scale
            all_features.append(deg_features)
            all_errors.append(deg_errors)
            all_types.extend([deg_type] * len(features))

    return (np.concatenate(all_features),
            np.concatenate(all_errors),
            all_types)
