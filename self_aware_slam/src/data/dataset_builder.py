"""
Task 2.5: Dataset Construction and Split

V2 dataset builder for a better-defined learning problem:

- regression target: future max pose error over next H frames
- classification target: same future target thresholded into failure/non-failure
- train/val/test all use the same target semantics
- trend-aware learning features are used for training

Runtime inference remains on the legacy 7-D feature set.
"""

from __future__ import annotations

import os
import pickle
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.feature_engineering import (
    LEARNING_FEATURE_COLUMNS,
    compute_normalization_stats,
    extract_learning_features,
    normalize_features,
)
from src.data.temporal_window import create_temporal_windows_with_labels
from src.data.failure_labels import (
    create_future_error_targets,
    create_future_failure_labels,
)
from src.data.degradation_simulation import augment_dataset_with_degradation
from src.utils.config_loader import load_config


def load_sequence_data(seq_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load SLAM metrics and pose errors for a sequence."""
    metrics = pd.read_csv(os.path.join(seq_dir, 'slam_metrics.csv'))
    errors = pd.read_csv(os.path.join(seq_dir, 'pose_errors.csv'))
    return metrics, errors


def _resolve_dataset_path(config: dict) -> str:
    return config['paths'].get(
        'train_dataset_path',
        os.path.join(config['paths']['results_dir'], 'train_dataset_v2.pkl'),
    )


def _resolve_sequence_splits(all_sequence_data: List[Tuple[str, pd.DataFrame, pd.DataFrame]], config: dict) -> Dict[str, List]:
    n_seq = len(all_sequence_data)
    if n_seq >= 5:
        train_indices = [0, 1, 2]
        val_indices = [3]
        test_indices = [4]
    elif n_seq == 1:
        print("  Note: single sequence mode — using same data for train/val/test")
        train_indices = [0]
        val_indices = [0]
        test_indices = [0]
    else:
        rng = np.random.RandomState(42)
        indices = rng.permutation(n_seq)
        n_train = max(1, int(n_seq * config['dataset']['train_split']))
        n_val = max(1, int(n_seq * config['dataset']['val_split']))
        train_indices = indices[:n_train].tolist()
        val_indices = indices[n_train:n_train + n_val].tolist()
        test_indices = indices[n_train + n_val:].tolist()
        if not test_indices:
            test_indices = val_indices[-1:]

    return {
        'train': [all_sequence_data[i] for i in train_indices],
        'val': [all_sequence_data[i] for i in val_indices],
        'test': [all_sequence_data[i] for i in test_indices],
    }


def _prepare_targets(
    metrics: pd.DataFrame,
    errors: pd.DataFrame,
    prediction_horizon: int,
    error_threshold: float,
    use_tracking_lost: bool,
    regression_aggregation: str,
) -> Tuple[np.ndarray, np.ndarray]:
    regression_targets = create_future_error_targets(
        pose_errors=errors,
        prediction_horizon=prediction_horizon,
        aggregation=regression_aggregation,
    )
    failure_targets = create_future_failure_labels(
        pose_errors=errors,
        slam_metrics=metrics,
        error_threshold=error_threshold,
        prediction_horizon=prediction_horizon,
        use_tracking_state=use_tracking_lost,
        aggregation=regression_aggregation,
    )
    return regression_targets.astype(np.float32), failure_targets.astype(np.float32)


def _trim_for_future_targets(
    features: np.ndarray,
    regression_targets: np.ndarray,
    failure_targets: np.ndarray,
    prediction_horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid_length = len(features) - prediction_horizon
    if valid_length <= 0:
        raise ValueError(
            f"Sequence length {len(features)} is too short for prediction_horizon={prediction_horizon}"
        )
    features = features[:valid_length]
    regression_targets = regression_targets[:valid_length]
    failure_targets = failure_targets[:valid_length]

    valid_mask = (~np.isnan(regression_targets)) & (~np.isnan(failure_targets))
    return features[valid_mask], regression_targets[valid_mask], failure_targets[valid_mask]


def build_dataset(config: dict | None = None) -> Dict:
    """Build the complete reliability dataset."""
    if config is None:
        config = load_config()

    data_dir = config['paths']['slam_metrics_dir']
    sequences = config['dataset']['euroc_sequences']
    window_size = config['temporal']['window_size']
    rolling_window = config['features'].get('rolling_window', 5)
    learning_feature_names = config['features'].get('learning_names', LEARNING_FEATURE_COLUMNS)
    prediction_horizon = config['targets']['prediction_horizon']
    error_threshold = config['targets']['classification_error_threshold']
    use_tracking_lost = config['targets'].get('use_tracking_lost', True)
    regression_aggregation = config['targets'].get('regression_aggregation', 'max')
    augment_training = config['dataset'].get('augment_training', False)

    all_sequence_data = []
    for seq in sequences:
        seq_dir = os.path.join(data_dir, seq)
        if not os.path.exists(seq_dir):
            print(f"  Warning: {seq_dir} not found, skipping")
            continue
        metrics, errors = load_sequence_data(seq_dir)
        all_sequence_data.append((seq, metrics, errors))
        print(f"  Loaded {seq}: {len(metrics)} frames")

    if not all_sequence_data:
        raise FileNotFoundError(f"No sequence data found in {data_dir}")

    splits = _resolve_sequence_splits(all_sequence_data, config)
    print(
        f"\nSplit: train={[s[0] for s in splits['train']]}, "
        f"val={[s[0] for s in splits['val']]}, "
        f"test={[s[0] for s in splits['test']]}"
    )

    train_features_list = []
    for _, metrics, _ in splits['train']:
        features = extract_learning_features(
            metrics,
            feature_columns=learning_feature_names,
            rolling_window=rolling_window,
        )
        train_features_list.append(features)
    all_train_features = np.concatenate(train_features_list)
    norm_stats = compute_normalization_stats(all_train_features)

    dataset = {
        'norm_stats': norm_stats,
        'feature_names': learning_feature_names,
        'window_size': window_size,
        'target_definition': {
            'mode': config['targets']['mode'],
            'prediction_horizon': prediction_horizon,
            'regression_target': config['targets']['regression_target'],
            'regression_aggregation': regression_aggregation,
            'classification_error_threshold': error_threshold,
            'use_tracking_lost': use_tracking_lost,
        },
    }

    for split_name, split_data in splits.items():
        X_list, y_error_list, y_failure_list = [], [], []

        for seq_name, metrics, errors in split_data:
            features = extract_learning_features(
                metrics,
                feature_columns=learning_feature_names,
                rolling_window=rolling_window,
            )
            features_norm = normalize_features(features, norm_stats)
            regression_targets, failure_targets = _prepare_targets(
                metrics=metrics,
                errors=errors,
                prediction_horizon=prediction_horizon,
                error_threshold=error_threshold,
                use_tracking_lost=use_tracking_lost,
                regression_aggregation=regression_aggregation,
            )

            if split_name == 'train' and augment_training:
                aug_features, aug_errors, _ = augment_dataset_with_degradation(
                    features_norm,
                    errors['pose_error'].values.astype(np.float32),
                    n_augmented=2,
                    seed=hash(seq_name) % 10000,
                )
                # Keep augmentation optional; future targets remain aligned only with pose error.
                regression_targets = create_future_error_targets(
                    pose_errors=aug_errors,
                    prediction_horizon=prediction_horizon,
                    aggregation=regression_aggregation,
                )
                failure_targets = (regression_targets > error_threshold).astype(np.float32)
                features_for_windows, regression_targets, failure_targets = _trim_for_future_targets(
                    aug_features,
                    regression_targets,
                    failure_targets,
                    prediction_horizon,
                )
            else:
                features_for_windows, regression_targets, failure_targets = _trim_for_future_targets(
                    features_norm,
                    regression_targets,
                    failure_targets,
                    prediction_horizon,
                )

            if len(features_for_windows) < window_size:
                print(
                    f"  Warning: {seq_name} too short after horizon trim "
                    f"(len={len(features_for_windows)}), skipping"
                )
                continue

            X, y_err, y_fail = create_temporal_windows_with_labels(
                features_for_windows,
                regression_targets,
                failure_targets,
                window_size=window_size,
            )

            X_list.append(X)
            y_error_list.append(y_err)
            y_failure_list.append(y_fail)

        if not X_list:
            raise ValueError(
                f"No usable samples produced for split '{split_name}'. "
                f"Check prediction_horizon={prediction_horizon}, window_size={window_size}, "
                "and dataset sequence lengths."
            )

        dataset[split_name] = {
            'X': np.concatenate(X_list),
            'y_error': np.concatenate(y_error_list),
            'y_failure': np.concatenate(y_failure_list),
        }

        n = len(dataset[split_name]['X'])
        n_fail = dataset[split_name]['y_failure'].sum()
        print(f"  {split_name}: {n} samples, {int(n_fail)} failures ({100 * n_fail / max(n, 1):.1f}%)")

    return dataset


def save_dataset(dataset: Dict, output_path: str):
    """Save dataset to pickle file."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'wb') as handle:
        pickle.dump(dataset, handle)
    print(f"Dataset saved to {output_path}")


def load_dataset(path: str) -> Dict:
    """Load dataset from pickle file."""
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def main():
    config = load_config()
    print("Building reliability dataset...")
    dataset = build_dataset(config)

    output_path = _resolve_dataset_path(config)
    save_dataset(dataset, output_path)

    print("\nDataset summary:")
    for split in ['train', 'val', 'test']:
        X = dataset[split]['X']
        print(
            f"  {split}: X={X.shape}, "
            f"y_error range=[{dataset[split]['y_error'].min():.3f}, {dataset[split]['y_error'].max():.3f}]"
        )


if __name__ == '__main__':
    main()
