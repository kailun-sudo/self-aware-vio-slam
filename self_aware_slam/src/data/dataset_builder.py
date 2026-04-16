"""
Task 2.5: Dataset Construction and Split

Build the complete training dataset from SLAM metrics and pose errors.
Combines feature engineering, temporal windows, failure labels, and
degradation simulation into a single pipeline.

Output: train_dataset.pkl with each sample containing:
  - feature_tensor
  - pose_error
  - failure_label
  - degradation_type

Split: 70% train, 15% val, 15% test (sequence-independent)
"""

import numpy as np
import pandas as pd
import pickle
import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.feature_engineering import (
    extract_features, compute_normalization_stats, normalize_features, FEATURE_COLUMNS
)
from src.data.temporal_window import create_temporal_windows_with_labels
from src.data.failure_labels import create_failure_labels, create_predictive_failure_labels
from src.data.degradation_simulation import augment_dataset_with_degradation
from src.utils.config_loader import load_config


def load_sequence_data(seq_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load SLAM metrics and pose errors for a sequence."""
    metrics = pd.read_csv(os.path.join(seq_dir, 'slam_metrics.csv'))
    errors = pd.read_csv(os.path.join(seq_dir, 'pose_errors.csv'))
    return metrics, errors


def build_dataset(config: dict = None) -> Dict:
    """Build the complete reliability dataset.

    Returns:
        Dict with keys: 'train', 'val', 'test', each containing
        {'X': tensor, 'y_error': array, 'y_failure': array,
         'degradation_type': list, 'norm_stats': dict}
    """
    if config is None:
        config = load_config()

    data_dir = config['paths']['slam_metrics_dir']
    sequences = config['dataset']['euroc_sequences']
    window_size = config['temporal']['window_size']
    error_threshold = config['failure']['pose_error_threshold']

    # Load all sequences
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

    # Split sequences: use most sequences for train, but ensure each split
    # has sequences with failures. With 5 sequences, use 3 train / 1 val / 1 test
    # but assign higher-difficulty sequences across splits.
    n_seq = len(all_sequence_data)
    if n_seq >= 5:
        # Sort by sequence name to get consistent difficulty ordering
        # MH_01(easy) to MH_05(hard) - spread difficult ones across splits
        train_indices = [0, 1, 2]  # MH_01, MH_02, MH_03 (+ augmentation adds failures)
        val_indices = [3]          # MH_04 (harder, has failures)
        test_indices = [4]         # MH_05 (hardest, has failures)
    elif n_seq == 1:
        # Single sequence: reuse for all splits (validation/debugging only)
        print("  Note: single sequence mode — using same data for train/val/test")
        train_indices = [0]
        val_indices = [0]
        test_indices = [0]
    else:
        # Fallback: simple split for 2-4 sequences
        rng = np.random.RandomState(42)
        indices = rng.permutation(n_seq)
        n_train = max(1, int(n_seq * config['dataset']['train_split']))
        n_val = max(1, int(n_seq * config['dataset']['val_split']))
        train_indices = indices[:n_train].tolist()
        val_indices = indices[n_train:n_train + n_val].tolist()
        test_indices = indices[n_train + n_val:].tolist()
        if not test_indices:
            test_indices = val_indices[-1:]

    splits = {
        'train': [all_sequence_data[i] for i in train_indices],
        'val': [all_sequence_data[i] for i in val_indices],
        'test': [all_sequence_data[i] for i in test_indices],
    }

    print(f"\nSplit: train={[s[0] for s in splits['train']]}, "
          f"val={[s[0] for s in splits['val']]}, "
          f"test={[s[0] for s in splits['test']]}")

    # Compute normalization stats on training data only
    train_features_list = []
    for seq_name, metrics, errors in splits['train']:
        features = extract_features(metrics)
        train_features_list.append(features)
    all_train_features = np.concatenate(train_features_list)
    norm_stats = compute_normalization_stats(all_train_features)

    # Process each split
    dataset = {'norm_stats': norm_stats, 'feature_names': FEATURE_COLUMNS,
               'window_size': window_size}

    for split_name, split_data in splits.items():
        X_list, y_error_list, y_failure_list, deg_types_list = [], [], [], []

        for seq_name, metrics, errors in split_data:
            features = extract_features(metrics)
            features_norm = normalize_features(features, norm_stats)
            pose_errors = errors['pose_error'].values
            failure_labels = create_failure_labels(
                errors, metrics, error_threshold=error_threshold)
            # Predictive labels (predict ~1 second ahead at 20fps)
            pred_failure = create_predictive_failure_labels(failure_labels, prediction_horizon=20)

            # Apply degradation augmentation (training only)
            if split_name == 'train':
                aug_features, aug_errors, aug_types = augment_dataset_with_degradation(
                    features_norm, pose_errors, n_augmented=2, seed=hash(seq_name) % 10000)
                aug_failure = (aug_errors > error_threshold).astype(int)
            else:
                aug_features = features_norm
                aug_errors = pose_errors
                aug_failure = pred_failure
                aug_types = ['none'] * len(features)

            # Create temporal windows
            X, y_err, y_fail = create_temporal_windows_with_labels(
                aug_features, aug_errors, aug_failure, window_size=window_size)

            X_list.append(X)
            y_error_list.append(y_err)
            y_failure_list.append(y_fail)
            # Trim degradation types to match windowed length
            deg_types_list.extend(aug_types[window_size - 1:len(aug_types)])

        dataset[split_name] = {
            'X': np.concatenate(X_list),
            'y_error': np.concatenate(y_error_list),
            'y_failure': np.concatenate(y_failure_list),
        }

        n = len(dataset[split_name]['X'])
        n_fail = dataset[split_name]['y_failure'].sum()
        print(f"  {split_name}: {n} samples, {int(n_fail)} failures ({100*n_fail/max(n,1):.1f}%)")

    return dataset


def save_dataset(dataset: Dict, output_path: str):
    """Save dataset to pickle file."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Dataset saved to {output_path}")


def load_dataset(path: str) -> Dict:
    """Load dataset from pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def main():
    config = load_config()
    print("Building reliability dataset...")
    dataset = build_dataset(config)

    output_path = os.path.join(config['paths']['results_dir'], 'train_dataset.pkl')
    save_dataset(dataset, output_path)

    print(f"\nDataset summary:")
    for split in ['train', 'val', 'test']:
        X = dataset[split]['X']
        print(f"  {split}: X={X.shape}, "
              f"y_error range=[{dataset[split]['y_error'].min():.3f}, {dataset[split]['y_error'].max():.3f}]")


if __name__ == '__main__':
    main()
