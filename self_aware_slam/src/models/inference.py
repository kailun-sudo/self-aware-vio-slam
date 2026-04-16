"""
Offline and streaming inference utilities for unified ORB-SLAM with self-awareness.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.dataset_builder import load_dataset
from src.data.feature_engineering import FEATURE_COLUMNS, extract_features, normalize_features
from src.models.failure_predictor import build_model
from src.utils.config_loader import load_config

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def _create_inference_windows(features: np.ndarray, window_size: int) -> np.ndarray:
    """Create temporal windows without requiring labels."""
    n_samples = len(features) - window_size + 1
    if n_samples <= 0:
        raise ValueError(
            f"Metrics file has {len(features)} rows, but window_size={window_size}. "
            "Need at least one full temporal window."
        )

    n_features = features.shape[1]
    windows = np.zeros((n_samples, window_size, n_features), dtype=np.float32)
    for idx in range(n_samples):
        windows[idx] = features[idx:idx + window_size]
    return windows


def _resolve_runtime_artifacts(config: Dict,
                               checkpoint_path: Optional[str],
                               dataset_path: Optional[str]) -> Dict[str, str]:
    """Resolve checkpoint and dataset-stat paths."""
    model_type = config['model']['type']
    resolved_checkpoint = checkpoint_path or os.path.join(
        config['paths']['model_save_dir'],
        f'{model_type}_failure_predictor.pt',
    )
    resolved_dataset = dataset_path or os.path.join(
        config['paths']['results_dir'],
        'inference_stats.json',
    )

    if not os.path.isabs(resolved_checkpoint):
        resolved_checkpoint = os.path.join(PROJECT_ROOT, resolved_checkpoint)
    if not os.path.isabs(resolved_dataset):
        resolved_dataset = os.path.join(PROJECT_ROOT, resolved_dataset)

    return {
        'checkpoint_path': resolved_checkpoint,
        'dataset_path': resolved_dataset,
    }


def _load_runtime_stats(dataset_path: str) -> Dict:
    """Load either the full training cache or a lightweight inference stats file."""
    _, extension = os.path.splitext(dataset_path)
    if extension.lower() == '.json':
        with open(dataset_path, 'r', encoding='utf-8') as handle:
            data = json.load(handle)
        return {
            'norm_stats': {
                'mean': np.asarray(data['norm_stats']['mean'], dtype=np.float32),
                'std': np.asarray(data['norm_stats']['std'], dtype=np.float32),
            },
            'feature_names': data.get('feature_names', FEATURE_COLUMNS),
            'window_size': data['window_size'],
        }

    if extension.lower() in {'.pkl', '.pickle'}:
        dataset = load_dataset(dataset_path)
        return {
            'norm_stats': dataset['norm_stats'],
            'feature_names': dataset.get('feature_names', FEATURE_COLUMNS),
            'window_size': dataset.get('window_size'),
        }

    raise ValueError(f'Unsupported dataset stats format: {dataset_path}')


class OnlinePredictorRuntime:
    """Keep the self-aware model loaded for row-by-row online prediction."""

    def __init__(self,
                 config_path: Optional[str] = None,
                 checkpoint_path: Optional[str] = None,
                 dataset_path: Optional[str] = None):
        config = load_config(config_path)
        artifacts = _resolve_runtime_artifacts(config, checkpoint_path, dataset_path)

        checkpoint = torch.load(
            artifacts['checkpoint_path'],
            map_location='cpu',
            weights_only=False,
        )
        checkpoint_config = checkpoint.get('config', config)
        runtime_stats = _load_runtime_stats(artifacts['dataset_path'])

        self.norm_stats = runtime_stats['norm_stats']
        self.feature_names = runtime_stats.get('feature_names', FEATURE_COLUMNS)
        self.window_size = runtime_stats.get('window_size', checkpoint_config['temporal']['window_size'])
        self.buffer: List[Dict] = []

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else
            'mps' if torch.backends.mps.is_available() else
            'cpu'
        )
        self.model = build_model(checkpoint_config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def process_row(self, metrics_row: Dict) -> Optional[Dict]:
        """Push one metrics row and return a prediction once enough context exists."""
        self.buffer.append(dict(metrics_row))
        if len(self.buffer) < self.window_size:
            return None

        window_df = pd.DataFrame(self.buffer[-self.window_size:])
        features = extract_features(window_df, feature_columns=self.feature_names)
        features_norm = normalize_features(features, self.norm_stats)
        window = np.expand_dims(features_norm[-self.window_size:], axis=0).astype(np.float32)

        with torch.no_grad():
            tensor_window = torch.FloatTensor(window).to(self.device)
            failure_prob, pred_error = self.model(tensor_window)

        failure_prob_value = float(failure_prob.squeeze().cpu().item())
        pred_error_value = float(pred_error.squeeze().cpu().item())
        timestamp = float(metrics_row.get('timestamp', len(self.buffer) - 1))
        frame_id = int(metrics_row.get('frame_id', len(self.buffer) - 1))

        return {
            'timestamp': timestamp,
            'frame_id': frame_id,
            'failure_probability': failure_prob_value,
            'confidence_score': 1.0 - failure_prob_value,
            'predicted_pose_error': pred_error_value,
            'predicted_localization_reliability': (1.0 - failure_prob_value) / (1.0 + pred_error_value),
            'predicted_failure': int(failure_prob_value >= 0.5),
            'window_size': self.window_size,
            'frames_seen': len(self.buffer),
        }


def run_inference(
    metrics_path: str,
    output_path: str,
    config_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    dataset_path: Optional[str] = None,
    pose_errors_path: Optional[str] = None,
) -> pd.DataFrame:
    """Run failure prediction on exported SLAM metrics."""
    config = load_config(config_path)
    artifacts = _resolve_runtime_artifacts(config, checkpoint_path, dataset_path)

    checkpoint = torch.load(
        artifacts['checkpoint_path'],
        map_location='cpu',
        weights_only=False,
    )
    checkpoint_config = checkpoint.get('config', config)
    runtime_stats = _load_runtime_stats(artifacts['dataset_path'])
    norm_stats = runtime_stats['norm_stats']
    feature_names = runtime_stats.get('feature_names', FEATURE_COLUMNS)
    window_size = runtime_stats.get('window_size', checkpoint_config['temporal']['window_size'])

    metrics_df = pd.read_csv(metrics_path)
    features = extract_features(metrics_df, feature_columns=feature_names)
    features_norm = normalize_features(features, norm_stats)
    windows = _create_inference_windows(features_norm, window_size)

    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )
    model = build_model(checkpoint_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad():
        tensor_windows = torch.FloatTensor(windows).to(device)
        failure_prob, pred_error = model(tensor_windows)

    failure_prob = failure_prob.squeeze(-1).cpu().numpy()
    pred_error = pred_error.squeeze(-1).cpu().numpy()
    aligned_rows = metrics_df.iloc[window_size - 1:].reset_index(drop=True)

    predictions = pd.DataFrame({
        'timestamp': aligned_rows['timestamp'] if 'timestamp' in aligned_rows else aligned_rows.index.astype(float),
        'frame_id': aligned_rows['frame_id'] if 'frame_id' in aligned_rows else aligned_rows.index,
        'failure_probability': failure_prob,
        'confidence_score': 1.0 - failure_prob,
        'predicted_pose_error': pred_error,
        'predicted_localization_reliability': (1.0 - failure_prob) / (1.0 + pred_error),
    })
    predictions['predicted_failure'] = (predictions['failure_probability'] >= 0.5).astype(int)

    if pose_errors_path and os.path.exists(pose_errors_path):
        pose_errors = pd.read_csv(pose_errors_path)
        pose_errors = pose_errors.iloc[window_size - 1:].reset_index(drop=True)
        if len(pose_errors) == len(predictions):
            predictions['actual_pose_error'] = pose_errors['pose_error']

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    predictions.to_csv(output_path, index=False)
    return predictions


def stream_inference(config_path: Optional[str] = None,
                     checkpoint_path: Optional[str] = None,
                     dataset_path: Optional[str] = None):
    """Read metrics rows from stdin and emit one JSON line per input row."""
    runtime = OnlinePredictorRuntime(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        dataset_path=dataset_path,
    )

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if payload.get('command') == 'close':
            print(json.dumps({'status': 'closed'}), flush=True)
            return

        prediction = runtime.process_row(payload)
        if prediction is None:
            print(
                json.dumps(
                    {
                        'status': 'warmup',
                        'frames_seen': len(runtime.buffer),
                        'required_window_size': runtime.window_size,
                    }
                ),
                flush=True,
            )
            continue

        prediction['status'] = 'prediction'
        print(json.dumps(prediction), flush=True)


def main():
    parser = argparse.ArgumentParser(description='Run self-aware SLAM inference on metrics CSV')
    parser.add_argument('--metrics', type=str, default=None, help='Path to slam_metrics.csv')
    parser.add_argument('--output', type=str, default=None, help='Path to output predictions CSV')
    parser.add_argument('--config', type=str, default=None, help='Path to config.yaml')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to trained model checkpoint')
    parser.add_argument('--dataset-stats', type=str, default=None,
                        help='Path to train_dataset.pkl containing normalization stats')
    parser.add_argument('--pose-errors', type=str, default=None,
                        help='Optional pose_errors.csv for side-by-side comparison')
    parser.add_argument('--stream', action='store_true',
                        help='Run as a streaming predictor that reads JSON metrics rows from stdin')
    args = parser.parse_args()

    if args.stream:
        stream_inference(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            dataset_path=args.dataset_stats,
        )
        return

    if not args.metrics or not args.output:
        parser.error('--metrics and --output are required unless --stream is set')

    predictions = run_inference(
        metrics_path=args.metrics,
        output_path=args.output,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset_stats,
        pose_errors_path=args.pose_errors,
    )

    print(f'Wrote {len(predictions)} predictions to {args.output}')
    print(
        f"Average confidence={predictions['confidence_score'].mean():.4f}, "
        f"average failure_probability={predictions['failure_probability'].mean():.4f}"
    )


if __name__ == '__main__':
    main()
