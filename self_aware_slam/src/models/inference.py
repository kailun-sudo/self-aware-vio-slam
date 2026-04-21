"""
Offline and streaming inference utilities for unified Self-Aware VIO-SLAM.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.dataset_builder import load_dataset
from src.data.feature_engineering import (
    FEATURE_COLUMNS,
    extract_features,
    extract_learning_features,
    normalize_features,
)
from src.models.failure_predictor import build_model
from src.utils.config_loader import load_config

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PRIMARY_HEURISTIC_PRIORITY = (
    'heuristic_epipolar_error_risk',
    'heuristic_num_inliers_risk',
    'heuristic_inlier_ratio_risk',
    'heuristic_pose_residual_risk',
    'heuristic_imu_residual_risk',
)
EPIPOLAR_RISK_SCALE = 900.0
NUM_INLIERS_RISK_CENTER = 120.0
NUM_INLIERS_RISK_SCALE = 40.0
POSE_RESIDUAL_RISK_SCALE = 0.005


def _rank_normalize(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors='coerce')
    ranks = pd.Series(np.nan, index=series.index, dtype=float)
    mask = np.isfinite(values.to_numpy(dtype=float))
    if not np.any(mask):
        return ranks
    ranked = values[mask].rank(method='average', pct=True).astype(float)
    ranks.loc[mask] = ranked.to_numpy(dtype=float)
    return ranks


def _heuristic_score_definitions(metrics_df: pd.DataFrame) -> Dict[str, pd.Series]:
    scores: Dict[str, pd.Series] = {}
    if 'inlier_ratio' in metrics_df.columns:
        scores['heuristic_inlier_ratio_risk'] = 1.0 - pd.to_numeric(metrics_df['inlier_ratio'], errors='coerce')
    if 'pose_optimization_residual' in metrics_df.columns:
        residual = pd.to_numeric(metrics_df['pose_optimization_residual'], errors='coerce')
        scores['heuristic_pose_residual_risk'] = residual
        scores['heuristic_imu_residual_risk'] = residual
    if 'num_inliers' in metrics_df.columns:
        scores['heuristic_num_inliers_risk'] = -pd.to_numeric(metrics_df['num_inliers'], errors='coerce')
    if 'mean_epipolar_error' in metrics_df.columns:
        scores['heuristic_epipolar_error_risk'] = pd.to_numeric(metrics_df['mean_epipolar_error'], errors='coerce')
    return scores


def _sigmoid(values: pd.Series | np.ndarray) -> pd.Series:
    array = pd.to_numeric(values, errors='coerce').to_numpy(dtype=float) if isinstance(values, pd.Series) else np.asarray(values, dtype=float)
    clipped = np.clip(array, -60.0, 60.0)
    return pd.Series(1.0 / (1.0 + np.exp(-clipped)), index=values.index if isinstance(values, pd.Series) else None, dtype=float)


def _calibrated_heuristic_scores(raw_scores: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
    calibrated: Dict[str, pd.Series] = {}

    if 'heuristic_epipolar_error_risk' in raw_scores:
        epipolar = pd.to_numeric(raw_scores['heuristic_epipolar_error_risk'], errors='coerce').clip(lower=0.0)
        calibrated['heuristic_epipolar_error_risk_calibrated'] = 1.0 - np.exp(-epipolar / EPIPOLAR_RISK_SCALE)

    if 'heuristic_num_inliers_risk' in raw_scores:
        num_inliers = -pd.to_numeric(raw_scores['heuristic_num_inliers_risk'], errors='coerce')
        logits = -(num_inliers - NUM_INLIERS_RISK_CENTER) / NUM_INLIERS_RISK_SCALE
        calibrated['heuristic_num_inliers_risk_calibrated'] = _sigmoid(logits)

    if 'heuristic_inlier_ratio_risk' in raw_scores:
        calibrated['heuristic_inlier_ratio_risk_calibrated'] = (
            pd.to_numeric(raw_scores['heuristic_inlier_ratio_risk'], errors='coerce').clip(lower=0.0, upper=1.0)
        )

    if 'heuristic_pose_residual_risk' in raw_scores:
        residual = pd.to_numeric(raw_scores['heuristic_pose_residual_risk'], errors='coerce').clip(lower=0.0)
        calibrated['heuristic_pose_residual_risk_calibrated'] = (residual / POSE_RESIDUAL_RISK_SCALE).clip(lower=0.0, upper=1.0)

    if 'heuristic_imu_residual_risk' in raw_scores:
        residual = pd.to_numeric(raw_scores['heuristic_imu_residual_risk'], errors='coerce').clip(lower=0.0)
        calibrated['heuristic_imu_residual_risk_calibrated'] = (residual / POSE_RESIDUAL_RISK_SCALE).clip(lower=0.0, upper=1.0)

    return calibrated


def _primary_risk_columns(metrics_df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    heuristic_scores = _heuristic_score_definitions(metrics_df)
    columns = pd.DataFrame(index=metrics_df.index)
    for name, values in heuristic_scores.items():
        columns[name] = values.to_numpy(dtype=float)
        columns[f'{name}_rank'] = _rank_normalize(values).to_numpy(dtype=float)
    calibrated_scores = _calibrated_heuristic_scores(heuristic_scores)
    for name, values in calibrated_scores.items():
        columns[name] = values.to_numpy(dtype=float)

    primary_source = 'learned_failure_probability'
    if (
        'heuristic_epipolar_error_risk_calibrated' in columns.columns
        and 'heuristic_num_inliers_risk_calibrated' in columns.columns
    ):
        columns['primary_risk_score'] = 0.5 * (
            columns['heuristic_epipolar_error_risk_calibrated'] +
            columns['heuristic_num_inliers_risk_calibrated']
        )
        primary_source = 'heuristic_epipolar_num_inliers_absolute_fusion'
    else:
        calibrated_priority = (
            'heuristic_epipolar_error_risk_calibrated',
            'heuristic_num_inliers_risk_calibrated',
            'heuristic_inlier_ratio_risk_calibrated',
            'heuristic_pose_residual_risk_calibrated',
            'heuristic_imu_residual_risk_calibrated',
        )
        for candidate in calibrated_priority:
            if candidate in columns.columns and columns[candidate].notna().any():
                columns['primary_risk_score'] = columns[candidate]
                primary_source = candidate
                break

    if 'primary_risk_score' not in columns.columns:
        columns['primary_risk_score'] = np.nan

    columns['primary_confidence_score'] = 1.0 - columns['primary_risk_score']
    columns['primary_predicted_failure'] = (
        pd.to_numeric(columns['primary_risk_score'], errors='coerce').fillna(0.0) >= 0.5
    ).astype(int)
    columns['primary_risk_source'] = primary_source
    return columns, primary_source


def _extract_features_for_model(metrics_df: pd.DataFrame, feature_names: List[str], config: Dict) -> np.ndarray:
    """Support both legacy runtime features and v2 learning features."""
    if all(name in FEATURE_COLUMNS for name in feature_names):
        return extract_features(metrics_df, feature_columns=feature_names)
    return extract_learning_features(
        metrics_df,
        feature_columns=feature_names,
        rolling_window=config['features'].get('rolling_window', 5),
    )


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
        self.feature_config = checkpoint_config.get('features', {})
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
        features = _extract_features_for_model(window_df, self.feature_names, {'features': self.feature_config})
        features_norm = normalize_features(features, self.norm_stats)
        window = np.expand_dims(features_norm[-self.window_size:], axis=0).astype(np.float32)

        with torch.no_grad():
            tensor_window = torch.FloatTensor(window).to(self.device)
            failure_prob, pred_error = self.model(tensor_window)

        failure_prob_value = float(failure_prob.squeeze().cpu().item())
        pred_error_value = float(pred_error.squeeze().cpu().item())
        timestamp = float(metrics_row.get('timestamp', len(self.buffer) - 1))
        frame_id = int(metrics_row.get('frame_id', len(self.buffer) - 1))

        aligned_history_df = pd.DataFrame(self.buffer[self.window_size - 1:])
        primary_columns, primary_source = _primary_risk_columns(aligned_history_df)
        latest_primary = primary_columns.iloc[-1].to_dict() if len(primary_columns) else {}
        primary_risk = latest_primary.get('primary_risk_score', np.nan)
        primary_confidence = latest_primary.get('primary_confidence_score', np.nan)

        return {
            'timestamp': timestamp,
            'frame_id': frame_id,
            'failure_probability': failure_prob_value,
            'confidence_score': 1.0 - failure_prob_value,
            'predicted_pose_error': pred_error_value,
            'predicted_localization_reliability': (1.0 - failure_prob_value) / (1.0 + pred_error_value),
            'predicted_failure': int(failure_prob_value >= 0.5),
            'learned_failure_probability': failure_prob_value,
            'learned_confidence_score': 1.0 - failure_prob_value,
            'learned_predicted_pose_error': pred_error_value,
            'learned_predicted_localization_reliability': (1.0 - failure_prob_value) / (1.0 + pred_error_value),
            'learned_predicted_failure': int(failure_prob_value >= 0.5),
            'primary_risk_score': float(primary_risk) if pd.notna(primary_risk) else None,
            'primary_confidence_score': float(primary_confidence) if pd.notna(primary_confidence) else None,
            'primary_predicted_failure': int(latest_primary.get('primary_predicted_failure', 0)),
            'primary_localization_reliability': (
                float(primary_confidence) / (1.0 + pred_error_value) if pd.notna(primary_confidence) else None
            ),
            'primary_risk_source': primary_source,
            'window_size': self.window_size,
            'frames_seen': len(self.buffer),
            **{
                key: (float(value) if pd.notna(value) and not isinstance(value, str) else value)
                for key, value in latest_primary.items()
                if key.startswith('heuristic_') and key not in {'primary_risk_source'}
            },
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
    features = _extract_features_for_model(metrics_df, feature_names, checkpoint_config)
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
    predictions['learned_failure_probability'] = predictions['failure_probability']
    predictions['learned_confidence_score'] = predictions['confidence_score']
    predictions['learned_predicted_pose_error'] = predictions['predicted_pose_error']
    predictions['learned_predicted_localization_reliability'] = predictions['predicted_localization_reliability']
    predictions['learned_predicted_failure'] = predictions['predicted_failure']

    primary_columns, primary_source = _primary_risk_columns(aligned_rows)
    predictions = pd.concat([predictions, primary_columns], axis=1)
    predictions['primary_localization_reliability'] = (
        predictions['primary_confidence_score'] / (1.0 + predictions['predicted_pose_error'])
    )
    predictions['primary_risk_source'] = primary_source

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
        f"Average learned confidence={predictions['confidence_score'].mean():.4f}, "
        f"average learned failure_probability={predictions['failure_probability'].mean():.4f}"
    )
    if 'primary_risk_score' in predictions.columns:
        primary_source = predictions['primary_risk_source'].dropna().iloc[0] if predictions['primary_risk_source'].notna().any() else 'unknown'
        print(
            f"Average primary risk={predictions['primary_risk_score'].mean():.4f} "
            f"(source={primary_source})"
        )


if __name__ == '__main__':
    main()
