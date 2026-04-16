"""
Generate synthetic SLAM metrics data for pipeline testing.

Simulates realistic SLAM internal signals with correlated failure patterns:
- Normal operation: stable features, low reprojection error
- Degraded operation: fewer features, higher errors, leading to failure
- Failure events: tracking lost, high pose error

Produces data mimicking 5 EuRoC MH sequences.
"""

import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.utils.config_loader import load_config


def generate_sequence(seq_name: str, duration_s: float = 120.0,
                      fps: float = 20.0, difficulty: float = 0.5,
                      seed: int = 42) -> tuple:
    """Generate a synthetic SLAM metrics sequence.

    Args:
        seq_name: Sequence identifier (e.g., 'MH_01')
        duration_s: Duration in seconds
        fps: Frame rate
        difficulty: 0.0 (easy) to 1.0 (hard), controls failure frequency
        seed: Random seed

    Returns:
        (slam_metrics_df, pose_errors_df, ground_truth_df, estimated_df)
    """
    rng = np.random.RandomState(seed)
    n_frames = int(duration_s * fps)
    timestamps = np.arange(n_frames) / fps

    # Generate underlying "health" signal (smooth, with degradation periods)
    health = np.ones(n_frames)
    n_degradations = int(3 + difficulty * 7)
    for _ in range(n_degradations):
        center = rng.randint(50, n_frames - 50)
        width = rng.randint(20, 80)
        severity = rng.uniform(0.2, 1.0) * difficulty
        start = max(0, center - width // 2)
        end = min(n_frames, center + width // 2)
        degradation = severity * np.exp(-0.5 * ((np.arange(start, end) - center) / (width / 4)) ** 2)
        health[start:end] -= degradation

    health = np.clip(health, 0.0, 1.0)

    # Feature count: 80-200 normally, drops during degradation
    base_features = 150
    feature_count = (base_features * health + rng.normal(0, 10, n_frames)).astype(int)
    feature_count = np.clip(feature_count, 5, 300)

    # Feature tracking ratio: high when healthy
    feature_tracking_ratio = 0.85 * health + 0.1 + rng.normal(0, 0.05, n_frames)
    feature_tracking_ratio = np.clip(feature_tracking_ratio, 0.0, 1.0)

    # Reprojection error: low when healthy, spikes during degradation
    reproj_mean = 0.5 * (1 - health) + 0.3 + np.abs(rng.normal(0, 0.1, n_frames))
    reproj_std = 0.3 * (1 - health) + 0.1 + np.abs(rng.normal(0, 0.05, n_frames))

    # IMU residual norm
    imu_residual = 0.05 + 0.5 * (1 - health) + np.abs(rng.normal(0, 0.02, n_frames))

    # Camera motion magnitude (smooth trajectory with occasional fast motion)
    base_motion = 0.05 + 0.02 * np.sin(2 * np.pi * timestamps / 30)
    motion_spikes = np.zeros(n_frames)
    for _ in range(int(5 * difficulty)):
        idx = rng.randint(0, n_frames)
        motion_spikes[idx:min(idx + 10, n_frames)] = rng.uniform(0.1, 0.3)
    camera_motion = base_motion + motion_spikes + np.abs(rng.normal(0, 0.01, n_frames))

    # Tracking length (cumulative, resets on failure)
    tracking_length = np.zeros(n_frames)
    length = 0
    for i in range(n_frames):
        if health[i] < 0.2:
            length = 0
        else:
            length += 1.0 / fps
        tracking_length[i] = length

    # Tracking state: lost when health is very low
    tracking_state = (health > 0.15).astype(int)

    # Pose error: correlated with health - scaled so degraded regions exceed 0.3m
    # Base error inversely proportional to health
    pose_error = 0.1 / (health + 0.02) + np.abs(rng.normal(0, 0.03, n_frames))
    # Substantial spikes during any degradation
    degraded_mask = health < 0.6
    pose_error[degraded_mask] += 0.5 * (1 - health[degraded_mask]) ** 0.5
    pose_error = np.clip(pose_error, 0.01, 5.0)

    # Generate synthetic ground truth trajectory (smooth 3D path)
    gt_px = np.cumsum(0.05 * np.cos(2 * np.pi * timestamps / 60) + rng.normal(0, 0.001, n_frames))
    gt_py = np.cumsum(0.05 * np.sin(2 * np.pi * timestamps / 45) + rng.normal(0, 0.001, n_frames))
    gt_pz = 1.5 + 0.3 * np.sin(2 * np.pi * timestamps / 90)
    gt_qw = np.ones(n_frames)
    gt_qx = np.zeros(n_frames)
    gt_qy = np.zeros(n_frames)
    gt_qz = 0.01 * np.sin(2 * np.pi * timestamps / 30)
    # Normalize quaternions
    qnorm = np.sqrt(gt_qw**2 + gt_qx**2 + gt_qy**2 + gt_qz**2)
    gt_qw /= qnorm
    gt_qz /= qnorm

    # Estimated trajectory: ground truth + noise scaled by health
    noise_scale = 0.01 / (health + 0.1)
    est_px = gt_px + rng.normal(0, 1, n_frames) * noise_scale
    est_py = gt_py + rng.normal(0, 1, n_frames) * noise_scale
    est_pz = gt_pz + rng.normal(0, 1, n_frames) * noise_scale
    est_qw = gt_qw + rng.normal(0, 0.001, n_frames) * noise_scale
    est_qx = gt_qx + rng.normal(0, 0.001, n_frames) * noise_scale
    est_qy = gt_qy + rng.normal(0, 0.001, n_frames) * noise_scale
    est_qz = gt_qz + rng.normal(0, 0.001, n_frames) * noise_scale

    # Build DataFrames
    slam_metrics = pd.DataFrame({
        'timestamp': timestamps,
        'feature_count': feature_count,
        'feature_tracking_ratio': feature_tracking_ratio,
        'reprojection_error_mean': reproj_mean,
        'reprojection_error_std': reproj_std,
        'imu_residual_norm': imu_residual,
        'optimization_iterations': np.clip(rng.poisson(5, n_frames) + (1 - health) * 3, 1, 20).astype(int),
        'tracking_time_ms': 15 + 10 * (1 - health) + np.abs(rng.normal(0, 2, n_frames)),
        'tracking_state': tracking_state,
        'camera_motion_magnitude': camera_motion,
        'tracking_length': tracking_length,
    })

    pose_errors = pd.DataFrame({
        'timestamp': timestamps,
        'pose_error': pose_error,
        'rotation_error_deg': pose_error * 10 + np.abs(rng.normal(0, 0.5, n_frames)),
    })

    ground_truth = pd.DataFrame({
        'timestamp': timestamps,
        'px': gt_px, 'py': gt_py, 'pz': gt_pz,
        'qw': gt_qw, 'qx': gt_qx, 'qy': gt_qy, 'qz': gt_qz,
    })

    estimated = pd.DataFrame({
        'timestamp': timestamps,
        'px': est_px, 'py': est_py, 'pz': est_pz,
        'qw': est_qw, 'qx': est_qx, 'qy': est_qy, 'qz': est_qz,
    })

    return slam_metrics, pose_errors, ground_truth, estimated


def main():
    config = load_config()
    output_dir = config['paths']['slam_metrics_dir']

    sequences = {
        'MH_01': {'duration': 120, 'difficulty': 0.2, 'seed': 1},
        'MH_02': {'duration': 130, 'difficulty': 0.3, 'seed': 2},
        'MH_03': {'duration': 110, 'difficulty': 0.5, 'seed': 3},
        'MH_04': {'duration': 100, 'difficulty': 0.7, 'seed': 4},
        'MH_05': {'duration': 115, 'difficulty': 0.9, 'seed': 5},
    }

    for seq_name, params in sequences.items():
        print(f"Generating {seq_name} (difficulty={params['difficulty']})...")
        seq_dir = os.path.join(output_dir, seq_name)
        os.makedirs(seq_dir, exist_ok=True)

        metrics, errors, gt, est = generate_sequence(
            seq_name,
            duration_s=params['duration'],
            fps=20.0,
            difficulty=params['difficulty'],
            seed=params['seed']
        )

        metrics.to_csv(os.path.join(seq_dir, 'slam_metrics.csv'), index=False)
        errors.to_csv(os.path.join(seq_dir, 'pose_errors.csv'), index=False)
        gt.to_csv(os.path.join(seq_dir, 'groundtruth.csv'), index=False)
        est.to_csv(os.path.join(seq_dir, 'estimated.csv'), index=False)

        n_failures = (errors['pose_error'] > 0.3).sum()
        print(f"  {len(metrics)} frames, {n_failures} failure frames "
              f"({100*n_failures/len(metrics):.1f}%)")

    print(f"\nSynthetic data saved to {output_dir}")


if __name__ == '__main__':
    main()
