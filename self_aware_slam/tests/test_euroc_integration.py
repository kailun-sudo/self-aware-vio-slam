#!/usr/bin/env python3
"""
Integration test for the EuRoC + VINS-Fusion pipeline.

Creates mock EuRoC ground truth and VINS-Fusion output files,
then runs the full processing pipeline to verify:
  1. EuRoC GT parsing
  2. VINS trajectory parsing
  3. Trajectory association and alignment
  4. SLAM metrics generation
  5. Output CSV format compatibility with dataset_builder

Run:
    python tests/test_euroc_integration.py
"""

import os
import sys
import tempfile
import shutil
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd


def create_mock_euroc_gt(seq_dir, n_frames=500, dt=0.05):
    """Create a mock EuRoC ASL ground truth CSV."""
    gt_dir = os.path.join(seq_dir, 'mav0', 'state_groundtruth_estimate0')
    os.makedirs(gt_dir, exist_ok=True)

    t0 = 1403636579763555584  # typical EuRoC ns timestamp
    timestamps = t0 + np.arange(n_frames) * int(dt * 1e9)

    # Simple circular trajectory
    theta = np.linspace(0, 2 * np.pi, n_frames)
    px = 2.0 * np.cos(theta)
    py = 2.0 * np.sin(theta)
    pz = np.linspace(0, 1, n_frames)

    # Identity quaternion (w,x,y,z)
    qw = np.ones(n_frames)
    qx = np.zeros(n_frames)
    qy = np.zeros(n_frames)
    qz = np.zeros(n_frames)

    # Extra columns (velocity, biases) - 9 more columns
    extra = np.zeros((n_frames, 9))

    data = np.column_stack([
        timestamps, px, py, pz, qw, qx, qy, qz, extra
    ])

    gt_path = os.path.join(gt_dir, 'data.csv')
    with open(gt_path, 'w') as f:
        f.write('#timestamp [ns],p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],'
                'q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z [],'
                'v_RS_R_x [m s^-1],v_RS_R_y [m s^-1],v_RS_R_z [m s^-1],'
                'b_w_RS_S_x [rad s^-1],b_w_RS_S_y [rad s^-1],b_w_RS_S_z [rad s^-1],'
                'b_a_RS_S_x [m s^-2],b_a_RS_S_y [m s^-2],b_a_RS_S_z [m s^-2]\n')
        np.savetxt(f, data, delimiter=',', fmt='%.6f')

    return gt_path


def create_mock_vins_trajectory(output_path, n_frames=480, dt=0.05,
                                noise_std=0.02, t0_ns=1403636579763555584):
    """Create a mock VINS-Fusion trajectory in TUM format.

    Adds noise to the ground truth circular trajectory.
    Uses slightly fewer frames to simulate tracking gaps.
    """
    t0_s = t0_ns / 1e9
    timestamps = t0_s + np.arange(n_frames) * dt

    theta = np.linspace(0, 2 * np.pi * (n_frames / 500), n_frames)
    px = 2.0 * np.cos(theta) + np.random.randn(n_frames) * noise_std
    py = 2.0 * np.sin(theta) + np.random.randn(n_frames) * noise_std
    pz = np.linspace(0, 1 * (n_frames / 500), n_frames) + np.random.randn(n_frames) * noise_std

    # TUM format quaternion order: qx, qy, qz, qw
    qx = np.zeros(n_frames)
    qy = np.zeros(n_frames)
    qz = np.zeros(n_frames)
    qw = np.ones(n_frames)

    data = np.column_stack([timestamps, px, py, pz, qx, qy, qz, qw])

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    np.savetxt(output_path, data, fmt='%.9f')
    return output_path


def create_mock_vins_log(output_path, n_entries=100):
    """Create a mock VINS-Fusion log with feature counts."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        for i in range(n_entries):
            t = 1403636579.763 + i * 0.25
            n_feat = np.random.randint(80, 200)
            reproj = np.random.uniform(0.3, 1.5)
            f.write(f"[{t:.3f}] time: {t:.6f} feature num: {n_feat} "
                    f"reprojection error: {reproj:.4f} iteration: 5\n")
    return output_path


def test_euroc_gt_parser(tmp_dir):
    """Test parsing EuRoC ground truth."""
    from src.euroc.euroc_gt_parser import parse_euroc_gt

    seq_dir = os.path.join(tmp_dir, 'MH_01_easy')
    create_mock_euroc_gt(seq_dir)

    gt = parse_euroc_gt(seq_dir)

    assert len(gt) == 500, f"Expected 500 rows, got {len(gt)}"
    assert list(gt.columns) == ['timestamp', 'px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz']
    assert gt['timestamp'].iloc[0] < 1e12, "Timestamp should be in seconds"
    print("  [PASS] euroc_gt_parser")
    return gt


def test_vins_trajectory_parser(tmp_dir):
    """Test parsing VINS-Fusion trajectory."""
    from src.euroc.vins_output_parser import parse_vins_trajectory

    traj_path = os.path.join(tmp_dir, 'vins_output', 'MH_01_easy',
                             'vins_result_no_loop.csv')
    create_mock_vins_trajectory(traj_path)

    est = parse_vins_trajectory(traj_path)

    assert len(est) == 480, f"Expected 480 rows, got {len(est)}"
    assert list(est.columns) == ['timestamp', 'px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz']
    print("  [PASS] vins_trajectory_parser")
    return est


def test_vins_log_parser(tmp_dir):
    """Test parsing VINS-Fusion log."""
    from src.euroc.vins_output_parser import parse_vins_log

    log_path = os.path.join(tmp_dir, 'vins_output', 'MH_01_easy', 'vins_log.txt')
    create_mock_vins_log(log_path)

    log_df = parse_vins_log(log_path)

    assert log_df is not None, "Log parser returned None"
    assert len(log_df) > 0, "No log entries parsed"
    assert 'feature_count' in log_df.columns
    print(f"  [PASS] vins_log_parser ({len(log_df)} entries)")
    return log_df


def test_slam_metrics_builder(tmp_dir):
    """Test building slam_metrics from trajectory."""
    from src.euroc.vins_output_parser import (
        parse_vins_trajectory, parse_vins_log,
        build_slam_metrics_from_trajectory
    )

    traj_path = os.path.join(tmp_dir, 'vins_output', 'MH_01_easy',
                             'vins_result_no_loop.csv')
    log_path = os.path.join(tmp_dir, 'vins_output', 'MH_01_easy', 'vins_log.txt')

    est = parse_vins_trajectory(traj_path)
    log_df = parse_vins_log(log_path)

    metrics = build_slam_metrics_from_trajectory(est, log_df)

    expected_cols = [
        'timestamp', 'feature_count', 'feature_tracking_ratio',
        'reprojection_error_mean', 'reprojection_error_std',
        'imu_residual_norm', 'optimization_iterations',
        'tracking_time_ms', 'tracking_state',
        'camera_motion_magnitude', 'tracking_length'
    ]
    for col in expected_cols:
        assert col in metrics.columns, f"Missing column: {col}"

    assert len(metrics) == len(est)
    assert metrics['camera_motion_magnitude'].iloc[1:].sum() > 0, \
        "Motion should be non-zero"
    assert (metrics['tracking_state'] == 1).all(), \
        "All frames should be tracking (no gaps in mock data)"
    print(f"  [PASS] slam_metrics_builder ({len(metrics)} frames, "
          f"{len(expected_cols)} columns)")


def test_full_sequence_processing(tmp_dir):
    """Test end-to-end sequence processing."""
    from src.euroc.process_sequence import process_sequence

    euroc_dir = os.path.join(tmp_dir, 'MH_01_easy')
    vins_traj = os.path.join(tmp_dir, 'vins_output', 'MH_01_easy',
                             'vins_result_no_loop.csv')
    vins_log = os.path.join(tmp_dir, 'vins_output', 'MH_01_easy', 'vins_log.txt')
    output_dir = os.path.join(tmp_dir, 'slam_metrics_dataset')

    out = process_sequence(
        euroc_seq_dir=euroc_dir,
        vins_trajectory_path=vins_traj,
        sequence_name='MH_01',
        output_base_dir=output_dir,
        vins_log_path=vins_log,
    )

    # Verify all 4 output files exist
    for fname in ['slam_metrics.csv', 'pose_errors.csv',
                  'groundtruth.csv', 'estimated.csv']:
        fpath = os.path.join(out, fname)
        assert os.path.isfile(fpath), f"Missing: {fpath}"
        df = pd.read_csv(fpath)
        assert len(df) > 0, f"Empty: {fpath}"

    print(f"  [PASS] full sequence processing -> {out}")
    return out


def test_dataset_builder_compatibility(tmp_dir):
    """Test that processed real data is compatible with dataset_builder."""
    from src.data.feature_engineering import extract_features, FEATURE_COLUMNS

    output_dir = os.path.join(tmp_dir, 'slam_metrics_dataset', 'MH_01')
    metrics = pd.read_csv(os.path.join(output_dir, 'slam_metrics.csv'))
    errors = pd.read_csv(os.path.join(output_dir, 'pose_errors.csv'))

    # Verify row-count alignment (the core invariant)
    assert len(metrics) == len(errors), (
        f"slam_metrics ({len(metrics)}) and pose_errors ({len(errors)}) "
        f"must have same row count")

    # Check feature extraction works
    features = extract_features(metrics)
    assert features.shape[1] == len(FEATURE_COLUMNS), \
        f"Expected {len(FEATURE_COLUMNS)} features, got {features.shape[1]}"
    assert not np.any(np.isnan(features)), "NaN in features"
    assert not np.any(np.isinf(features)), "Inf in features"

    # Check failure labels work
    from src.data.failure_labels import create_failure_labels
    labels = create_failure_labels(errors, metrics)
    assert len(labels) == len(metrics), "Label count mismatch"

    print(f"  [PASS] dataset_builder compatibility "
          f"(features shape={features.shape}, labels={len(labels)})")


def test_single_sequence_dataset_build(tmp_dir):
    """Test that dataset_builder works with only 1 sequence available."""
    from src.data.dataset_builder import build_dataset

    config = {
        'paths': {'slam_metrics_dir': os.path.join(tmp_dir, 'slam_metrics_dataset')},
        'dataset': {
            'euroc_sequences': ['MH_01'],
            'train_split': 0.70,
            'val_split': 0.15,
            'test_split': 0.15,
        },
        'features': {'names': [
            'feature_count', 'feature_tracking_ratio',
            'reprojection_error_mean', 'reprojection_error_std',
            'imu_residual_norm', 'camera_motion_magnitude', 'tracking_length',
        ], 'normalize': True},
        'temporal': {'window_size': 10},
        'failure': {'pose_error_threshold': 0.3, 'tracking_lost_is_failure': True},
    }

    dataset = build_dataset(config)

    for split in ['train', 'val', 'test']:
        assert split in dataset, f"Missing split: {split}"
        assert len(dataset[split]['X']) > 0, f"Empty {split} split"
        assert dataset[split]['X'].shape[2] == 7, f"Wrong feature dim in {split}"

    print(f"  [PASS] single-sequence dataset build "
          f"(train={len(dataset['train']['X'])}, "
          f"val={len(dataset['val']['X'])}, "
          f"test={len(dataset['test']['X'])})")


def test_gt_extra_nesting(tmp_dir):
    """Test that GT parser finds data.csv one extra directory level deep."""
    from src.euroc.euroc_gt_parser import parse_euroc_gt

    # Simulate: user pointed at parent dir, actual data is one level deeper
    wrapper_dir = os.path.join(tmp_dir, 'nested_test')
    inner_dir = os.path.join(wrapper_dir, 'MH_01_easy')
    create_mock_euroc_gt(inner_dir)

    gt = parse_euroc_gt(wrapper_dir)
    assert len(gt) == 500, f"Expected 500 rows from nested path, got {len(gt)}"
    print("  [PASS] gt_extra_nesting")


def test_log_parser_no_false_timestamps(tmp_dir):
    """Test that log parser doesn't match 't' inside words like 'count'."""
    from src.euroc.vins_output_parser import parse_vins_log

    log_path = os.path.join(tmp_dir, 'tricky_log.txt')
    os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)
    with open(log_path, 'w') as f:
        # Lines where 'count:' and 'point:' should NOT be parsed as timestamps
        f.write("feature point count: 150\n")
        f.write("constraint: 42\n")
        f.write("time: 1403636580.000 feature num: 120\n")
        f.write("point count: 95\n")
        f.write("time: 1403636580.050 feature num: 130\n")

    log_df = parse_vins_log(log_path)

    assert log_df is not None, "Log parser returned None"
    # Should only have 2 records (the two 'time:' lines), not 4+
    assert len(log_df) == 2, (
        f"Expected 2 log records, got {len(log_df)}. "
        f"Regex may be matching 't' inside 'count', 'point', etc."
    )
    # All timestamps should be in the EuRoC range, not small numbers like 150
    assert (log_df['timestamp'] > 1e9).all(), (
        f"Bogus timestamps found: {log_df['timestamp'].tolist()}")

    print(f"  [PASS] log_parser no false timestamps ({len(log_df)} records)")


def main():
    np.random.seed(42)
    tmp_dir = tempfile.mkdtemp(prefix='test_euroc_')

    try:
        print(f"Test directory: {tmp_dir}\n")

        # Create mock data
        print("Creating mock data...")
        create_mock_euroc_gt(os.path.join(tmp_dir, 'MH_01_easy'))
        create_mock_vins_trajectory(
            os.path.join(tmp_dir, 'vins_output', 'MH_01_easy',
                         'vins_result_no_loop.csv'))
        create_mock_vins_log(
            os.path.join(tmp_dir, 'vins_output', 'MH_01_easy', 'vins_log.txt'))

        print("\nRunning tests:")
        test_euroc_gt_parser(tmp_dir)
        test_vins_trajectory_parser(tmp_dir)
        test_vins_log_parser(tmp_dir)
        test_gt_extra_nesting(tmp_dir)
        test_log_parser_no_false_timestamps(tmp_dir)
        test_slam_metrics_builder(tmp_dir)
        test_full_sequence_processing(tmp_dir)
        test_dataset_builder_compatibility(tmp_dir)
        test_single_sequence_dataset_build(tmp_dir)

        print("\n" + "=" * 50)
        print("ALL TESTS PASSED")
        print("=" * 50)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
