"""
Task 1.4: Ground Truth Alignment

Load ground truth trajectory and SLAM estimated trajectory,
align them (SE3 Umeyama alignment), compute pose_error per frame.

Supports EuRoC MAV and TUM-VI ground truth formats.

Output: CSV with [timestamp, pose_error]
"""

import numpy as np
import pandas as pd
import argparse
import os
from scipy.spatial.transform import Rotation


def load_euroc_groundtruth(gt_path: str) -> pd.DataFrame:
    """Load EuRoC MAV ground truth CSV.

    Format: timestamp, p_x, p_y, p_z, q_w, q_x, q_y, q_z, v_x, v_y, v_z, ...
    """
    df = pd.read_csv(gt_path, comment='#')
    # EuRoC has header or headerless format
    if df.shape[1] >= 8:
        cols = ['timestamp', 'px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz']
        extra_cols = [f'col_{i}' for i in range(8, df.shape[1])]
        df.columns = cols + extra_cols
    df['timestamp'] = df['timestamp'].astype(float)
    # Convert nanoseconds to seconds if needed
    if df['timestamp'].iloc[0] > 1e15:
        df['timestamp'] = df['timestamp'] / 1e9
    return df[['timestamp', 'px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz']]


def load_tum_trajectory(traj_path: str) -> pd.DataFrame:
    """Load TUM-format trajectory: timestamp tx ty tz qx qy qz qw"""
    df = pd.read_csv(traj_path, sep=' ', comment='#', header=None)
    df.columns = ['timestamp', 'px', 'py', 'pz', 'qx', 'qy', 'qz', 'qw']
    return df


def associate_trajectories(gt: pd.DataFrame, est: pd.DataFrame,
                           max_diff: float = 0.01) -> pd.DataFrame:
    """Associate ground truth and estimated poses by nearest timestamp.

    Args:
        gt: Ground truth dataframe with 'timestamp' column
        est: Estimated trajectory dataframe with 'timestamp' column
        max_diff: Maximum timestamp difference in seconds

    Returns:
        DataFrame with matched pairs
    """
    matches = []
    gt_times = gt['timestamp'].values
    est_times = est['timestamp'].values

    for i, et in enumerate(est_times):
        diffs = np.abs(gt_times - et)
        min_idx = np.argmin(diffs)
        if diffs[min_idx] <= max_diff:
            matches.append({
                'timestamp': et,
                'gt_px': gt.iloc[min_idx]['px'],
                'gt_py': gt.iloc[min_idx]['py'],
                'gt_pz': gt.iloc[min_idx]['pz'],
                'gt_qw': gt.iloc[min_idx]['qw'],
                'gt_qx': gt.iloc[min_idx]['qx'],
                'gt_qy': gt.iloc[min_idx]['qy'],
                'gt_qz': gt.iloc[min_idx]['qz'],
                'est_px': est.iloc[i]['px'],
                'est_py': est.iloc[i]['py'],
                'est_pz': est.iloc[i]['pz'],
                'est_qw': est.iloc[i]['qw'],
                'est_qx': est.iloc[i]['qx'],
                'est_qy': est.iloc[i]['qy'],
                'est_qz': est.iloc[i]['qz'],
            })

    return pd.DataFrame(matches)


def umeyama_alignment(source: np.ndarray, target: np.ndarray):
    """Compute SE3 Umeyama alignment from source to target point clouds.

    Args:
        source: Nx3 estimated positions
        target: Nx3 ground truth positions

    Returns:
        s (scale), R (3x3 rotation), t (3x1 translation)
    """
    assert source.shape == target.shape
    n = source.shape[0]

    mu_s = source.mean(axis=0)
    mu_t = target.mean(axis=0)

    source_centered = source - mu_s
    target_centered = target - mu_t

    sigma_s = np.sum(source_centered ** 2) / n
    cov = target_centered.T @ source_centered / n

    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / sigma_s
    t = mu_t - s * R @ mu_s

    return s, R, t


def compute_pose_errors(matched: pd.DataFrame, align: bool = True) -> pd.DataFrame:
    """Compute per-frame pose error (translation + rotation).

    Args:
        matched: DataFrame from associate_trajectories
        align: Whether to apply Umeyama alignment first

    Returns:
        DataFrame with [timestamp, pose_error, rotation_error_deg]
    """
    gt_pos = matched[['gt_px', 'gt_py', 'gt_pz']].values
    est_pos = matched[['est_px', 'est_py', 'est_pz']].values

    if align and len(gt_pos) >= 3:
        s, R, t = umeyama_alignment(est_pos, gt_pos)
        est_pos_aligned = (s * (R @ est_pos.T).T + t)
    else:
        est_pos_aligned = est_pos

    # Translation error (Euclidean distance)
    trans_errors = np.linalg.norm(gt_pos - est_pos_aligned, axis=1)

    # Rotation error
    rot_errors = []
    for _, row in matched.iterrows():
        gt_q = [row['gt_qx'], row['gt_qy'], row['gt_qz'], row['gt_qw']]
        est_q = [row['est_qx'], row['est_qy'], row['est_qz'], row['est_qw']]
        try:
            R_gt = Rotation.from_quat(gt_q)
            R_est = Rotation.from_quat(est_q)
            R_diff = R_gt.inv() * R_est
            angle = R_diff.magnitude()
            rot_errors.append(np.degrees(angle))
        except Exception:
            rot_errors.append(0.0)

    result = pd.DataFrame({
        'timestamp': matched['timestamp'].values,
        'pose_error': trans_errors,
        'rotation_error_deg': rot_errors
    })
    return result


def main():
    parser = argparse.ArgumentParser(description='Ground Truth Alignment (Task 1.4)')
    parser.add_argument('--gt', type=str, required=True, help='Ground truth trajectory file')
    parser.add_argument('--est', type=str, required=True, help='Estimated trajectory file')
    parser.add_argument('--gt-format', type=str, choices=['euroc', 'tum'], default='euroc')
    parser.add_argument('--est-format', type=str, choices=['euroc', 'tum'], default='tum')
    parser.add_argument('--output', type=str, default='slam_metrics_dataset/pose_errors.csv')
    parser.add_argument('--no-align', action='store_true', help='Skip Umeyama alignment')
    parser.add_argument('--max-diff', type=float, default=0.01,
                        help='Max timestamp difference for association (seconds)')
    args = parser.parse_args()

    # Load trajectories
    if args.gt_format == 'euroc':
        gt = load_euroc_groundtruth(args.gt)
    else:
        gt = load_tum_trajectory(args.gt)

    if args.est_format == 'euroc':
        est = load_euroc_groundtruth(args.est)
    else:
        est = load_tum_trajectory(args.est)

    print(f"Ground truth: {len(gt)} poses")
    print(f"Estimated: {len(est)} poses")

    # Associate
    matched = associate_trajectories(gt, est, max_diff=args.max_diff)
    print(f"Matched pairs: {len(matched)}")

    if len(matched) == 0:
        print("ERROR: No matched pairs found. Check timestamp formats.")
        return

    # Compute errors
    errors = compute_pose_errors(matched, align=not args.no_align)

    # Save
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    errors.to_csv(args.output, index=False)
    print(f"Pose errors saved to {args.output}")
    print(f"  Mean ATE: {errors['pose_error'].mean():.4f} m")
    print(f"  RMSE ATE: {np.sqrt((errors['pose_error']**2).mean()):.4f} m")
    print(f"  Mean rotation error: {errors['rotation_error_deg'].mean():.2f} deg")


if __name__ == '__main__':
    main()
