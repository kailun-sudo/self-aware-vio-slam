"""
Parse EuRoC MAV dataset ground truth from the ASL format.

EuRoC ASL ground truth CSV format (state_groundtruth_estimate0/data.csv):
  #timestamp [ns], p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m],
  q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z [],
  v_RS_R_x [m/s], v_RS_R_y [m/s], v_RS_R_z [m/s],
  b_w_RS_S_x [rad/s], b_w_RS_S_y [rad/s], b_w_RS_S_z [rad/s],
  b_a_RS_S_x [m/s^2], b_a_RS_S_y [m/s^2], b_a_RS_S_z [m/s^2]

Output: groundtruth.csv matching the project format:
  timestamp, px, py, pz, qw, qx, qy, qz
"""

import numpy as np
import pandas as pd
import glob
import os


# Standard EuRoC ASL directory layout
EUROC_GT_SUBPATH = "mav0/state_groundtruth_estimate0/data.csv"
EUROC_GT_SUBPATH_ALT = "state_groundtruth_estimate0/data.csv"


def find_euroc_gt_csv(euroc_seq_dir: str) -> str:
    """Locate the ground truth CSV within an EuRoC sequence directory.

    Handles:
      - Standard:  seq_dir/mav0/state_groundtruth_estimate0/data.csv
      - Flat:      seq_dir/state_groundtruth_estimate0/data.csv
      - One extra nesting: seq_dir/*/mav0/state_groundtruth_estimate0/data.csv
      - Direct CSV path
    """
    candidates = [
        os.path.join(euroc_seq_dir, EUROC_GT_SUBPATH),
        os.path.join(euroc_seq_dir, EUROC_GT_SUBPATH_ALT),
        euroc_seq_dir if euroc_seq_dir.endswith('.csv') else None,
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            return c

    # Glob fallback: one extra nesting level (e.g. double-extracted zips)
    glob_patterns = [
        os.path.join(euroc_seq_dir, '*', EUROC_GT_SUBPATH),
        os.path.join(euroc_seq_dir, '*', EUROC_GT_SUBPATH_ALT),
    ]
    for pattern in glob_patterns:
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[0]

    raise FileNotFoundError(
        f"Cannot find EuRoC ground truth CSV in {euroc_seq_dir}. "
        f"Expected at {EUROC_GT_SUBPATH} or {EUROC_GT_SUBPATH_ALT} "
        f"(also tried one subdirectory deeper)."
    )


def parse_euroc_gt(euroc_seq_dir: str) -> pd.DataFrame:
    """Parse EuRoC ASL ground truth into project-standard format.

    Args:
        euroc_seq_dir: Path to the EuRoC sequence root
            (e.g., /data/euroc/MH_01_easy)

    Returns:
        DataFrame with columns [timestamp, px, py, pz, qw, qx, qy, qz]
        timestamp in seconds (float64)
    """
    gt_csv = find_euroc_gt_csv(euroc_seq_dir)
    # EuRoC CSVs have a comment header line starting with #
    df = pd.read_csv(gt_csv, comment='#', header=None)

    # Select first 8 columns: timestamp, px, py, pz, qw, qx, qy, qz
    df = df.iloc[:, :8]
    df.columns = ['timestamp', 'px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz']

    # Convert nanosecond timestamps to seconds
    if df['timestamp'].iloc[0] > 1e15:
        df['timestamp'] = df['timestamp'].astype(np.float64) / 1e9

    # Sanity check: EuRoC timestamps should be in a plausible Unix range
    t0 = df['timestamp'].iloc[0]
    if not (1e8 < t0 < 2e10):
        raise ValueError(
            f"Timestamp {t0} is outside plausible range after conversion. "
            f"Expected Unix seconds (~1.4e9). Check CSV format in {gt_csv}."
        )

    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def save_groundtruth(df: pd.DataFrame, output_path: str):
    """Save ground truth in project CSV format."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    df.to_csv(output_path, index=False,
              columns=['timestamp', 'px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz'])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Parse EuRoC ground truth')
    parser.add_argument('euroc_dir', help='Path to EuRoC sequence directory')
    parser.add_argument('--output', '-o', default='groundtruth.csv')
    args = parser.parse_args()

    gt = parse_euroc_gt(args.euroc_dir)
    save_groundtruth(gt, args.output)
    print(f"Parsed {len(gt)} ground truth poses -> {args.output}")
