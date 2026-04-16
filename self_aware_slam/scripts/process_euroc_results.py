#!/usr/bin/env python3
"""
Batch-process all EuRoC sequences from VINS-Fusion output.

Expected directory layout for VINS-Fusion results:

    vins_output/
      MH_01_easy/
        vins_result_no_loop.csv   (or vins_result_loop.csv)
        vins_log.txt              (optional, stdout capture)
      MH_02_easy/
        ...

Expected EuRoC dataset layout:

    euroc_dataset/
      MH_01_easy/
        mav0/state_groundtruth_estimate0/data.csv
      MH_02_easy/
        ...

Usage:
    python scripts/process_euroc_results.py \\
        --euroc-dir /data/euroc \\
        --vins-dir /data/vins_output \\
        --output-dir slam_metrics_dataset

    # Or process a single sequence:
    python scripts/process_euroc_results.py \\
        --euroc-dir /data/euroc \\
        --vins-dir /data/vins_output \\
        --sequences MH_01_easy
"""

import os
import sys
import argparse
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.euroc.process_sequence import (
    process_sequence,
    auto_detect_sequence_name,
)


# VINS trajectory file names to search for (in order of preference)
VINS_TRAJ_NAMES = [
    'vins_result_no_loop.csv',
    'vins_result_loop.csv',
    'vins_result.csv',
    'stamped_traj_estimate.txt',  # evo format
    'trajectory.txt',
]

VINS_LOG_NAMES = [
    'vins_log.txt',
    'vins_output.log',
    'output.log',
]


def find_vins_trajectory(vins_seq_dir: str) -> str:
    """Find VINS trajectory file in a sequence output directory."""
    for name in VINS_TRAJ_NAMES:
        path = os.path.join(vins_seq_dir, name)
        if os.path.isfile(path):
            return path
    # Try any CSV in the directory
    csvs = glob.glob(os.path.join(vins_seq_dir, '*.csv'))
    if csvs:
        return csvs[0]
    raise FileNotFoundError(
        f"No VINS trajectory file found in {vins_seq_dir}. "
        f"Looked for: {VINS_TRAJ_NAMES}")


def find_vins_log(vins_seq_dir: str) -> str:
    """Find VINS log file (optional)."""
    for name in VINS_LOG_NAMES:
        path = os.path.join(vins_seq_dir, name)
        if os.path.isfile(path):
            return path
    return None


def discover_sequences(euroc_dir: str, vins_dir: str):
    """Discover available sequences by finding directories present in both
    the EuRoC dataset and VINS output directories."""
    euroc_seqs = set()
    vins_seqs = set()

    for d in os.listdir(euroc_dir):
        if os.path.isdir(os.path.join(euroc_dir, d)):
            euroc_seqs.add(d)

    for d in os.listdir(vins_dir):
        if os.path.isdir(os.path.join(vins_dir, d)):
            vins_seqs.add(d)

    common = euroc_seqs & vins_seqs
    if not common:
        print(f"Warning: No matching sequence directories found.")
        print(f"  EuRoC dirs: {sorted(euroc_seqs)}")
        print(f"  VINS dirs:  {sorted(vins_seqs)}")

    return sorted(common)


def main():
    parser = argparse.ArgumentParser(
        description='Batch process EuRoC + VINS-Fusion results')
    parser.add_argument('--euroc-dir', required=True,
                        help='Root directory of EuRoC dataset')
    parser.add_argument('--vins-dir', required=True,
                        help='Root directory of VINS-Fusion output')
    parser.add_argument('--output-dir', default='slam_metrics_dataset',
                        help='Output directory for processed data')
    parser.add_argument('--sequences', nargs='*', default=None,
                        help='Specific sequences to process (default: all)')
    parser.add_argument('--max-diff', type=float, default=0.02,
                        help='Max timestamp diff for GT association')
    args = parser.parse_args()

    if args.sequences:
        sequences = args.sequences
    else:
        sequences = discover_sequences(args.euroc_dir, args.vins_dir)

    if not sequences:
        print("No sequences to process. Exiting.")
        return

    print(f"Processing {len(sequences)} sequences: {sequences}")
    print(f"Output: {args.output_dir}/\n")

    results = {}
    for seq_dir_name in sequences:
        euroc_path = os.path.join(args.euroc_dir, seq_dir_name)
        vins_path = os.path.join(args.vins_dir, seq_dir_name)

        if not os.path.isdir(euroc_path):
            print(f"Skipping {seq_dir_name}: EuRoC dir not found at {euroc_path}")
            continue
        if not os.path.isdir(vins_path):
            print(f"Skipping {seq_dir_name}: VINS dir not found at {vins_path}")
            continue

        try:
            traj_file = find_vins_trajectory(vins_path)
            log_file = find_vins_log(vins_path)
            seq_name = auto_detect_sequence_name(seq_dir_name)

            out = process_sequence(
                euroc_seq_dir=euroc_path,
                vins_trajectory_path=traj_file,
                sequence_name=seq_name,
                output_base_dir=args.output_dir,
                vins_log_path=log_file,
                max_assoc_diff=args.max_diff,
            )
            results[seq_name] = 'OK'
        except Exception as e:
            print(f"  ERROR processing {seq_dir_name}: {e}")
            results[seq_dir_name] = f'FAILED: {e}'

    print("\n" + "=" * 50)
    print("Batch processing complete")
    print("=" * 50)
    for name, status in results.items():
        print(f"  {name}: {status}")

    n_ok = sum(1 for s in results.values() if s == 'OK')
    print(f"\n{n_ok}/{len(results)} sequences processed successfully.")
    if n_ok > 0:
        print(f"\nNext step: run the pipeline with --skip-datagen")
        print(f"  python run_pipeline.py --skip-datagen")


if __name__ == '__main__':
    main()
