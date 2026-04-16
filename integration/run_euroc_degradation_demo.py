#!/usr/bin/env python3
"""
Run a baseline vs degraded EuRoC playback comparison for Self-Aware VIO-SLAM.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
VIO_ROOT = ROOT_DIR / "VIO-SLAM"
SELF_AWARE_ROOT = ROOT_DIR / "self_aware_slam"


def _run(command: list[str]):
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def _read_summary(path: Path) -> Dict[str, object]:
    values: Dict[str, object] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if ":" not in line:
                continue
            key, raw_value = line.strip().split(":", 1)
            key = key.strip()
            raw_value = raw_value.strip()
            if raw_value.lower() in {"true", "false"}:
                values[key] = raw_value.lower() == "true"
                continue
            try:
                if "." in raw_value or "e" in raw_value.lower():
                    values[key] = float(raw_value)
                else:
                    values[key] = int(raw_value)
            except ValueError:
                values[key] = raw_value
    return values


def _build_groundtruth_path(data_path: Path) -> Path:
    return data_path / "state_groundtruth_estimate0" / "data.csv"


def _plot_comparison(frame: pd.DataFrame, output_path: Path):
    metrics = [
        "mean_inlier_ratio",
        "tracking_success_ratio",
        "pose_error_mean",
        "failure_probability_mean",
        "confidence_mean",
    ]
    available = [metric for metric in metrics if metric in frame.columns]
    if not available:
        return

    fig, axes = plt.subplots(len(available), 1, figsize=(10, 3.2 * len(available)))
    if len(available) == 1:
        axes = [axes]

    for axis, metric in zip(axes, available):
        axis.bar(frame["scenario"], frame[metric], color=["tab:blue", "tab:red"])
        axis.set_title(metric)
        axis.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run EuRoC playback + degradation injection demo")
    parser.add_argument("--data-path", type=str, default=None, help="EuRoC mav0 directory")
    parser.add_argument("--output-root", type=str, default=str(ROOT_DIR / "outputs" / "euroc_degradation_demo"))
    parser.add_argument("--camera-degradation", type=str, default="motion_blur",
                        choices=["motion_blur", "gaussian_noise", "brightness_change", "image_dropout"])
    parser.add_argument("--imu-degradation", type=str, default=None,
                        choices=["bias_drift", "noise_amplification"])
    parser.add_argument("--severity", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vio-python", type=str, default=str(VIO_ROOT / ".venv" / "bin" / "python"))
    parser.add_argument("--self-aware-python", type=str, default=str(SELF_AWARE_ROOT / "venv" / "bin" / "python"))
    parser.add_argument("--self-aware-config", type=str, default=str(SELF_AWARE_ROOT / "configs" / "config.yaml"))
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--dataset-stats", type=str, default=None)
    parser.add_argument("--downsample", type=int, default=None, help="Optional override for VIO playback downsample factor.")
    parser.add_argument("--window-size", type=int, default=None, help="Optional override for VIO sliding window size.")
    parser.add_argument("--orb-features", type=int, default=None, help="Optional override for ORB feature count.")
    args = parser.parse_args()

    data_path = Path(args.data_path).expanduser().resolve() if args.data_path else (VIO_ROOT / "data" / "mav0").resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"EuRoC mav0 not found: {data_path}")

    groundtruth_path = _build_groundtruth_path(data_path)
    if not groundtruth_path.exists():
        raise FileNotFoundError(f"Ground truth not found: {groundtruth_path}")

    output_root = Path(args.output_root).expanduser().resolve()
    baseline_vio_dir = output_root / "baseline_vio"
    degraded_vio_dir = output_root / "degraded_vio"
    baseline_self_aware_dir = output_root / "baseline_self_aware"
    degraded_self_aware_dir = output_root / "degraded_self_aware"
    comparison_dir = output_root / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    baseline_run = [
        args.vio_python,
        str(VIO_ROOT / "run_pipeline.py"),
        "--data_path",
        str(data_path),
        "--output",
        str(baseline_vio_dir),
    ]
    degraded_run = [
        args.vio_python,
        str(VIO_ROOT / "run_pipeline.py"),
        "--data_path",
        str(data_path),
        "--output",
        str(degraded_vio_dir),
        "--simulate_degradation",
        "--camera_degradation",
        args.camera_degradation,
        "--degradation_severity",
        str(args.severity),
        "--degradation_seed",
        str(args.seed),
    ]
    if args.downsample is not None:
        baseline_run.extend(["--downsample", str(args.downsample)])
        degraded_run.extend(["--downsample", str(args.downsample)])
    if args.window_size is not None:
        baseline_run.extend(["--window_size", str(args.window_size)])
        degraded_run.extend(["--window_size", str(args.window_size)])
    if args.orb_features is not None:
        baseline_run.extend(["--orb_features", str(args.orb_features)])
        degraded_run.extend(["--orb_features", str(args.orb_features)])
    if args.imu_degradation:
        degraded_run.extend(["--imu_degradation", args.imu_degradation])

    _run(baseline_run)
    _run(degraded_run)

    common_self_aware_args = [
        "--config",
        args.self_aware_config,
    ]
    if args.checkpoint:
        common_self_aware_args.extend(["--checkpoint", args.checkpoint])
    if args.dataset_stats:
        common_self_aware_args.extend(["--dataset-stats", args.dataset_stats])

    baseline_demo = [
        args.self_aware_python,
        str(ROOT_DIR / "integration" / "run_offline_unified_demo.py"),
        "--metrics",
        str(baseline_vio_dir / "slam_metrics.csv"),
        "--estimated",
        str(baseline_vio_dir / "estimated_tum.txt"),
        "--groundtruth",
        str(groundtruth_path),
        "--output-dir",
        str(baseline_self_aware_dir),
        *common_self_aware_args,
    ]
    degraded_demo = [
        args.self_aware_python,
        str(ROOT_DIR / "integration" / "run_offline_unified_demo.py"),
        "--metrics",
        str(degraded_vio_dir / "slam_metrics.csv"),
        "--estimated",
        str(degraded_vio_dir / "estimated_tum.txt"),
        "--groundtruth",
        str(groundtruth_path),
        "--output-dir",
        str(degraded_self_aware_dir),
        *common_self_aware_args,
    ]

    _run(baseline_demo)
    _run(degraded_demo)

    baseline_vio_summary = _read_summary(baseline_vio_dir / "summary.txt")
    degraded_vio_summary = _read_summary(degraded_vio_dir / "summary.txt")
    baseline_self_summary = _read_summary(baseline_self_aware_dir / "summary.txt")
    degraded_self_summary = _read_summary(degraded_self_aware_dir / "summary.txt")

    comparison = pd.DataFrame([
        {
            "scenario": "baseline",
            **baseline_vio_summary,
            **baseline_self_summary,
        },
        {
            "scenario": "degraded",
            **degraded_vio_summary,
            **degraded_self_summary,
            "camera_degradation": args.camera_degradation,
            "imu_degradation": args.imu_degradation or "none",
            "degradation_severity": args.severity,
        },
    ])
    comparison_csv = comparison_dir / "comparison_metrics.csv"
    comparison.to_csv(comparison_csv, index=False)

    summary_txt = comparison_dir / "comparison_summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as handle:
        handle.write("EuRoC degradation playback comparison\n")
        handle.write(f"data_path: {data_path}\n")
        handle.write(f"camera_degradation: {args.camera_degradation}\n")
        handle.write(f"imu_degradation: {args.imu_degradation or 'none'}\n")
        handle.write(f"degradation_severity: {args.severity}\n\n")

        for metric in [
            "mean_inlier_ratio",
            "tracking_success_ratio",
            "trajectory_length_m",
            "pose_error_mean",
            "failure_probability_mean",
            "confidence_mean",
        ]:
            if metric not in comparison.columns:
                continue
            baseline_value = comparison.loc[comparison["scenario"] == "baseline", metric].iloc[0]
            degraded_value = comparison.loc[comparison["scenario"] == "degraded", metric].iloc[0]
            handle.write(f"{metric}: baseline={baseline_value} degraded={degraded_value}\n")

    _plot_comparison(comparison, comparison_dir / "comparison_overview.png")

    comparison_gui = [
        args.self_aware_python,
        str(ROOT_DIR / "integration" / "create_degradation_comparison_demo.py"),
        "--baseline-dir",
        str(baseline_self_aware_dir),
        "--degraded-dir",
        str(degraded_self_aware_dir),
        "--output-dir",
        str(comparison_dir / "gui"),
    ]
    _run(comparison_gui)

    print(f"baseline_vio_dir: {baseline_vio_dir}")
    print(f"degraded_vio_dir: {degraded_vio_dir}")
    print(f"baseline_self_aware_dir: {baseline_self_aware_dir}")
    print(f"degraded_self_aware_dir: {degraded_self_aware_dir}")
    print(f"comparison_summary: {summary_txt}")
    print(f"comparison_csv: {comparison_csv}")
    print(f"comparison_gui: {comparison_dir / 'gui' / 'visual_demo.html'}")


if __name__ == "__main__":
    raise SystemExit(main())
