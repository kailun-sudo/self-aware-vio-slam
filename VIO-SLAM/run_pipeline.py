#!/usr/bin/env python3
"""
Primary command-line entrypoint for the notebook-derived VIO-SLAM pipeline.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from vio_pipeline import (
    NotebookDerivedVIOPipeline,
    OnlineSelfAwareBridge,
    SensorDegradationConfig,
    load_config,
    resolve_data_path,
)


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run the primary notebook-derived VIO-SLAM pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="EuRoC mav0 directory. If omitted, the runner will look for data/mav0 automatically.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(root / "config" / "default.yaml"),
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(root.parent / "outputs" / "vio_pipeline"),
        help="Output directory for trajectory and metrics files.",
    )
    parser.add_argument("--downsample", type=int, default=None, help="Optional override for downsample factor.")
    parser.add_argument("--window_size", type=int, default=None, help="Optional override for sliding-window size.")
    parser.add_argument("--orb_features", type=int, default=None, help="Optional override for ORB feature count.")
    parser.add_argument("--save_plot", action="store_true", help="Save an XY trajectory plot.")
    parser.add_argument("--show_plot", action="store_true", help="Display the XY trajectory plot.")
    parser.add_argument("--enable_online_self_aware", action="store_true",
                        help="Enable online self-aware prediction during the VIO loop.")
    parser.add_argument("--self_aware_python", type=str,
                        default=str(root.parent / "self_aware_slam" / "venv" / "bin" / "python"),
                        help="Python executable for the self-aware runtime sidecar.")
    parser.add_argument("--self_aware_config", type=str,
                        default=str(root.parent / "self_aware_slam" / "configs" / "config.yaml"),
                        help="Config file for self-aware inference.")
    parser.add_argument("--self_aware_checkpoint", type=str, default=None,
                        help="Optional trained model checkpoint path.")
    parser.add_argument("--self_aware_dataset_stats", type=str, default=None,
                        help="Optional train_dataset.pkl path with normalization stats.")
    parser.add_argument("--online_predictions_output", type=str, default=None,
                        help="Optional output path for online self-aware predictions CSV.")
    parser.add_argument("--simulate_degradation", action="store_true",
                        help="Inject deterministic camera/IMU degradation during EuRoC playback.")
    parser.add_argument("--camera_degradation", type=str, default=None,
                        choices=["motion_blur", "gaussian_noise", "brightness_change", "image_dropout"],
                        help="Camera degradation mode for simulation playback.")
    parser.add_argument("--imu_degradation", type=str, default=None,
                        choices=["bias_drift", "noise_amplification"],
                        help="IMU degradation mode for simulation playback.")
    parser.add_argument("--degradation_severity", type=float, default=0.5,
                        help="Degradation severity in [0, 1].")
    parser.add_argument("--degradation_seed", type=int, default=42,
                        help="Random seed used to deterministically replay degraded observations.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    root = Path(__file__).resolve().parent

    data_path = resolve_data_path(args.data_path)
    if data_path is None:
        logger.error(
            "No valid EuRoC mav0 directory was found. Pass --data_path explicitly or place the dataset at VIO-SLAM/data/mav0."
        )
        return 1

    config = load_config(args.config)
    if args.downsample is not None:
        config["dataset"]["downsample_factor"] = args.downsample
    if args.window_size is not None:
        config["slam"]["window_size"] = args.window_size
    if args.orb_features is not None:
        config["slam"]["orb_features"] = args.orb_features
    config["visualization"]["save_plot"] = bool(args.save_plot)
    config["visualization"]["show_trajectory"] = bool(args.show_plot)

    output_dir = Path(args.output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Using dataset: %s", data_path)
    logger.info("Writing outputs to: %s", output_dir)

    online_predictor = None
    if args.enable_online_self_aware:
        inference_script = root.parent / "self_aware_slam" / "src" / "models" / "inference.py"
        online_predictor = OnlineSelfAwareBridge(
            python_executable=args.self_aware_python,
            inference_script=str(inference_script),
            config_path=args.self_aware_config,
            checkpoint_path=args.self_aware_checkpoint,
            dataset_stats_path=args.self_aware_dataset_stats,
            verbose=args.verbose,
        )
        logger.info("Online self-aware predictor enabled via %s", args.self_aware_python)

    degradation_config = None
    if args.simulate_degradation:
        if args.camera_degradation is None and args.imu_degradation is None:
            parser.error("--simulate_degradation requires --camera_degradation and/or --imu_degradation")
        degradation_config = SensorDegradationConfig(
            camera_degradation=args.camera_degradation,
            imu_degradation=args.imu_degradation,
            severity=args.degradation_severity,
            seed=args.degradation_seed,
        )
        logger.info(
            "Simulation mode enabled: camera=%s imu=%s severity=%.2f seed=%s",
            args.camera_degradation or "none",
            args.imu_degradation or "none",
            args.degradation_severity,
            args.degradation_seed,
        )

    pipeline = NotebookDerivedVIOPipeline(
        config,
        online_predictor=online_predictor,
        degradation_config=degradation_config,
    )
    trajectory = pipeline.run(str(data_path))

    pipeline.save_metrics_csv(str(output_dir / "slam_metrics.csv"))
    pipeline.save_tum_trajectory(str(output_dir / "estimated_tum.txt"))
    pipeline.save_trajectory_text(str(output_dir / "trajectory.txt"))
    pipeline.save_trajectory_pickle(str(output_dir / "trajectory.pkl"))
    if args.enable_online_self_aware:
        online_output = args.online_predictions_output or str(output_dir / "online_predictions.csv")
        pipeline.save_online_predictions_csv(online_output)

    if args.save_plot or args.show_plot:
        pipeline.save_plot(
            output_path=str(output_dir / "trajectory_xy.png"),
            show_plot=args.show_plot,
        )

    stats = pipeline.get_statistics()
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("Notebook-derived VIO-SLAM summary\n")
        for key, value in stats.items():
            handle.write(f"{key}: {value}\n")

    logger.info("Pipeline finished with %d poses", len(trajectory))
    for key, value in stats.items():
        logger.info("  %s: %s", key, value)
    logger.info("Generated files:")
    for name in [
        "slam_metrics.csv",
        "estimated_tum.txt",
        "trajectory.txt",
        "trajectory.pkl",
        "summary.txt",
    ]:
        logger.info("  %s", output_dir / name)
    if args.enable_online_self_aware:
        logger.info("  %s", args.online_predictions_output or output_dir / "online_predictions.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
