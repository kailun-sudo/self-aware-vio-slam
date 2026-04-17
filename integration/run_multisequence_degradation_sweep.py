#!/usr/bin/env python3
"""
Run a representative multi-sequence EuRoC degradation sweep with baseline reuse.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Dict, List

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
VIO_ROOT = ROOT_DIR / "VIO-SLAM"
SELF_AWARE_ROOT = ROOT_DIR / "self_aware_slam"
INTEGRATION_ROOT = ROOT_DIR / "integration"

SCENARIO_PRESETS: Dict[str, Dict[str, object]] = {
    "blur_bias": {
        "camera_degradation": "motion_blur",
        "imu_degradation": "bias_drift",
        "severity": 0.60,
        "description": "Motion blur with IMU bias drift",
    },
    "noise_amp": {
        "camera_degradation": "gaussian_noise",
        "imu_degradation": "noise_amplification",
        "severity": 0.65,
        "description": "Image noise with amplified IMU noise",
    },
    "lighting_dropout": {
        "camera_degradation": "brightness_change",
        "imu_degradation": None,
        "severity": 0.55,
        "description": "Lighting variation without IMU corruption",
    },
    "dropout_bias": {
        "camera_degradation": "image_dropout",
        "imu_degradation": "bias_drift",
        "severity": 0.50,
        "description": "Image dropout with IMU bias drift",
    },
}


def _run(command: list[str], dry_run: bool):
    print("Running:", " ".join(command))
    if not dry_run:
        env = os.environ.copy()
        env.setdefault("MPLCONFIGDIR", str((ROOT_DIR / ".mplcache").resolve()))
        subprocess.run(command, check=True, env=env)


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


def _normalize_name(sequence: str) -> str:
    mapping = {
        "MH_01_easy": "MH_01",
        "MH_02_easy": "MH_02",
        "MH_03_medium": "MH_03",
        "MH_04_difficult": "MH_04",
        "MH_05_difficult": "MH_05",
        "V1_01_easy": "V1_01",
        "V1_02_medium": "V1_02",
        "V1_03_difficult": "V1_03",
        "V2_01_easy": "V2_01",
        "V2_02_medium": "V2_02",
        "V2_03_difficult": "V2_03",
    }
    return mapping.get(sequence, sequence)


def _severity_slug(severity: float) -> str:
    return f"s{int(round(severity * 100)):02d}"


def _build_scenario_instances(
    scenario_names: List[str],
    severity_grid: List[float],
) -> List[Dict[str, object]]:
    instances: List[Dict[str, object]] = []
    for base_name in scenario_names:
        preset = SCENARIO_PRESETS[base_name]
        severities = severity_grid or [float(preset["severity"])]
        for severity in severities:
            scenario_slug = base_name if not severity_grid else f"{base_name}_{_severity_slug(severity)}"
            instances.append(
                {
                    "base_scenario": base_name,
                    "scenario_slug": scenario_slug,
                    "camera_degradation": preset["camera_degradation"],
                    "imu_degradation": preset.get("imu_degradation"),
                    "severity": float(severity),
                    "description": f"{preset['description']} (severity={float(severity):.2f})",
                }
            )
    return instances


def _ensure_baseline(
    *,
    data_path: Path,
    groundtruth_path: Path,
    baseline_vio_dir: Path,
    baseline_self_aware_dir: Path,
    vio_python: str,
    self_aware_python: str,
    self_aware_config: str,
    checkpoint: str | None,
    dataset_stats: str | None,
    downsample: int | None,
    window_size: int | None,
    orb_features: int | None,
    dry_run: bool,
    force: bool,
):
    baseline_ready = (
        (baseline_vio_dir / "slam_metrics.csv").exists()
        and (baseline_vio_dir / "estimated_tum.txt").exists()
        and (baseline_self_aware_dir / "reliability_predictions.csv").exists()
        and (baseline_self_aware_dir / "summary.txt").exists()
    )
    if baseline_ready and not force:
        print(f"Reusing baseline: {baseline_vio_dir}")
        return

    baseline_vio_dir.mkdir(parents=True, exist_ok=True)
    baseline_self_aware_dir.mkdir(parents=True, exist_ok=True)

    baseline_run = [
        vio_python,
        str(VIO_ROOT / "run_pipeline.py"),
        "--data_path",
        str(data_path),
        "--output",
        str(baseline_vio_dir),
    ]
    if downsample is not None:
        baseline_run.extend(["--downsample", str(downsample)])
    if window_size is not None:
        baseline_run.extend(["--window_size", str(window_size)])
    if orb_features is not None:
        baseline_run.extend(["--orb_features", str(orb_features)])
    _run(baseline_run, dry_run)

    baseline_demo = [
        self_aware_python,
        str(INTEGRATION_ROOT / "run_offline_unified_demo.py"),
        "--metrics",
        str(baseline_vio_dir / "slam_metrics.csv"),
        "--estimated",
        str(baseline_vio_dir / "estimated_tum.txt"),
        "--groundtruth",
        str(groundtruth_path),
        "--output-dir",
        str(baseline_self_aware_dir),
        "--config",
        self_aware_config,
    ]
    if checkpoint:
        baseline_demo.extend(["--checkpoint", checkpoint])
    if dataset_stats:
        baseline_demo.extend(["--dataset-stats", dataset_stats])
    _run(baseline_demo, dry_run)


def _run_scenario(
    *,
    sequence: str,
    scenario_name: str,
    scenario: Dict[str, object],
    data_path: Path,
    groundtruth_path: Path,
    sequence_dir: Path,
    baseline_vio_dir: Path,
    baseline_self_aware_dir: Path,
    vio_python: str,
    self_aware_python: str,
    self_aware_config: str,
    checkpoint: str | None,
    dataset_stats: str | None,
    downsample: int | None,
    window_size: int | None,
    orb_features: int | None,
    dry_run: bool,
    force: bool,
) -> Dict[str, object]:
    scenario_root = sequence_dir / scenario_name
    degraded_vio_dir = scenario_root / "degraded_vio"
    degraded_self_aware_dir = scenario_root / "degraded_self_aware"
    comparison_dir = scenario_root / "comparison"

    degraded_ready = (
        (degraded_vio_dir / "summary.txt").exists()
        and (degraded_self_aware_dir / "summary.txt").exists()
        and (comparison_dir / "gui" / "visual_demo.html").exists()
    )
    if not degraded_ready or force:
        degraded_vio_dir.mkdir(parents=True, exist_ok=True)
        degraded_self_aware_dir.mkdir(parents=True, exist_ok=True)
        comparison_dir.mkdir(parents=True, exist_ok=True)

        degraded_run = [
            vio_python,
            str(VIO_ROOT / "run_pipeline.py"),
            "--data_path",
            str(data_path),
            "--output",
            str(degraded_vio_dir),
            "--simulate_degradation",
            "--camera_degradation",
            str(scenario["camera_degradation"]),
            "--degradation_severity",
            str(scenario["severity"]),
            "--degradation_seed",
            "42",
        ]
        imu_degradation = scenario.get("imu_degradation")
        if imu_degradation:
            degraded_run.extend(["--imu_degradation", str(imu_degradation)])
        if downsample is not None:
            degraded_run.extend(["--downsample", str(downsample)])
        if window_size is not None:
            degraded_run.extend(["--window_size", str(window_size)])
        if orb_features is not None:
            degraded_run.extend(["--orb_features", str(orb_features)])
        _run(degraded_run, dry_run)

        degraded_demo = [
            self_aware_python,
            str(INTEGRATION_ROOT / "run_offline_unified_demo.py"),
            "--metrics",
            str(degraded_vio_dir / "slam_metrics.csv"),
            "--estimated",
            str(degraded_vio_dir / "estimated_tum.txt"),
            "--groundtruth",
            str(groundtruth_path),
            "--output-dir",
            str(degraded_self_aware_dir),
            "--config",
            self_aware_config,
        ]
        if checkpoint:
            degraded_demo.extend(["--checkpoint", checkpoint])
        if dataset_stats:
            degraded_demo.extend(["--dataset-stats", dataset_stats])
        _run(degraded_demo, dry_run)

        comparison_gui = [
            self_aware_python,
            str(INTEGRATION_ROOT / "create_degradation_comparison_demo.py"),
            "--baseline-dir",
            str(baseline_self_aware_dir),
            "--degraded-dir",
            str(degraded_self_aware_dir),
            "--output-dir",
            str(comparison_dir / "gui"),
        ]
        _run(comparison_gui, dry_run)
    else:
        print(f"Reusing scenario artifacts: {sequence} / {scenario_name}")

    if dry_run:
        return {
            "sequence": sequence,
            "scenario": scenario_name,
            "base_scenario": scenario["base_scenario"],
            "camera_degradation": scenario["camera_degradation"],
            "imu_degradation": scenario.get("imu_degradation") or "none",
            "severity": scenario["severity"],
            "description": scenario["description"],
        }

    baseline_vio_summary = _read_summary(baseline_vio_dir / "summary.txt")
    degraded_vio_summary = _read_summary(degraded_vio_dir / "summary.txt")
    baseline_self_summary = _read_summary(baseline_self_aware_dir / "summary.txt")
    degraded_self_summary = _read_summary(degraded_self_aware_dir / "summary.txt")

    row: Dict[str, object] = {
        "sequence": sequence,
        "sequence_short": _normalize_name(sequence),
        "scenario": scenario_name,
        "base_scenario": scenario["base_scenario"],
        "camera_degradation": scenario["camera_degradation"],
        "imu_degradation": scenario.get("imu_degradation") or "none",
        "severity": scenario["severity"],
        "description": scenario["description"],
        "comparison_gui_path": str((comparison_dir / "gui" / "visual_demo.html").resolve()),
        "comparison_summary_path": str((comparison_dir / "gui" / "comparison_gui_summary.txt").resolve()),
    }

    metric_keys = [
        "mean_inlier_ratio",
        "tracking_success_ratio",
        "trajectory_length_m",
        "pose_error_mean",
        "failure_probability_mean",
        "confidence_mean",
    ]
    for key in metric_keys:
        baseline_value = baseline_vio_summary.get(key, baseline_self_summary.get(key))
        degraded_value = degraded_vio_summary.get(key, degraded_self_summary.get(key))
        if baseline_value is not None:
            row[f"baseline_{key}"] = baseline_value
        if degraded_value is not None:
            row[f"degraded_{key}"] = degraded_value
        if baseline_value is not None and degraded_value is not None:
            try:
                row[f"{key}_delta"] = float(degraded_value) - float(baseline_value)
            except (TypeError, ValueError):
                pass

    return row


def main():
    parser = argparse.ArgumentParser(description="Run a representative multi-sequence degradation sweep")
    parser.add_argument("--dataset-root", type=str, default=str(VIO_ROOT / "data" / "sequences"))
    parser.add_argument(
        "--sequences",
        type=str,
        default="MH_01_easy,MH_02_easy,MH_03_medium,MH_04_difficult,MH_05_difficult",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default="blur_bias,noise_amp,lighting_dropout,dropout_bias",
    )
    parser.add_argument(
        "--severity-grid",
        type=str,
        default="",
        help="Optional comma-separated severity override grid, e.g. 0.45,0.65",
    )
    parser.add_argument("--output-root", type=str, default=str(ROOT_DIR / "outputs" / "multisequence_degradation_sweep"))
    parser.add_argument("--vio-python", type=str, default=str(VIO_ROOT / ".venv" / "bin" / "python"))
    parser.add_argument("--self-aware-python", type=str, default=str(SELF_AWARE_ROOT / "venv" / "bin" / "python"))
    parser.add_argument("--self-aware-config", type=str, default=str(SELF_AWARE_ROOT / "configs" / "config.yaml"))
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--dataset-stats", type=str, default=None)
    parser.add_argument("--downsample", type=int, default=120)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--orb-features", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Ignore existing outputs and rerun everything.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    sequences = [item.strip() for item in args.sequences.split(",") if item.strip()]
    scenario_names = [item.strip() for item in args.scenarios.split(",") if item.strip()]
    severity_grid = [float(item.strip()) for item in args.severity_grid.split(",") if item.strip()]

    unknown = [name for name in scenario_names if name not in SCENARIO_PRESETS]
    if unknown:
        raise ValueError(f"Unknown scenarios: {', '.join(unknown)}")
    if severity_grid:
        for value in severity_grid:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Severity must be in [0, 1], got {value}")

    scenario_instances = _build_scenario_instances(scenario_names, severity_grid)

    rows: list[Dict[str, object]] = []
    for sequence in sequences:
        data_path = dataset_root / sequence / "mav0"
        groundtruth_path = data_path / "state_groundtruth_estimate0" / "data.csv"
        if not data_path.exists():
            raise FileNotFoundError(f"Sequence not found: {data_path}")
        if not groundtruth_path.exists():
            raise FileNotFoundError(f"Ground truth not found: {groundtruth_path}")

        sequence_dir = output_root / sequence
        baseline_vio_dir = sequence_dir / "baseline_vio"
        baseline_self_aware_dir = sequence_dir / "baseline_self_aware"

        _ensure_baseline(
            data_path=data_path,
            groundtruth_path=groundtruth_path,
            baseline_vio_dir=baseline_vio_dir,
            baseline_self_aware_dir=baseline_self_aware_dir,
            vio_python=args.vio_python,
            self_aware_python=args.self_aware_python,
            self_aware_config=args.self_aware_config,
            checkpoint=args.checkpoint,
            dataset_stats=args.dataset_stats,
            downsample=args.downsample,
            window_size=args.window_size,
            orb_features=args.orb_features,
            dry_run=args.dry_run,
            force=args.force,
        )

        for scenario in scenario_instances:
            scenario_name = str(scenario["scenario_slug"])
            row = _run_scenario(
                sequence=sequence,
                scenario_name=scenario_name,
                scenario=scenario,
                data_path=data_path,
                groundtruth_path=groundtruth_path,
                sequence_dir=sequence_dir,
                baseline_vio_dir=baseline_vio_dir,
                baseline_self_aware_dir=baseline_self_aware_dir,
                vio_python=args.vio_python,
                self_aware_python=args.self_aware_python,
                self_aware_config=args.self_aware_config,
                checkpoint=args.checkpoint,
                dataset_stats=args.dataset_stats,
                downsample=args.downsample,
                window_size=args.window_size,
                orb_features=args.orb_features,
                dry_run=args.dry_run,
                force=args.force,
            )
            rows.append(row)

    results_csv = output_root / "sweep_results.csv"
    pd.DataFrame(rows).to_csv(results_csv, index=False)
    print(f"sweep_results_csv: {results_csv}")

    report_cmd = [
        args.self_aware_python,
        str(INTEGRATION_ROOT / "create_multisequence_degradation_report.py"),
        "--results-csv",
        str(results_csv),
        "--output-dir",
        str(output_root / "report"),
    ]
    _run(report_cmd, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
